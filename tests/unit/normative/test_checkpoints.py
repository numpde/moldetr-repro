"""Expected checkpoint preparation and admission behavior."""

from collections import OrderedDict
import json
import subprocess
import sys

import torch

import sanitize
from checkpoint_fixtures import CheckpointCase


class CheckpointNormativeTests(CheckpointCase):
    def test_receipt_covers_checkpoint_bundle_without_rewriting(self):
        weights, config = self.make_bundle()
        inode = weights.stat().st_ino
        before = self.digest(weights)

        with sanitize.verified_files(weights, companions=(config,)) as streams:
            loaded = torch.load(streams[weights], map_location="cpu", weights_only=True)
            self.assertEqual(loaded["metadata"], {"fixture": "epoch1"})
            self.assertEqual(streams[config].read().decode(), "fixture:\n  name: epoch1\n")

        receipt = json.loads(sanitize.marker_path(weights).read_text())
        self.assertEqual(receipt["policy"], sanitize.POLICY)
        self.assertEqual(receipt["files"], {
            "model.pt": before,
            "config.yaml": self.digest(config),
        })
        self.assertEqual(weights.stat().st_ino, inode)

    def test_admission_checks_only_the_selected_checkpoint(self):
        selected, _ = self.make_bundle(tag="selected")
        unselected = self.directory / "not-selected" / "model.pt"
        self.save(unselected, {"weights": {"weight": torch.tensor([9.0])}})

        config = selected.with_name("config.yaml")
        with sanitize.verified_files(selected, companions=(config,)) as streams:
            admitted = torch.load(streams[selected], weights_only=True)
        self.assertEqual(admitted["metadata"], {"fixture": "selected"})

        with self.assertRaisesRegex(FileNotFoundError, "sanitization receipt is missing"):
            with sanitize.verified_files(unselected):
                self.fail("unmarked weights must not be admitted")

    def test_moldetr_extracts_and_casts_state_while_preserving_aliases_and_metadata(self):
        weights, source_hash, inert_global = self.mol_source()

        sanitize.sanitize(
            weights,
            source_hash,
            precision="fp16",
            extract="model",
            inert_globals=(inert_global,),
        )

        with sanitize.verified_files(weights) as streams:
            state = torch.load(streams[weights], map_location="cpu", weights_only=True)
        self.assertEqual(set(state), {"first", "alias", "integer_buffer"})
        self.assertEqual(state["first"].dtype, torch.float16)
        self.assertEqual(state["integer_buffer"].dtype, torch.int64)
        self.assertEqual(state["first"].data_ptr(), state["alias"].data_ptr())
        self.assertEqual(state._metadata, OrderedDict((("", {"version": 3}),)))
        self.assertEqual(json.loads(sanitize.marker_path(weights).read_text())["precision"], "fp16")

    def test_repeated_sanitization_verifies_and_cleans_partials(self):
        weights, source_hash, inert_global = self.mol_source()
        sanitize.sanitize(weights, source_hash, precision="fp16", extract="model",
                             inert_globals=(inert_global,))
        self.directory.chmod(0o755)
        partials = [weights.with_name(weights.name + ".part"),
                    weights.with_name(weights.name + ".sanitize.part"),
                    sanitize.marker_path(weights).with_suffix(".part")]
        for partial in partials:
            partial.write_bytes(b"leftover")
        before = self.digest(weights)
        sanitize.sanitize(weights, source_hash, precision="fp16", extract="model")
        self.assertEqual(self.digest(weights), before)
        self.assertTrue(all(not partial.exists() for partial in partials))

    def test_verify_cli_needs_only_standard_library(self):
        weights, config = self.make_bundle()
        manifest = self.directory / "checkpoints.sha256"
        manifest.write_text("".join(f"{self.digest(path)} {path.stat().st_size} {path.name}\n"
                                    for path in (weights, config)))
        result = subprocess.run(
            [sys.executable, "-S", sanitize.__file__, "--verify", "--directory", str(weights.parent),
             "--manifest", str(manifest), "--pattern", "model.pt", "--companion", "config.yaml"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified", result.stdout)
