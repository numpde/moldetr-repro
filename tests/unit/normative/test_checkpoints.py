"""Expected checkpoint preparation and verification behavior."""

from collections import OrderedDict
import json
import subprocess
import sys

import torch

import sanitize
from checkpoint_fixtures import CheckpointCase


class CheckpointNormativeTests(CheckpointCase):
    def test_receipt_covers_checkpoint_bundle_without_rewriting(self):
        (weights, config) = self.make_bundle()
        inode = weights.stat().st_ino
        before = self.digest(weights)

        with sanitize.verified_files(weights, companions=(config,)) as streams:
            loaded = torch.load(streams[weights], map_location="cpu", weights_only=True)

            self.assertEqual(first={"fixture": "epoch1"}, second=loaded["metadata"])
            self.assertEqual(first="fixture:\n" "  name: epoch1\n", second=streams[config].read().decode())

        receipt = json.loads(sanitize.marker_path(weights).read_text())

        self.assertEqual(first=sanitize.POLICY, second=receipt["policy"])

        self.assertEqual(
            first={
                "model.pt": before,
                "config.yaml": self.digest(config),
            },
            second=receipt["files"],
        )

        self.assertEqual(first=inode, second=weights.stat().st_ino)

    def test_verification_checks_only_the_selected_checkpoint(self):
        (selected, _) = self.make_bundle(tag="selected")
        unselected = self.directory / "not-selected" / "model.pt"

        self.save(unselected, {"weights": {"weight": torch.tensor([9.0])}})

        config = selected.with_name("config.yaml")

        with sanitize.verified_files(selected, companions=(config,)) as streams:
            loaded_checkpoint = torch.load(streams[selected], weights_only=True)

        self.assertEqual(first={"fixture": "selected"}, second=loaded_checkpoint["metadata"])

        with self.assertRaisesRegex(FileNotFoundError, "sanitization receipt is missing"):
            with sanitize.verified_files(unselected):
                self.fail("unmarked weights must not be loaded")

    def test_moldetr_extracts_and_casts_state_while_preserving_aliases_and_metadata(self):
        (weights, source_hash, inert_global) = self.mol_source()

        sanitize.sanitize(
            weights,
            source_hash,
            precision="fp16",
            extract="model",
            inert_globals=(inert_global,),
        )

        with sanitize.verified_files(weights) as streams:
            state = torch.load(streams[weights], map_location="cpu", weights_only=True)

        self.assertEqual(first={"first", "alias", "integer_buffer"}, second=set(state))
        self.assertEqual(first=torch.float16, second=state["first"].dtype)
        self.assertEqual(first=torch.int64, second=state["integer_buffer"].dtype)
        self.assertEqual(first=state["alias"].data_ptr(), second=state["first"].data_ptr())
        self.assertEqual(first=OrderedDict((("", {"version": 3}),)), second=state._metadata)
        self.assertEqual(first="fp16", second=json.loads(sanitize.marker_path(weights).read_text())["precision"])

    def test_repeated_sanitization_verifies_and_cleans_partials(self):
        (weights, source_hash, inert_global) = self.mol_source()

        sanitize.sanitize(
            weights,
            source_hash,
            precision="fp16",
            extract="model",
            inert_globals=(inert_global,)
        )

        self.directory.chmod(0o755)

        partials = [
            weights.with_name(weights.name + ".part"),
            weights.with_name(weights.name + ".sanitize.part"),
            sanitize.marker_path(weights).with_suffix(".part")
        ]

        for partial in partials:
            partial.write_bytes(b"leftover")

        before = self.digest(weights)

        sanitize.sanitize(weights, source_hash, precision="fp16", extract="model")
        self.assertEqual(first=before, second=self.digest(weights))
        self.assertTrue(all(not partial.exists() for partial in partials))

    def test_verify_cli_needs_only_standard_library(self):
        (weights, config) = self.make_bundle()
        manifest = self.directory / "checkpoints.sha256"

        manifest.write_text(
            "".join(
                f"{self.digest(checkpoint_path)} {checkpoint_path.stat().st_size} {checkpoint_path.name}\n"  # machine-string: Checkpoint manifest row consumed by the verification CLI.
                for checkpoint_path in (weights, config)
            )
        )

        verification_result = subprocess.run(
            [
                sys.executable,
                "-S",
                sanitize.__file__,
                "--verify",
                "--directory",
                str(weights.parent),
                "--manifest",
                str(manifest),
                "--pattern",
                "model.pt",
                "--companion",
                "config.yaml"
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(first=0, second=verification_result.returncode, msg=verification_result.stderr)
        self.assertIn("Verified", verification_result.stdout)
