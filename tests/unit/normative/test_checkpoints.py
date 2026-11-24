"""Expected checkpoint preparation and admission behavior."""

from collections import OrderedDict
import json

import torch

import checkpoints
from checkpoint_fixtures import CheckpointCase, TrainingMetadata


class CheckpointNormativeTests(CheckpointCase):
    def test_receipt_covers_checkpoint_bundle_without_rewriting(self):
        weights, config = self.make_bundle()
        inode = weights.stat().st_ino
        before = self.digest(weights)

        with checkpoints.verified_weights(weights, companions=(config,)) as streams:
            loaded = torch.load(streams[weights], map_location="cpu", weights_only=True)
            self.assertEqual(loaded["metadata"], {"fixture": "epoch1"})
            self.assertEqual(streams[config].read().decode(), "fixture:\n  name: epoch1\n")

        receipt = json.loads(checkpoints.marker_path(weights).read_text())
        self.assertEqual(receipt["policy"], checkpoints.POLICY)
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
        with checkpoints.verified_weights(selected, companions=(config,)) as streams:
            admitted = torch.load(streams[selected], weights_only=True)
        self.assertEqual(admitted["metadata"], {"fixture": "selected"})

        with self.assertRaisesRegex(FileNotFoundError, "sanitization receipt is missing"):
            with checkpoints.verified_weights(unselected):
                self.fail("unmarked weights must not be admitted")

    def test_load_checkpoint_loads_selected_model_beside_unmarked_decoy(self):
        directory = self.directory / "selected"
        directory.mkdir()
        weights = directory / "weights.pt"
        model = torch.nn.Linear(2, 1)
        expected = model.state_dict()
        self.save(weights, expected)
        self.save(directory / "unselected-decoy.pt", {"unapproved": TrainingMetadata([])})
        source_hash = self.digest(weights)
        checkpoints.sanitize(weights, source_hash)

        loaded = torch.nn.Linear(2, 1)
        checkpoints.load_checkpoint(loaded, weights)

        for name, tensor in loaded.state_dict().items():
            self.assertTrue(torch.equal(tensor, expected[name]))

    def test_moldetr_extracts_and_casts_state_while_preserving_aliases_and_metadata(self):
        weights, source_hash, output_hash, definitions, inert_global = self.mol_source_and_outputs()

        checkpoints.sanitize(
            weights,
            source_hash,
            output_definitions=definitions,
            precision="fp16",
            extract="model",
            inert_globals=(inert_global,),
        )

        self.assertEqual(self.digest(weights), output_hash)
        with checkpoints.verified_weights(weights) as streams:
            state = torch.load(streams[weights], map_location="cpu", weights_only=True)
        self.assertEqual(set(state), {"first", "alias", "integer_buffer"})
        self.assertEqual(state["first"].dtype, torch.float16)
        self.assertEqual(state["integer_buffer"].dtype, torch.int64)
        self.assertEqual(state["first"].data_ptr(), state["alias"].data_ptr())
        self.assertEqual(state._metadata, OrderedDict((("", {"version": 3}),)))
        self.assertEqual(json.loads(checkpoints.marker_path(weights).read_text())["precision"], "fp16")
