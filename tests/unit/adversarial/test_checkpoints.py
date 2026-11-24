"""Checkpoint refusal, interruption, and replacement boundaries."""

from collections import OrderedDict
import pathlib
from unittest.mock import patch

import torch

import sanitize
from checkpoint_fixtures import CheckpointCase, TrainingMetadata


class CheckpointAdversarialTests(CheckpointCase):
    def test_unapproved_global_fails_with_restricted_loader_only(self):
        weights = self.directory / "unapproved.pth"

        self.save(weights, {"metadata": TrainingMetadata(["not listed"])})
        source_hash = self.digest(weights)
        real_load = torch.load
        calls = []

        def record_load(*args, **kwargs):
            calls.append(kwargs.get("weights_only"))

            return real_load(*args, **kwargs)

        with patch.object(torch, "load", side_effect=record_load):
            with self.assertRaises(Exception):
                sanitize.sanitize(weights, source_hash)

        self.assertTrue(calls)
        self.assertTrue(all((weights_only is True) for weights_only in calls))
        self.assertFalse(sanitize.marker_path(weights).exists())

    def test_prepublication_failure_keeps_old_valid_receipt(self):
        weights = self.directory / "weights.pth"
        source = OrderedDict((("weight", torch.tensor([1.0], dtype=torch.float32)),))

        self.save(weights, source)
        source_hash = self.digest(weights)

        sanitize.sanitize(
            weights,
            source_hash,
        )

        marker = sanitize.marker_path(weights)
        old_receipt = marker.read_bytes()

        with patch.object(torch, "save", side_effect=OSError("injected serialization failure")):
            with self.assertRaisesRegex(OSError, "serialization failure"):
                sanitize.sanitize(weights, source_hash, precision="fp16")

        self.assertEqual(first=old_receipt, second=marker.read_bytes())

        with sanitize.verified_files(weights) as streams:
            loaded_checkpoint = torch.load(streams[weights], weights_only=True)

        self.assertEqual(first=torch.float32, second=loaded_checkpoint["weight"].dtype)

    def test_postreplacement_failure_requires_source_restore(self):
        (weights, source_hash, inert_global) = self.mol_source()
        marker = sanitize.marker_path(weights)
        original_replace = pathlib.Path.replace

        def fail_receipt_replace(opened_path, target):
            if not (pathlib.Path(target) != marker):
                raise OSError(f"injected receipt publication failure for {target=}")

            return original_replace(opened_path, target)

        with patch.object(pathlib.Path, "replace", fail_receipt_replace):
            with self.assertRaisesRegex(OSError, "receipt publication failure"):
                sanitize.sanitize(
                    weights,
                    source_hash,
                    precision="fp16",
                    extract="model",
                    inert_globals=(inert_global,),
                )

        self.assertFalse(marker.exists())
        self.assertNotEqual(self.digest(weights), source_hash)

        with self.assertRaises(FileNotFoundError):
            with sanitize.verified_files(weights):
                self.fail("unmarked replacement weights must remain unavailable")

        with self.assertRaisesRegex(ValueError, "restore the pinned source"):
            sanitize.sanitize(weights, source_hash, precision="fp16", extract="model")

        self.assertFalse(marker.exists())
        self.assertFalse(weights.with_name(weights.name + ".sanitize.part").exists())

    def test_marker_invalidation_during_opening_rejects_mixed_generation(self):
        (weights, config) = self.make_bundle()
        marker = sanitize.marker_path(weights)
        replacement = self.directory / "replacement.pt"

        self.save(replacement, {"generation": 2})
        weights.parent.chmod(0o755)
        original_open = pathlib.Path.open
        replaced = False

        def replace_after_payload_open(opened_path, *args, **kwargs):
            nonlocal replaced
            stream = original_open(opened_path, *args, **kwargs)

            if (pathlib.Path(opened_path) == weights) and (not replaced):
                replaced = True

                marker.unlink()
                replacement.replace(weights)

            return stream

        with patch.object(pathlib.Path, "open", replace_after_payload_open):
            with self.assertRaises((FileNotFoundError, ValueError)):
                with sanitize.verified_files(weights, companions=(config,)):
                    self.fail("a mixed or invalidated generation must not be yielded")

        self.assertTrue(replaced)

    def test_atomic_marker_replacement_rejects_old_open_generation(self):
        (weights, config) = self.make_bundle()
        marker = sanitize.marker_path(weights)
        old_inode = marker.stat().st_ino
        replacement_marker = marker.with_name(marker.name + ".next")

        weights.parent.chmod(0o755)
        original_open = pathlib.Path.open
        replaced = False

        def replace_marker_after_payload_open(opened_path, *args, **kwargs):
            nonlocal replaced
            stream = original_open(opened_path, *args, **kwargs)

            if (pathlib.Path(opened_path) == weights) and (not replaced):
                replaced = True

                replacement_marker.write_bytes(marker.read_bytes())
                replacement_marker.replace(marker)

            return stream

        with patch.object(pathlib.Path, "open", replace_marker_after_payload_open):
            with self.assertRaisesRegex(ValueError, "sanitization changed while opening"):
                with sanitize.verified_files(weights, companions=(config,)):
                    self.fail("a replacement marker must not validate an older opened generation")

        self.assertTrue(replaced)
        self.assertTrue(marker.exists())
        self.assertNotEqual(marker.stat().st_ino, old_inode)

    def test_open_descriptor_stays_on_loaded_checkpoint_generation_after_rename(self):
        weights = self.directory / "weights.pt"

        self.save(weights, {"generation": torch.tensor([1])})
        source_hash = self.digest(weights)

        sanitize.sanitize(weights, source_hash)
        self.directory.chmod(0o755)
        replacement = self.directory / "replacement.pt"

        self.save(replacement, {"generation": torch.tensor([2])})

        with sanitize.verified_files(weights) as streams:
            replacement.replace(weights)
            loaded_checkpoint = torch.load(streams[weights], map_location="cpu", weights_only=True)

        current = torch.load(weights, map_location="cpu", weights_only=True)

        self.assertEqual(first=1, second=loaded_checkpoint["generation"].item())
        self.assertEqual(first=2, second=current["generation"].item())

    def test_payload_hash_mismatch_refuses_before_loading(self):
        (weights, config) = self.make_bundle()

        weights.chmod(0o644)
        weights.write_bytes(b"changed weights")

        with patch.object(torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                with sanitize.verified_files(weights, companions=(config,)):
                    self.fail("changed weights must not be loaded")

            load.assert_not_called()

    def test_companion_hash_mismatch_refuses(self):
        (weights, config) = self.make_bundle()

        config.write_text("changed config")

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            with sanitize.verified_files(weights, companions=(config,)):
                self.fail("changed config must not be loaded")
