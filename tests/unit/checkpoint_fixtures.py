"""Shared synthetic checkpoint fixtures for the unit-test categories."""

from collections import OrderedDict
import hashlib
import pathlib
import tempfile
import unittest

import torch

import sanitize


class TrainingMetadata:
    """Stand in for the inert optimizer metadata global in the MolDeTr source."""

    def __init__(self, metadata_values):
        self.metadata_values = metadata_values


class CheckpointCase(unittest.TestCase):
    """Temporary files and synthetic states shared by checkpoint contract tests."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="checkpoint-contract-")
        self.directory = pathlib.Path(self.temporary.name)

        self.directory.chmod(0o755)

    def tearDown(self):
        for checkpoint_path in self.directory.rglob("*"):
            if checkpoint_path.is_dir():
                checkpoint_path.chmod(0o755)

        self.directory.chmod(0o755)
        self.temporary.cleanup()

    @staticmethod
    def digest(checkpoint_path):
        with pathlib.Path(checkpoint_path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def save(checkpoint_path, checkpoint_value):
        checkpoint_path = pathlib.Path(checkpoint_path)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with checkpoint_path.open("wb") as stream:
            torch.save(checkpoint_value, stream)

        return checkpoint_path

    def make_bundle(self, directory=None, tag="epoch1"):
        directory = pathlib.Path(directory or (self.directory / tag))

        directory.mkdir(parents=True, exist_ok=True)
        weights = directory / "model.pt"
        config = directory / "config.yaml"
        state = OrderedDict((("weight", torch.tensor([1.0, 2.0])),))
        state._metadata = OrderedDict((("", {"version": 1}),))

        self.save(
            weights,
            {
                "weights": state,
                "metadata": {"fixture": tag},
            },
        )

        config.write_text(
            f"fixture:\n"
            f"  name: {tag}\n"  # machine-string: YAML fixture content.
        )

        sanitize.sanitize(
            weights,
            self.digest(weights),
            companions={config: self.digest(config)},
        )

        return (weights, config)

    def mol_source(self):
        weights = self.directory / "moldetr.pth"
        shared = torch.tensor([1.0, 2.0], dtype=torch.float32)

        state = OrderedDict(
            (
                ("first", shared),
                ("alias", shared),
                ("integer_buffer", torch.tensor([2], dtype=torch.int64)),
            )
        )

        state._metadata = OrderedDict((("", {"version": 3}),))
        source = {"model": state, "optimizer": TrainingMetadata(["discarded"])}

        self.save(weights, source)
        source_hash = self.digest(weights)

        inert_global = f"{TrainingMetadata.__module__}.{TrainingMetadata.__qualname__}"  # machine-string: Dotted pickle global name.

        return (weights, source_hash, inert_global)
