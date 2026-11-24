"""Shared synthetic checkpoint fixtures for the unit-test categories."""

from collections import OrderedDict
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

import sanitize


class TrainingMetadata:
    """Stand in for the inert optimizer metadata global in the MolDeTr source."""

    def __init__(self, values):
        self.values = values


class CheckpointCase(unittest.TestCase):
    """Temporary files and synthetic states shared by checkpoint contract tests."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="checkpoint-contract-")
        self.directory = Path(self.temporary.name)
        self.directory.chmod(0o755)

    def tearDown(self):
        for path in self.directory.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        self.directory.chmod(0o755)
        self.temporary.cleanup()

    @staticmethod
    def digest(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def save(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            torch.save(value, stream)
        return path

    def make_bundle(self, directory=None, tag="epoch1"):
        directory = Path(directory or self.directory / tag)
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
        config.write_text(f"fixture:\n  name: {tag}\n")
        sanitize.sanitize(
            weights,
            self.digest(weights),
            companions={config: self.digest(config)},
        )
        return weights, config

    def mol_source(self):
        weights = self.directory / "moldetr.pth"
        shared = torch.tensor([1.0, 2.0], dtype=torch.float32)
        state = OrderedDict((
            ("first", shared),
            ("alias", shared),
            ("integer_buffer", torch.tensor([2], dtype=torch.int64)),
        ))
        state._metadata = OrderedDict((("", {"version": 3}),))
        source = {"model": state, "optimizer": TrainingMetadata(["discarded"])}
        self.save(weights, source)
        source_hash = self.digest(weights)

        inert_global = f"{TrainingMetadata.__module__}.{TrainingMetadata.__qualname__}"
        return weights, source_hash, inert_global
