"""Prepare PyTorch inference checkpoints and admit only marked generations.

Preparation verifies pinned input digests and uses restricted deserialization.
Admission checks only the selected receipt generation, without rehashing weights.
Managed writers must remove that receipt before replacing any covered file,
publish files by rename, and publish a fresh receipt last. Admission hands the
same open files to the loader, so later renames cannot change its inputs.
"""

import argparse
from contextlib import contextmanager, ExitStack
import fnmatch
import hashlib
import json
import os
from pathlib import Path


POLICY = "torch-weights-only-v1"


def rows(manifest):
    return [line.split() for line in Path(manifest).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def exports(manifest):
    return {mode: (digest, dtype) for digest, mode, dtype in rows(manifest)}


def marker_path(path):
    return path.with_name(path.name + ".sanitized.json")


def sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def verified_weights(path, companions=()):
    """Admit selected files without repeating preparation's full-file hashes."""
    path = Path(path)
    paths = [path, *(Path(companion) for companion in companions)]
    marker = marker_path(path)
    with ExitStack() as stack:
        try:
            receipt_stream = stack.enter_context(marker.open())
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Cannot load {path}: its sanitization receipt is missing; run make weights/sanitize."
            ) from error
        receipt = json.load(receipt_stream)
        if (receipt["schema"] != 1 or receipt["policy"] != POLICY or
                set(receipt["files"]) != {item.name for item in paths}):
            raise ValueError(f"Cannot load {path}: its sanitization receipt covers another policy or file set")
        streams = {item: stack.enter_context(item.open("rb")) for item in paths}
        # Writers unlink before replacement. A changed marker means these files
        # may straddle generations; after this check their open inodes stay fixed.
        if not os.path.samestat(os.fstat(receipt_stream.fileno()), marker.stat()):
            raise ValueError(f"Cannot load {path}: sanitization changed while opening the checkpoint")
        yield streams


def load_checkpoint(model, path, map_location="cpu"):
    """Load a selected, sanitized state dict without an unsafe fallback."""
    import torch

    path = Path(path)
    with verified_weights(path) as streams:
        state = torch.load(streams[path], map_location=map_location, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


class DiscardedState:
    """Receive explicitly named training metadata without importing its code."""


def convert_precision(value, dtype, converted):
    import torch

    if type(value) is torch.Tensor:
        if value.is_floating_point() and value.dtype != dtype:
            view = (value.data_ptr(), value.shape, value.stride(), value.dtype)
            if view not in converted:
                converted[view] = value.to(dtype)
            return converted[view]
    elif isinstance(value, dict):
        # Mutating retains OrderedDict module metadata used by load_state_dict.
        for key, item in value.items():
            value[key] = convert_precision(item, dtype, converted)
    elif isinstance(value, list):
        value[:] = [convert_precision(item, dtype, converted) for item in value]
    elif isinstance(value, tuple):
        return tuple(convert_precision(item, dtype, converted) for item in value)
    return value


def sanitize(path, source_hash, *, output_definitions=None, precision="keep",
             extract=None, inert_globals=(), companions=None):
    """Certify an existing restricted-loadable file, or publish a converted one."""
    import torch

    path = Path(path)
    companions = companions or {}
    definitions = output_definitions or {}
    target_hash, dtype = definitions.get(precision, (source_hash, "preserve"))
    hashes = {}
    for companion, expected in companions.items():
        with companion.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Cannot sanitize {path}: companion {companion} differs from its pinned SHA-256")
        hashes[companion.name] = actual

    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        if digest == target_hash:
            # Existing approved outputs need no rewrite or extra safe globals.
            torch.load(stream, map_location="cpu", weights_only=True)
            state = None
        elif digest == source_hash:
            # Class objects work on both older and newer PyTorch versions;
            # named (class, global) pairs are unavailable in the CPU profile.
            receivers = []
            for name in inert_globals:
                module, _, class_name = name.rpartition(".")
                receivers.append(type(class_name, (DiscardedState,), {"__module__": module}))
            with torch.serialization.safe_globals(receivers):
                state = torch.load(stream, map_location="cpu", weights_only=True)
            if extract:
                state = state[extract]
        elif definitions and digest == definitions["keep"][0]:
            state = torch.load(stream, map_location="cpu", weights_only=True)
        elif digest in {item[0] for item in definitions.values()}:
            raise ValueError("To change precision, restore the full-precision source with make weights/download first")
        else:
            raise ValueError(f"Cannot sanitize {path}: SHA-256 {digest} is not a pinned source or selected output")

    marker = marker_path(path)
    partial = path.with_name(path.name + ".sanitize.part")
    marker_partial = marker.with_suffix(".part")
    path.parent.chmod(0o755)
    try:
        partial.unlink(missing_ok=True)
        marker_partial.unlink(missing_ok=True)
        if state is not None:
            if dtype != "preserve":
                state = convert_precision(state, getattr(torch, dtype), {})
            with partial.open("xb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # The published result must load with the defaults, without the
            # source's inert training-metadata receivers.
            torch.load(partial, map_location="cpu", weights_only=True)
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != target_hash:
                raise ValueError(f"Cannot publish {path}: output SHA-256 {digest} differs from {target_hash}")
            partial.chmod(0o444)
            marker.unlink(missing_ok=True)
            sync_directory(path.parent)
            partial.replace(path)
            sync_directory(path.parent)
        hashes[path.name] = digest
        # A complete admitted output supersedes a resumed source download.
        path.with_name(path.name + ".part").unlink(missing_ok=True)
        receipt = {"schema": 1, "policy": POLICY, "files": hashes, "precision": precision}
        with marker_partial.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        marker_partial.chmod(0o444)
        marker_partial.replace(marker)
        sync_directory(path.parent)
        print(f"Sanitized {path}: {precision}, SHA-256 {digest}", flush=True)
    finally:
        try:
            partial.unlink(missing_ok=True)
            marker_partial.unlink(missing_ok=True)
        finally:
            path.parent.chmod(0o555)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/checkpoints"))
    parser.add_argument("--manifest", type=Path, default=Path("/checks/checkpoints.sha256"))
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--companion", action="append", default=[])
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--extract")
    parser.add_argument("--inert-global", action="append", default=[])
    parser.add_argument("--fp", default="keep")
    args = parser.parse_args()
    definitions = exports(args.output_manifest) if args.output_manifest else None
    if args.fp not in (definitions or {"keep": None}):
        parser.error(f"Choose --fp from {', '.join(definitions or {'keep': None})}")
    if (args.extract or args.inert_global) and not definitions:
        parser.error("Extraction needs --output-manifest so only pinned inference outputs can be published")
    import torch

    torch.set_num_threads(1)
    sources = {name: digest for digest, _limit, name in rows(args.manifest)}
    selected = [name for name in sources if fnmatch.fnmatchcase(name, args.pattern)]
    if not selected:
        parser.error(f"No checkpoint in {args.manifest} matches {args.pattern!r}")
    for name in selected:
        path = args.directory / name
        companions = {path.with_name(item): sources[str(Path(name).with_name(item))]
                      for item in args.companion}
        sanitize(path, sources[name], output_definitions=definitions, precision=args.fp,
                 extract=args.extract, inert_globals=args.inert_global, companions=companions)


if __name__ == "__main__":
    main()
