"""Sanitize pinned PyTorch checkpoints and publish their actual file hashes.

Writers invalidate markers before replacing files and publish fresh markers last.
Verification checks only the selected checkpoint and its declared companions.
"""

import argparse
from contextlib import contextmanager, ExitStack
import fnmatch
import hashlib
import json
import os
from pathlib import Path


POLICY = "torch-weights-only-v1"
FP_MODES = {"keep": None, "fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "fp64": "float64"}


def rows(manifest):
    return [line.split() for line in Path(manifest).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def marker_path(path):
    return path.with_name(path.name + ".sanitized.json")


def sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def verified_files(path, companions=()):
    """Hash selected files and keep those exact descriptors open for loading."""
    path = Path(path)
    paths = [path, *(Path(companion) for companion in companions)]
    marker = marker_path(path)
    with ExitStack() as stack:
        try:
            receipt_stream = stack.enter_context(marker.open())
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Cannot verify {path}: its sanitization receipt is missing; run make weights/sanitize."
            ) from error
        receipt = json.load(receipt_stream)
        if (receipt["schema"] != 1 or receipt["policy"] != POLICY or
                set(receipt["files"]) != {item.name for item in paths}):
            raise ValueError(f"Cannot verify {path}: its sanitization receipt covers another policy or file set")
        streams = {item: stack.enter_context(item.open("rb")) for item in paths}
        for item, stream in streams.items():
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != receipt["files"][item.name]:
                raise ValueError(f"Cannot verify {item}: SHA-256 {digest} differs from its sanitization marker")
            stream.seek(0)
        # Writers unlink before replacement. A changed marker means these files
        # may straddle generations; after this check their open inodes stay fixed.
        if not os.path.samestat(os.fstat(receipt_stream.fileno()), marker.stat()):
            raise ValueError(f"Cannot verify {path}: sanitization changed while opening the checkpoint")
        yield streams


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


def sanitize(path, source_hash, *, precision="keep",
             extract=None, inert_globals=(), companions=None):
    """Certify an existing restricted-loadable file, or publish a converted one."""
    import torch

    path = Path(path)
    companions = companions or {}
    dtype = FP_MODES[precision]
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
        if digest != source_hash:
            # Repeated preparation is a no-op only for an intact marked output.
            try:
                with verified_files(path, companions=companions):
                    receipt = json.loads(marker_path(path).read_text())
            except FileNotFoundError as error:
                raise ValueError(
                    f"Cannot sanitize {path}: no marker covers this output; restore the pinned source with make weights/download first"
                ) from error
            if receipt["precision"] != precision:
                raise ValueError("To change precision, restore the source with make weights/download first")
            state = None
        else:
            receivers = []
            for name in inert_globals:
                module, _, class_name = name.rpartition(".")
                receivers.append(type(class_name, (DiscardedState,), {"__module__": module}))
            with torch.serialization.safe_globals(receivers):
                state = torch.load(stream, map_location="cpu", weights_only=True)
            if extract:
                state = state[extract]

    marker = marker_path(path)
    partial = path.with_name(path.name + ".sanitize.part")
    marker_partial = marker.with_suffix(".part")
    path.parent.chmod(0o755)
    try:
        partial.unlink(missing_ok=True)
        marker_partial.unlink(missing_ok=True)
        if state is not None and (extract or dtype is not None):
            if dtype is not None:
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
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--list-fp", action="store_true")
    parser.add_argument("--extract")
    parser.add_argument("--inert-global", action="append", default=[])
    parser.add_argument("--fp", choices=FP_MODES, default="keep")
    args = parser.parse_args()
    if args.list_fp:
        print(" ".join(FP_MODES))
        return
    if not args.verify:
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
        if args.verify:
            with verified_files(path, companions=companions):
                print(f"Verified {path}", flush=True)
        else:
            sanitize(path, sources[name], precision=args.fp,
                     extract=args.extract, inert_globals=args.inert_global, companions=companions)


if __name__ == "__main__":
    main()
