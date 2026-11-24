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
import pathlib


# Marker format identity is independent of operator settings.
# repo-check: allow-module-constant
POLICY = "torch-weights-only-v1"

# Supported tensor conversions are capabilities, not runtime defaults.
# repo-check: allow-module-constant
FP_MODES = {"keep": None, "fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "fp64": "float64"}


def rows(manifest):
    return [
        line.split()
        for line in pathlib.Path(manifest).read_text().splitlines()
        if line.strip() and (not line.lstrip().startswith("#"))
    ]


def marker_path(checkpoint_path):
    return checkpoint_path.with_name(checkpoint_path.name + ".sanitized.json")


def sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def verified_files(checkpoint_path, companions=()):
    """Hash selected files and keep those exact descriptors open for loading."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    paths = [checkpoint_path, *(pathlib.Path(companion) for companion in companions)]
    marker = marker_path(checkpoint_path)

    with ExitStack() as stack:
        try:
            receipt_stream = stack.enter_context(marker.open())
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Cannot verify {checkpoint_path=}: its sanitization receipt is missing; run make weights/sanitize."
            ) from error

        receipt = json.load(receipt_stream)

        if not (
            (receipt["schema"] == 1) and
            (receipt["policy"] == POLICY) and
            (set(receipt["files"]) == {checkpoint_member.name for checkpoint_member in paths})
        ):
            raise ValueError(f"Cannot verify {checkpoint_path=}: its sanitization receipt covers another policy or file set")

        streams = {checkpoint_member: stack.enter_context(checkpoint_member.open("rb")) for checkpoint_member in paths}

        for (checkpoint_member, stream) in streams.items():
            digest = hashlib.file_digest(stream, "sha256").hexdigest()

            if not (digest == receipt["files"][checkpoint_member.name]):
                raise ValueError(f"Cannot verify {checkpoint_member=}: SHA-256 {digest=} differs from its sanitization marker")

            stream.seek(0)

        # Writers unlink before replacement. A changed marker means these files
        # may straddle generations; after this check their open inodes stay fixed.
        if not os.path.samestat(os.fstat(receipt_stream.fileno()), marker.stat()):
            raise ValueError(f"Cannot verify {checkpoint_path=}: sanitization changed while opening the checkpoint")

        yield streams


class DiscardedState:
    """Receive explicitly named training metadata without importing its code."""


def convert_precision(checkpoint_value, dtype, converted):
    import torch

    if type(checkpoint_value) is torch.Tensor:
        if checkpoint_value.is_floating_point() and (checkpoint_value.dtype != dtype):
            view = (checkpoint_value.data_ptr(), checkpoint_value.shape, checkpoint_value.stride(), checkpoint_value.dtype)

            if not (view in converted):
                converted[view] = checkpoint_value.to(dtype)

            return converted[view]
    elif isinstance(checkpoint_value, dict):
        # Mutating retains OrderedDict module metadata used by load_state_dict.
        for (key, checkpoint_member) in checkpoint_value.items():
            checkpoint_value[key] = convert_precision(checkpoint_member, dtype, converted)
    elif isinstance(checkpoint_value, list):
        checkpoint_value[:] = [convert_precision(checkpoint_member, dtype, converted) for checkpoint_member in checkpoint_value]
    elif isinstance(checkpoint_value, tuple):
        # repo-check: allow-tuple-generator -- Preserve tuple-valued checkpoint metadata while recursively converting its tensors.
        return tuple(convert_precision(checkpoint_member, dtype, converted) for checkpoint_member in checkpoint_value)

    return checkpoint_value


def sanitize(
    checkpoint_path,
    source_hash,
    *,
    precision="keep",
    extract=None,
    inert_globals=(),
    companions=None
):
    """Certify an existing restricted-loadable file, or publish a converted one."""
    import torch

    checkpoint_path = pathlib.Path(checkpoint_path)
    companions = companions or {}
    dtype = FP_MODES[precision]
    hashes = {}

    for (companion, expected) in companions.items():
        with companion.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()

        if not (actual == expected):
            raise ValueError(f"Cannot sanitize {checkpoint_path=}: companion {companion=} differs from its pinned SHA-256")

        hashes[companion.name] = actual

    with checkpoint_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()

        stream.seek(0)

        if digest != source_hash:
            # Repeated preparation is a no-op only for an intact marked output.
            try:
                with verified_files(checkpoint_path, companions=companions):
                    receipt = json.loads(marker_path(checkpoint_path).read_text())
            except FileNotFoundError as error:
                raise ValueError(
                    f"Cannot sanitize {checkpoint_path=}: no marker covers this output; restore the pinned source with make weights/download first"
                ) from error

            if not (receipt["precision"] == precision):
                raise ValueError(f"Cannot change {precision=} from {receipt["precision"]=}; restore the source with make weights/download first")

            state = None
        else:
            receivers = []

            for name in inert_globals:
                (module, _, class_name) = name.rpartition(".")

                receivers.append(type(class_name, (DiscardedState,), {"__module__": module}))

            with torch.serialization.safe_globals(receivers):
                state = torch.load(stream, map_location="cpu", weights_only=True)

            if extract:
                state = state[extract]

    marker = marker_path(checkpoint_path)
    partial = checkpoint_path.with_name(checkpoint_path.name + ".sanitize.part")
    marker_partial = marker.with_suffix(".part")

    checkpoint_path.parent.chmod(0o755)

    try:
        partial.unlink(missing_ok=True)
        marker_partial.unlink(missing_ok=True)

        if (state is not None) and (extract or (dtype is not None)):
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
            sync_directory(checkpoint_path.parent)
            partial.replace(checkpoint_path)
            sync_directory(checkpoint_path.parent)

        hashes[checkpoint_path.name] = digest

        # A complete verified output supersedes a resumed source download.
        checkpoint_path.with_name(checkpoint_path.name + ".part").unlink(missing_ok=True)

        receipt = {"schema": 1, "policy": POLICY, "files": hashes, "precision": precision}

        with marker_partial.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        marker_partial.chmod(0o444)
        marker_partial.replace(marker)
        sync_directory(checkpoint_path.parent)
        print(f"Sanitized {checkpoint_path=}: {precision=}, SHA-256 {digest=}", flush=True)
    finally:
        try:
            partial.unlink(missing_ok=True)
            marker_partial.unlink(missing_ok=True)
        finally:
            checkpoint_path.parent.chmod(0o555)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--directory", type=pathlib.Path, default=pathlib.Path("/checkpoints"))
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("/checks/checkpoints.sha256"))
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

    sources = {name: digest for (digest, _maximum_bytes, name) in rows(args.manifest)}
    selected = [name for name in sources if fnmatch.fnmatchcase(name, args.pattern)]

    if not selected:
        parser.error(f"No checkpoint in {args.manifest=} matches {args.pattern=!r}")

    for name in selected:
        checkpoint_path = args.directory / name

        companions = {
            checkpoint_path.with_name(checkpoint_member): sources[str(pathlib.Path(name).with_name(checkpoint_member))]
            for checkpoint_member in args.companion
        }

        if args.verify:
            with verified_files(checkpoint_path, companions=companions):
                print(f"Verified {checkpoint_path=}", flush=True)
        else:
            sanitize(
                checkpoint_path,
                sources[name],
                precision=args.fp,
                extract=args.extract,
                inert_globals=args.inert_global,
                companions=companions
            )


if __name__ == "__main__":
    main()
