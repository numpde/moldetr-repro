"""Own run directories and disposable workflow scratch under tmp."""

from contextlib import contextmanager
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid


def create_run_directory(root: Path, kind: str) -> Path:
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    directory = root / 'tmp' / kind / name

    directory.mkdir(parents=True)
    directory.chmod(0o755)

    return directory


def cleanup_run_directories(directories: Iterable[Path]) -> None:
    """Remove runs selected by their owner's lifetime or retention policy."""

    for directory in directories:
        shutil.rmtree(directory)


@contextmanager
def temporary_run_directory(root: Path, kind: str):
    """Remove this run's scratch on normal exit or exception."""
    directory = create_run_directory(root, kind)

    try:
        yield directory
    finally:
        cleanup_run_directories((directory,))
