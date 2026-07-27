"""Create and install a checksum-pinned, root-only LoRA adapter bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_adapter(directory: Path, expected_sha256: str) -> None:
    for name in REQUIRED_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"adapter bundle is missing {name}")
    observed = _sha256(directory / "adapter_model.safetensors")
    if observed != expected_sha256:
        raise ValueError(
            f"adapter SHA-256 mismatch: expected {expected_sha256}, found {observed}"
        )


def pack_adapter(source: Path, archive: Path, expected_sha256: str) -> int:
    """Pack regular root files only, excluding checkpoints and other directories."""
    _verify_adapter(source, expected_sha256)
    files = sorted(path for path in source.iterdir() if path.is_file())
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w") as bundle:
        for path in files:
            bundle.add(path, arcname=path.name, recursive=False)
    return len(files)


def unpack_adapter(archive: Path, destination: Path, expected_sha256: str) -> int:
    """Install a flat adapter archive atomically after checksum verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage.", dir=destination.parent)
    )
    try:
        with tarfile.open(archive, "r") as bundle:
            members = bundle.getmembers()
            for member in members:
                if (
                    not member.isfile()
                    or Path(member.name).name != member.name
                    or member.name in {"", ".", ".."}
                ):
                    raise ValueError(f"unsafe adapter bundle member: {member.name!r}")
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read adapter bundle member: {member.name}")
                with source, (temporary / member.name).open("wb") as target:
                    shutil.copyfileobj(source, target)
        _verify_adapter(temporary, expected_sha256)
        if destination.exists():
            raise FileExistsError(f"adapter destination already exists: {destination}")
        os.replace(temporary, destination)
        return len(members)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("pack", "unpack"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--expected-sha256", required=True)
        if command == "pack":
            subparser.add_argument("--source", required=True, type=Path)
        else:
            subparser.add_argument("--destination", required=True, type=Path)
        subparser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        count = pack_adapter(args.source, args.archive, args.expected_sha256)
    else:
        count = unpack_adapter(args.archive, args.destination, args.expected_sha256)
    print(f"{args.command}ed_files={count}")


if __name__ == "__main__":
    main()
