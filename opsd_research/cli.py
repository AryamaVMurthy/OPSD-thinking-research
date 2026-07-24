from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import discover_configs, load_config


def _validate_one(path: Path) -> None:
    loaded = load_config(path)
    print(f"valid {loaded.path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="opsd-research")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("validate-config")
    one.add_argument("path", type=Path)
    all_parser = sub.add_parser("validate-all")
    all_parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    if args.command == "validate-config":
        _validate_one(args.path)
        return

    paths = discover_configs(args.root)
    if not paths:
        raise SystemExit("no configs discovered")
    for path in paths:
        _validate_one(path)
    print(json.dumps({"validated": len(paths)}, sort_keys=True))


if __name__ == "__main__":
    main()
