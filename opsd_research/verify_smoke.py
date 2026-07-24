from __future__ import annotations

import argparse
import json
from pathlib import Path

from .records import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    records = read_jsonl(args.input)
    if len(records) != 12:
        raise RuntimeError(f"expected 12 smoke generations, found {len(records)}")
    if not all(record.get("enable_thinking") is True for record in records):
        raise RuntimeError("a smoke record did not enable thinking")
    if not all(record.get("nonempty_thinking") is True for record in records):
        raise RuntimeError("a smoke record did not contain a non-empty thinking segment")
    if any(record.get("finish_reason") == "length" for record in records):
        raise RuntimeError("a smoke generation hit the length limit")
    if len({record["seed"] for record in records}) != 12:
        raise RuntimeError("smoke seeds are not unique")
    print(
        json.dumps(
            {
                "records": 12,
                "thinking_rate": 1.0,
                "max_output_tokens": max(record["output_tokens"] for record in records),
                "formatted": sum(bool(record["formatted"]) for record in records),
                "correct": sum(bool(record["correct"]) for record in records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
