"""Validation boundary for immutable GRAF graph caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def validate_graph_cache_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Return a verified manifest or fail before any policy optimization begins."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid graph-cache manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("graph-cache manifest has an unsupported schema")
    try:
        cache_path = Path(str(manifest["cache"]))
        claimed_digest = str(manifest["cache_sha256"])
        requested = int(manifest["requested_examples"])
        accepted = int(manifest["accepted_examples"])
        rejected = int(manifest["rejected_examples"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("graph-cache manifest is missing required counters") from error
    if not cache_path.is_absolute():
        cache_path = path.parent / cache_path
    if not cache_path.is_file():
        raise ValueError(f"graph-cache file does not exist: {cache_path}")
    observed_digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if observed_digest != claimed_digest:
        raise ValueError("graph-cache SHA-256 does not match its manifest")
    if requested <= 0 or accepted <= 0 or rejected < 0 or accepted + rejected != requested:
        raise ValueError("graph-cache manifest counters are inconsistent")
    return manifest
