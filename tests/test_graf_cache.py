import hashlib
import json
from pathlib import Path

import pytest

from opsd_research.graf_cache import validate_graph_cache_manifest


def test_manifest_requires_immutable_cache_digest(tmp_path: Path) -> None:
    cache = tmp_path / "cache.jsonl"
    cache.write_text('{"accepted":true}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cache": "cache.jsonl",
                "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "requested_examples": 2,
                "accepted_examples": 1,
                "rejected_examples": 1,
            }
        ),
        encoding="utf-8",
    )
    assert validate_graph_cache_manifest(manifest)["accepted_examples"] == 1

    cache.write_text('{"accepted":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_graph_cache_manifest(manifest)
