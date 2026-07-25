import pytest

from opsd_research.build_graf_cache import _json_object


def test_extracts_plain_or_fenced_json_object() -> None:
    assert _json_object('{"forks": []}') == {"forks": []}
    assert _json_object('```json\n{"forks": []}\n```') == {"forks": []}


def test_rejects_non_json_builder_response() -> None:
    with pytest.raises(ValueError, match="JSON"):
        _json_object("I cannot provide that.")
