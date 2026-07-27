"""Validated contrastive-hindsight dossiers for privileged OPSD teachers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from pathlib import Path


def render_dynamic_teacher_dossier(
    *,
    problem: str,
    reference_solution: str,
    reference_answer: str,
    blind_attempts: tuple[str, ...],
    comparative_audit: str,
) -> str:
    """Render natural teacher context without imposing an audit ontology."""
    if len(blind_attempts) not in {2, 3} or any(
        not attempt.strip() for attempt in blind_attempts
    ):
        raise ValueError("dynamic dossiers require two or three blind attempts")
    audit = comparative_audit.strip()
    if not audit:
        raise ValueError("dynamic dossiers require a nonempty comparative audit")
    attempts = "\n\n".join(
        f"Blind attempt {index + 1}:\n{attempt}"
        for index, attempt in enumerate(blind_attempts)
    )
    return "\n".join(
        [
            "Privileged contrastive-hindsight context for the teacher only.",
            "The student sees only the original problem and reasons freely.",
            "",
            f"Problem:\n{problem}",
            "",
            f"Trusted final answer:\n{reference_answer}",
            "",
            f"Trusted reference reasoning:\n{reference_solution}",
            "",
            attempts,
            "",
            f"Natural comparative audit:\n{audit}",
            "",
            "Use all of this evidence flexibly when supervising the student's "
            "new on-policy reasoning. Preserve useful ideas, correct mistakes, "
            "and verify the final result; do not force the student to imitate "
            "a fixed template or any one prior attempt.",
        ]
    )


def accepted_teacher_dossiers(manifest_path: str | Path) -> dict[int, str]:
    """Load immutable teacher-only dossiers after verifying their manifest."""
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_path = Path(str(manifest["cache"]))
        requested = int(manifest["requested_examples"])
        accepted = int(manifest["accepted_examples"])
        rejected = int(manifest["rejected_examples"])
        blind_attempts = int(manifest["blind_attempts_per_problem"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid CH-OPSD dossier manifest: {error}") from error
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported CH-OPSD dossier manifest schema")
    if manifest.get("student_answer_context") is not False:
        raise ValueError("CH-OPSD student context must remain answer-free")
    if manifest.get("teacher_reference_context") is not True:
        raise ValueError("CH-OPSD teacher must receive verified reference context")
    if manifest.get("audit_format") != "natural_language_v1":
        raise ValueError("CH-OPSD audit must remain free-form natural language")
    if manifest.get("schema_based_selection") is not False:
        raise ValueError("CH-OPSD cannot use schema-based example selection")
    if blind_attempts not in {2, 3}:
        raise ValueError("CH-OPSD requires two or three blind attempts")
    if requested <= 0 or accepted <= 0 or rejected < 0 or accepted + rejected != requested:
        raise ValueError("CH-OPSD dossier manifest counters are inconsistent")
    if not cache_path.is_absolute():
        cache_path = manifest_path.parent / cache_path
    try:
        content = cache_path.read_bytes()
    except OSError as error:
        raise ValueError(f"CH-OPSD dossier cache does not exist: {cache_path}") from error
    if hashlib.sha256(content).hexdigest() != str(manifest.get("cache_sha256", "")):
        raise ValueError("CH-OPSD dossier cache SHA-256 does not match its manifest")
    dossiers: dict[int, str] = {}
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid CH dossier record at line {line_number}") from error
        if record.get("schema_version") != 1:
            raise ValueError(f"unsupported CH dossier record at line {line_number}")
        if not record.get("accepted"):
            continue
        index = int(record["example_index"])
        context = str(record["teacher_dossier"]).strip()
        if index in dossiers or not context:
            raise ValueError("accepted CH dossier identities and context must be valid")
        if int(record.get("blind_attempt_count", -1)) != blind_attempts:
            raise ValueError("CH dossier attempt count does not match its manifest")
        dossiers[index] = context
    if len(dossiers) != accepted:
        raise ValueError("accepted CH dossier count does not match its manifest")
    return dossiers


def select_teacher_dossiers(
    dossiers: dict[int, str], source_indices: Collection[int] | None
) -> dict[int, str]:
    """Return an exact, deterministic subset of accepted dossier identities."""
    if source_indices is None:
        return dossiers
    selected = {int(index) for index in source_indices}
    unknown = selected.difference(dossiers)
    if unknown:
        raise ValueError(
            "requested CH source indices are not accepted dossier rows: "
            f"{sorted(unknown)[:5]}"
        )
    if not selected:
        raise ValueError("CH source-index selection must not be empty")
    return {index: dossiers[index] for index in sorted(selected)}
