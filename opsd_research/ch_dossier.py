"""Validated contrastive-hindsight dossiers for privileged OPSD teachers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_VERDICTS = {"correct", "partial", "incorrect"}


@dataclass(frozen=True)
class AttemptAudit:
    attempt_index: int
    verdict: str
    method: str
    first_error: str | None
    valid_insights: tuple[str, ...]
    missing_checks: tuple[str, ...]


@dataclass(frozen=True)
class ContrastiveAudit:
    attempts: tuple[AttemptAudit, ...]
    best_attempt_index: int
    shared_failure_mode: str
    corrected_method: tuple[str, ...]
    verification_checks: tuple[str, ...]


def _strings(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or (not result and not allow_empty):
        raise ValueError(f"{field} contains no usable text")
    return result


def parse_audit(payload: dict[str, Any], *, expected_attempts: int) -> ContrastiveAudit:
    """Validate the auditor's public JSON contract."""
    if expected_attempts not in {2, 3}:
        raise ValueError("expected_attempts must be two or three")
    raw_attempts = payload.get("attempts")
    if not isinstance(raw_attempts, list) or len(raw_attempts) != expected_attempts:
        raise ValueError(f"audit requires exactly {expected_attempts} attempts")
    attempts: list[AttemptAudit] = []
    for expected_index, raw in enumerate(raw_attempts):
        if not isinstance(raw, dict) or raw.get("attempt_index") != expected_index:
            raise ValueError("attempt indices must be contiguous and ordered")
        verdict = str(raw.get("verdict", "")).strip()
        if verdict not in _VERDICTS:
            raise ValueError(f"unsupported attempt verdict {verdict!r}")
        method = str(raw.get("method", "")).strip()
        first_error = raw.get("first_error")
        if first_error is not None:
            first_error = str(first_error).strip() or None
        if not method or (verdict != "correct" and not first_error):
            raise ValueError("each non-correct attempt requires a method and first error")
        attempts.append(
            AttemptAudit(
                attempt_index=expected_index,
                verdict=verdict,
                method=method,
                first_error=first_error,
                valid_insights=_strings(
                    raw.get("valid_insights"), "valid_insights", allow_empty=True
                ),
                missing_checks=_strings(
                    raw.get("missing_checks"), "missing_checks", allow_empty=True
                ),
            )
        )
    best = payload.get("best_attempt_index")
    if not isinstance(best, int) or not 0 <= best < expected_attempts:
        raise ValueError("best_attempt_index is outside the attempt set")
    shared = str(payload.get("shared_failure_mode", "")).strip()
    if not shared:
        raise ValueError("shared_failure_mode must be nonempty")
    return ContrastiveAudit(
        attempts=tuple(attempts),
        best_attempt_index=best,
        shared_failure_mode=shared,
        corrected_method=_strings(payload.get("corrected_method"), "corrected_method"),
        verification_checks=_strings(
            payload.get("verification_checks"), "verification_checks"
        ),
    )


def render_teacher_dossier(
    *,
    problem: str,
    reference_solution: str,
    reference_answer: str,
    blind_attempts: tuple[str, ...],
    audit: ContrastiveAudit,
) -> str:
    """Render privileged teacher-only context from trusted and audited inputs."""
    if len(blind_attempts) != len(audit.attempts):
        raise ValueError("blind attempts do not match the audit")
    lines = [
        "Privileged contrastive-hindsight dossier.",
        "The student never receives this dossier; use it only to supervise the "
        "student's unprivileged on-policy reasoning.",
        f"Problem: {problem}",
        f"Verified answer: {reference_answer}",
        f"Verified reference solution: {reference_solution}",
        "",
        "Blind student attempts and audit:",
    ]
    for attempt, analysis in zip(blind_attempts, audit.attempts, strict=True):
        lines.extend(
            [
                f"Attempt {analysis.attempt_index} [{analysis.verdict}]: {attempt}",
                f"Method: {analysis.method}",
                f"First error: {analysis.first_error or 'none'}",
                "Valid insights: " + ("; ".join(analysis.valid_insights) or "none"),
                "Missing checks: " + ("; ".join(analysis.missing_checks) or "none"),
            ]
        )
    lines.extend(
        [
            f"Best attempt: {audit.best_attempt_index}",
            f"Shared failure mode: {audit.shared_failure_mode}",
            "Corrected method: " + " -> ".join(audit.corrected_method),
            "Required verification: " + "; ".join(audit.verification_checks),
            "Produce a rigorous solution, correct the diagnosed failures, and verify "
            "the final answer rather than copying an attempt.",
        ]
    )
    return "\n".join(lines)


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
