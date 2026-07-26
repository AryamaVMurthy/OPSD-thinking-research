"""GPU-efficient construction of contrastive-hindsight OPSD dossiers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from .ch_dossier import (
    parse_audit,
    render_dynamic_teacher_dossier,
    render_teacher_dossier,
)
from .generation_common import extract_last_boxed


def blind_student_prompt(problem: str, *, attempt_index: int) -> str:
    """Prompt one independent student attempt without privileged information."""
    if attempt_index < 0:
        raise ValueError("attempt_index must be nonnegative")
    return f"""Solve the following math problem as independent blind attempt {attempt_index + 1}.
Reason carefully, check your work, and put the final answer in \\boxed{{}}.

Problem:
{problem}"""


def auditor_prompt(
    *,
    problem: str,
    reference_solution: str,
    reference_answer: str,
    attempts: tuple[str, ...],
) -> str:
    """Request a compact structured audit with full privileged context."""
    if len(attempts) not in {2, 3}:
        raise ValueError("the auditor requires two or three blind attempts")
    attempts_text = "\n\n".join(
        f"Blind attempt {index}: {attempt}"
        for index, attempt in enumerate(attempts)
    )
    return f"""Compare independent blind attempts to the math problem below.
You have privileged access to a verified solution and answer. Diagnose each
attempt against that evidence. Explain naturally which methods are right,
partially right, or wrong; identify the first consequential mistakes, useful
ideas worth preserving, and missing verification. Then synthesize a corrected
approach and the checks it needs. Do not merely vote on final answers.

Problem:
{problem}

Verified answer: {reference_answer}

Verified reference solution:
{reference_solution}

{attempts_text}

Write naturally and concisely. Cover every supplied attempt, but do not return
JSON, follow a fixed schema, or force the mathematical methods into predefined
categories."""


def build_dossier_record(
    *,
    example_index: int,
    problem: str,
    reference_solution: str,
    blind_attempts: tuple[str, ...],
    audit_payload: dict[str, Any],
    builder_seed: int,
) -> dict[str, Any]:
    """Validate generated evidence and build one immutable cache record."""
    if example_index < 0:
        raise ValueError("example_index must be nonnegative")
    if len(blind_attempts) not in {2, 3} or any(
        not attempt.strip() for attempt in blind_attempts
    ):
        raise ValueError("each dossier requires two or three nonempty blind attempts")
    reference_answer = extract_last_boxed(reference_solution)
    if reference_answer is None:
        raise ValueError("reference solution has no boxed answer")
    audit = parse_audit(audit_payload, expected_attempts=len(blind_attempts))
    dossier = render_teacher_dossier(
        problem=problem,
        reference_solution=reference_solution,
        reference_answer=reference_answer,
        blind_attempts=blind_attempts,
        audit=audit,
    )
    return {
        "schema_version": 1,
        "accepted": True,
        "example_index": example_index,
        "problem_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
        "reference_solution_sha256": hashlib.sha256(
            reference_solution.encode("utf-8")
        ).hexdigest(),
        "reference_answer": reference_answer,
        "blind_attempt_count": len(blind_attempts),
        "blind_attempts": list(blind_attempts),
        "audit": asdict(audit),
        "teacher_dossier": dossier,
        "builder_seed": builder_seed,
    }


def build_dynamic_dossier_record(
    *,
    example_index: int,
    problem: str,
    reference_solution: str,
    blind_attempts: tuple[str, ...],
    audit_text: str,
    builder_seed: int,
) -> dict[str, Any]:
    """Build one free-form CH record without schema-based example rejection."""
    if example_index < 0:
        raise ValueError("example_index must be nonnegative")
    reference_answer = extract_last_boxed(reference_solution)
    if reference_answer is None:
        raise ValueError("reference solution has no boxed answer")
    audit = audit_text.strip()
    dossier = render_dynamic_teacher_dossier(
        problem=problem,
        reference_solution=reference_solution,
        reference_answer=reference_answer,
        blind_attempts=blind_attempts,
        comparative_audit=audit,
    )
    return {
        "schema_version": 1,
        "accepted": True,
        "example_index": example_index,
        "problem_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
        "reference_solution_sha256": hashlib.sha256(
            reference_solution.encode("utf-8")
        ).hexdigest(),
        "reference_answer": reference_answer,
        "blind_attempt_count": len(blind_attempts),
        "blind_attempts": list(blind_attempts),
        "audit_format": "natural_language_v1",
        "comparative_audit": audit,
        "teacher_dossier": dossier,
        "builder_seed": builder_seed,
    }
