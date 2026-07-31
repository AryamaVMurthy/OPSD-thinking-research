"""Strict problem-only guidance ensembles and matched negative controls."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any


GUIDANCE_INPUT_PROTOCOL = "problem-only-independent-v1"
CACHE_KIND = "answer-free-guidance-ensemble"
OFFICIAL_HARDCODED_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"

_NUMBER = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d+/\d+)"
)
_ANSWER_CLAIM = re.compile(
    r"(?:"
    r"\\boxed|"
    r"\bfinal\s+(?:answer|result|value)\b|"
    r"\b(?:answer|result)\s+(?:is\b|equals\b|=)|"
    r"\btherefore\s+(?:the\s+)?(?:answer|result|value)\b|"
    r"\b(?:hence|thus)\s+(?:the\s+)?(?:answer|result|value)\b"
    r")",
    re.IGNORECASE,
)
_NUMERIC_CLAIM = re.compile(
    r"(?:"
    r"=\s*[-+]?\d|"
    r"\b(?:equals?|gives?|yields?|obtains?|evaluates?\s+to)"
    r"\s+[-+]?\d"
    r")",
    re.IGNORECASE,
)
_INJECTION = re.compile(
    r"(?:ignore\s+(?:all\s+)?previous|system\s+prompt|"
    r"reveal\s+(?:the\s+)?solution|reference\s+(?:answer|solution))",
    re.IGNORECASE,
)


def _normalized_numbers(text: str) -> set[str]:
    return {
        match.group(0).replace(",", "").lstrip("+")
        for match in _NUMBER.finditer(text)
    }


def validate_answer_free_plan(problem: str, plan: str) -> None:
    """Reject conclusion-like or newly calculated content without an answer.

    The validator receives only the problem and candidate plan.  In
    particular, it cannot compare against a reference answer or solution.
    Numerals copied from the problem are permitted; newly introduced numerals
    are rejected because this cache is intended to contain procedures rather
    than partially solved trajectories.
    """
    problem = str(problem).strip()
    plan = str(plan).strip()
    if not plan:
        raise ValueError("answer-free plan is empty")
    if len(plan) > 2400:
        raise ValueError("answer-free plan exceeds the length limit")
    if _INJECTION.search(plan):
        raise ValueError("answer-free plan contains an injection pattern")
    if _ANSWER_CLAIM.search(plan):
        raise ValueError("answer-free plan contains an answer claim")
    if _NUMERIC_CLAIM.search(plan):
        raise ValueError("answer-free plan contains a derived numerical claim")
    problem_numbers = _normalized_numbers(problem)
    novel_numbers = _normalized_numbers(plan) - problem_numbers
    if novel_numbers:
        raise ValueError(
            "answer-free plan contains a derived numerical result: "
            + ", ".join(sorted(novel_numbers))
        )


def validate_guidance_ensemble(
    problem: str, plans: Sequence[str]
) -> None:
    """Require separately useful plans rather than duplicated samples."""
    if len(plans) < 2:
        raise ValueError("guidance ensemble requires at least two plans")
    normalized = []
    shingles = []
    for plan in plans:
        validate_answer_free_plan(problem, plan)
        words = re.findall(r"[a-z0-9]+", str(plan).lower())
        normalized.append(" ".join(words))
        shingles.append(
            {
                tuple(words[offset : offset + 3])
                for offset in range(max(1, len(words) - 2))
            }
        )
    for left in range(len(plans)):
        for right in range(left + 1, len(plans)):
            if normalized[left] == normalized[right]:
                raise ValueError(
                    "guidance plans must be independently distinct"
                )
            union = shingles[left] | shingles[right]
            similarity = (
                len(shingles[left] & shingles[right]) / len(union)
                if union
                else 1.0
            )
            if similarity > 0.8:
                raise ValueError(
                    "guidance plans must be independently distinct"
                )


def answer_free_guidance_row(
    *,
    question: str,
    guides: Sequence[str],
    controls: Sequence[str],
    source_index: int,
) -> dict[str, object]:
    """Create a training row with no answer/reference field in its schema."""
    problem = str(question).strip()
    normalized_guides = [str(plan).strip() for plan in guides]
    normalized_controls = [str(plan).strip() for plan in controls]
    if not problem:
        raise ValueError("Fisher guidance requires a nonempty problem")
    if len(normalized_guides) < 2:
        raise ValueError("Fisher guidance requires at least two plans")
    if len(normalized_guides) != len(normalized_controls):
        raise ValueError("guides and controls must be matched pairs")
    for plan in [*normalized_guides, *normalized_controls]:
        # Controls describe other problems, so their original problem numerals
        # were validated when the cache was built.  Here we recheck only the
        # conclusion/injection constraints by preserving their own numerals.
        if _INJECTION.search(plan):
            raise ValueError("guidance row contains an injection pattern")
        if _ANSWER_CLAIM.search(plan):
            raise ValueError("guidance row contains an answer claim")
        if _NUMERIC_CLAIM.search(plan):
            raise ValueError("guidance row contains a derived numerical claim")
    return {
        "problem": problem,
        "solution": normalized_guides[0],
        "fisher_guides": normalized_guides,
        "fisher_controls": normalized_controls,
        "fisher_source_index": int(source_index),
    }


def load_guidance_ensemble(
    manifest_path: str | Path,
) -> tuple[dict[int, list[str]], dict[str, Any]]:
    """Load and cryptographically validate an answer-free guidance cache."""
    manifest_path = Path(manifest_path)
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid guidance manifest: {error}") from error
    expected = {
        "schema_version": 1,
        "cache_kind": CACHE_KIND,
        "answer_access": False,
        "reference_solution_access": False,
        "guidance_input_protocol": GUIDANCE_INPUT_PROTOCOL,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"guidance manifest {key} must be {value!r}, "
                f"observed {metadata.get(key)!r}"
            )
    pair_count = metadata.get("plans_per_problem")
    if (
        not isinstance(pair_count, int)
        or isinstance(pair_count, bool)
        or pair_count < 2
    ):
        raise ValueError("guidance manifest plans_per_problem must be >= 2")
    records_name = metadata.get("records_file")
    if (
        not isinstance(records_name, str)
        or not records_name
        or Path(records_name).is_absolute()
    ):
        raise ValueError("guidance manifest records_file must be relative")
    records_path = manifest_path.parent / records_name
    try:
        raw = records_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read guidance records: {error}") from error
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != metadata.get("records_sha256"):
        raise ValueError("guidance records sha256 does not match manifest")

    ensembles: dict[int, list[str]] = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid guidance record at line {line_number}: {error}"
            ) from error
        allowed_fields = {
            "source_index",
            "problem",
            "problem_sha256",
            "plans",
            "seeds",
        }
        extra_fields = set(record) - allowed_fields
        if extra_fields:
            raise ValueError(
                f"guidance line {line_number} contains forbidden fields: "
                + ", ".join(sorted(extra_fields))
            )
        index = record.get("source_index")
        problem = record.get("problem")
        plans = record.get("plans")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in ensembles
        ):
            raise ValueError(
                f"invalid or duplicate source_index at line {line_number}"
            )
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"missing problem at line {line_number}")
        problem_sha = hashlib.sha256(problem.encode("utf-8")).hexdigest()
        if record.get("problem_sha256") != problem_sha:
            raise ValueError(
                f"problem hash mismatch at guidance line {line_number}"
            )
        if (
            not isinstance(plans, list)
            or len(plans) != pair_count
            or any(not isinstance(plan, str) for plan in plans)
        ):
            raise ValueError(
                f"guidance line {line_number} must contain {pair_count} plans"
            )
        validate_guidance_ensemble(problem, plans)
        ensembles[index] = [plan.strip() for plan in plans]
    if len(ensembles) != metadata.get("accepted_records"):
        raise ValueError("guidance accepted_records does not match records")
    return ensembles, metadata


def assign_matched_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    ensembles: Mapping[int, Sequence[str]],
    selected_indices: Collection[int],
) -> dict[int, list[str]]:
    """Assign source/length-matched plans using deterministic derangements."""
    selected = sorted({int(index) for index in selected_indices})
    if len(selected) < 3:
        raise ValueError("matched controls require at least three examples")
    if any(index not in ensembles for index in selected):
        raise ValueError("selected guidance index is missing an ensemble")
    pair_counts = {len(ensembles[index]) for index in selected}
    if len(pair_counts) != 1 or next(iter(pair_counts)) < 2:
        raise ValueError("guidance ensembles require a common pair count")
    pair_count = next(iter(pair_counts))

    by_source: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        by_source[str(rows[index].get("data_source") or "unknown")].append(
            index
        )
    groups: dict[int, list[int]] = {}
    for source_indices in by_source.values():
        ranked = sorted(
            source_indices,
            key=lambda index: (len(str(rows[index].get("question", ""))), index),
        )
        # Within each source, quartiles keep negative prompts comparable in
        # length while retaining enough donors for distinct pair controls.
        bins: list[list[int]] = [[] for _ in range(4)]
        for rank, index in enumerate(ranked):
            bins[min(3, rank * 4 // len(ranked))].append(index)
        for bin_indices in bins:
            donors = bin_indices if len(bin_indices) > pair_count else ranked
            if len(donors) <= pair_count:
                donors = selected
            for index in bin_indices:
                groups[index] = donors

    controls: dict[int, list[str]] = {}
    for index in selected:
        donors = groups[index]
        position = donors.index(index) if index in donors else 0
        chosen: list[str] = []
        used_donors: set[int] = set()
        shift = 1
        while len(chosen) < pair_count:
            donor = donors[(position + shift) % len(donors)]
            shift += 1
            if donor == index or donor in used_donors:
                continue
            used_donors.add(donor)
            pair_index = len(chosen)
            chosen.append(str(ensembles[donor][pair_index]).strip())
        controls[index] = chosen
    return controls


def install_fisher_guidance_dataset_redirect(
    manifest_path: str | Path,
    *,
    source_indices: Collection[int] | None = None,
) -> int:
    """Install plan ensembles with deterministic matched negative controls."""
    ensembles, metadata = load_guidance_ensemble(manifest_path)
    selected = (
        sorted(ensembles)
        if source_indices is None
        else sorted({int(index) for index in source_indices})
    )
    if not selected or any(index not in ensembles for index in selected):
        raise ValueError("selected Fisher guidance records are unavailable")

    import datasets

    from .training_data import load_math_cot_20k

    raw = load_math_cot_20k(heldout_fraction=0.0)["train"]
    if selected[-1] >= len(raw):
        raise ValueError("Fisher guidance source index is outside the dataset")

    # Bind the cache to the exact problem text without ever loading a
    # reference solution into the guidance row.
    records_path = Path(manifest_path).parent / str(metadata["records_file"])
    cached_problem_hashes = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            cached_problem_hashes[int(record["source_index"])] = str(
                record["problem_sha256"]
            )
    for index in selected:
        observed = hashlib.sha256(
            str(raw[index]["question"]).strip().encode("utf-8")
        ).hexdigest()
        if observed != cached_problem_hashes.get(index):
            raise ValueError(
                f"Fisher guidance problem mismatch at source index {index}"
            )

    controls = assign_matched_controls(
        raw,
        ensembles=ensembles,
        selected_indices=selected,
    )
    original_load_dataset = datasets.load_dataset

    def pinned_load_dataset(path, *args, **kwargs):
        if path != OFFICIAL_HARDCODED_DATASET:
            return original_load_dataset(path, *args, **kwargs)
        if args or kwargs:
            raise RuntimeError(
                "upstream OPSD dataset call unexpectedly supplied arguments"
            )
        subset = raw.select(selected).add_column(
            "_fisher_source_index", selected
        )

        def normalize(example):
            index = int(example["_fisher_source_index"])
            return answer_free_guidance_row(
                question=example["question"],
                guides=ensembles[index],
                controls=controls[index],
                source_index=index,
            )

        return datasets.DatasetDict(
            {
                "train": subset.map(
                    normalize,
                    remove_columns=subset.column_names,
                    desc="Attaching answer-free Fisher guidance pairs",
                )
            }
        )

    datasets.load_dataset = pinned_load_dataset
    return len(selected)
