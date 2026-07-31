"""Measure whether FiNOD's guide direction is problem-specific in Fisher space.

This diagnostic compares the true problem guide with two answer-free nulls:

* a generic verification guide; and
* a guide belonging to a different problem.

It intentionally uses the frozen base model and fixed recorded rollouts.  It
does not train a model or inspect benchmark answers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from .finod_dataset import remove_graph_identifiers
from .finod_prompts import build_finod_teacher_views
from .generation_common import extract_last_boxed, grade_math
from .graf_graph import (
    GraphAction,
    GraphFork,
    ReasoningGraph,
    render_graph_scaffold,
)
from .training_data import load_math_cot_20k


GENERIC_GUIDE = """Use a direct mathematical representation suited to the problem.
Track every assumption and quantity consistently.
Check algebra, counting boundaries, and geometric orientation before committing.
If a check fails, retract the faulty step and try a genuinely different route."""


@dataclass(frozen=True)
class AuditSample:
    source_index: int
    category: str
    problem: str
    guide: str
    completion: str


def _graph(payload: dict[str, Any]) -> ReasoningGraph:
    return ReasoningGraph(
        problem_sha256=str(payload["problem_sha256"]),
        forks=tuple(
            GraphFork(
                fork_id=str(fork["fork_id"]),
                state=str(fork["state"]),
                actions=tuple(
                    GraphAction(**action) for action in fork["actions"]
                ),
            )
            for fork in payload["forks"]
        ),
        schema_version=int(payload.get("schema_version", 1)),
    )


def _question_from_prompt(prompt: str) -> str:
    try:
        return (
            prompt.split("Problem: ", 1)[1]
            .split(
                "\n\nPlease reason step by step, and put your final answer",
                1,
            )[0]
            .strip()
        )
    except IndexError as error:
        raise ValueError("recorded rollout prompt has an unexpected format") from error


def _latex_surface(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(
        r"\s+",
        "",
        str(text).replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac"),
    )


def _completion_category(completion: str, reference: str) -> str:
    prediction = extract_last_boxed(completion)
    if prediction is None or prediction == "":
        return "unfinished"
    target = extract_last_boxed(reference)
    # Validate completed answers while keeping the diagnostic stratification
    # broad enough to sample reliably.  The current recorded dump contains
    # almost no boxed-but-wrong examples.
    _ = grade_math(prediction, target) or (
        _latex_surface(prediction) == _latex_surface(target)
    )
    return "boxed"


def load_audit_samples(
    *,
    graphs_jsonl: Path,
    generations_dir: Path,
    samples_per_group: int,
    categories: tuple[str, ...] = ("boxed", "unfinished"),
) -> list[AuditSample]:
    rows = load_math_cot_20k(heldout_fraction=0.0)["train"]
    if not categories or any(
        category not in {"boxed", "unfinished"} for category in categories
    ):
        raise ValueError("categories must contain boxed and/or unfinished")
    question_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        question_rows.setdefault(str(row["question"]).strip(), []).append(
            (index, row)
        )
    guides: dict[int, str] = {}
    for line in graphs_jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not record.get("accepted"):
            continue
        guides[int(record["example_index"])] = remove_graph_identifiers(
            render_graph_scaffold(_graph(record["graph"]))
        )

    candidates: dict[str, list[AuditSample]] = {
        "boxed": [],
        "unfinished": [],
    }
    for path in sorted(generations_dir.glob("generations_step_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["generations"]:
            problem = _question_from_prompt(str(record["prompt"]))
            matching_rows = [
                item
                for item in question_rows[problem]
                if item[0] in guides
            ]
            if not matching_rows:
                raise ValueError(
                    "recorded rollout has no accepted guide for its dataset "
                    f"row: {problem[:120]!r}"
                )
            source_index, row = min(matching_rows, key=lambda item: item[0])
            completion = str(record["completion"])
            category = _completion_category(completion, str(row["response"]))
            if category in categories:
                candidates[category].append(
                    AuditSample(
                        source_index=source_index,
                        category=category,
                        problem=problem,
                        guide=guides[source_index],
                        completion=completion,
                    )
                )

    selected: list[AuditSample] = []
    for category in categories:
        ordered_with_repeats = sorted(
            candidates[category],
            key=lambda item: (
                hashlib.sha256(
                    f"{category}:{item.source_index}".encode("utf-8")
                ).hexdigest(),
                item.source_index,
            ),
        )
        ordered: list[AuditSample] = []
        seen_source_indices: set[int] = set()
        for item in ordered_with_repeats:
            if item.source_index in seen_source_indices:
                continue
            seen_source_indices.add(item.source_index)
            ordered.append(item)
        if len(ordered) < samples_per_group:
            raise ValueError(
                f"need {samples_per_group} {category} samples, found {len(ordered)}"
            )
        selected.extend(ordered[:samples_per_group])
    return selected


def _render_prompt(tokenizer, problem: str, auxiliary: str) -> str:
    view = build_finod_teacher_views(
        problem=problem,
        guide=auxiliary,
        answer="unused",
    )["guide"]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": view}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _base_prompt(tokenizer, problem: str) -> str:
    view = build_finod_teacher_views(
        problem=problem,
        guide="unused",
        answer="unused",
    )["base"]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": view}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _selected_logits(
    model,
    tokenizer,
    *,
    prompt: str,
    completion_ids: torch.Tensor,
    rollout_positions: torch.Tensor,
) -> torch.Tensor:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    sequence = torch.cat((prompt_ids, completion_ids), dim=1)
    attention_mask = torch.ones_like(sequence)
    prediction_positions = prompt_ids.shape[1] - 1 + rollout_positions
    with torch.inference_mode():
        outputs = model(
            input_ids=sequence,
            attention_mask=attention_mask,
            logits_to_keep=prediction_positions,
        )
    return outputs.logits[0].float().cpu()


def _center(probability: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return direction - (probability * direction).sum(
        dim=-1, keepdim=True
    )


def _fisher_inner(
    probability: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    return (probability * left * right).sum(dim=-1)


def _cosine(
    probability: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    numerator = _fisher_inner(probability, left, right)
    left_energy = _fisher_inner(probability, left, left)
    right_energy = _fisher_inner(probability, right, right)
    return numerator / (left_energy * right_energy).clamp_min(1e-30).sqrt()


def _residual_fraction(
    probability: torch.Tensor,
    signal: torch.Tensor,
    nuisance: torch.Tensor,
) -> torch.Tensor:
    residual = _fisher_residual(probability, signal, nuisance)
    signal_energy = _fisher_inner(probability, signal, signal)
    residual_energy = _fisher_inner(probability, residual, residual)
    return residual_energy / signal_energy.clamp_min(1e-30)


def _fisher_residual(
    probability: torch.Tensor,
    signal: torch.Tensor,
    nuisance: torch.Tensor,
) -> torch.Tensor:
    nuisance_energy = _fisher_inner(probability, nuisance, nuisance)
    coefficient = _fisher_inner(
        probability, signal, nuisance
    ) / nuisance_energy.clamp_min(1e-30)
    return _center(
        probability, signal - coefficient.unsqueeze(-1) * nuisance
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = load_audit_samples(
        graphs_jsonl=args.graphs_jsonl,
        generations_dir=args.generations_dir,
        samples_per_group=args.samples_per_group,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).eval().to("cuda")

    records: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        shuffled_guide = samples[(sample_index + 1) % len(samples)].guide
        completion_ids = tokenizer(
            sample.completion,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(model.device)
        token_count = completion_ids.shape[1]
        if token_count < args.positions:
            raise ValueError("recorded completion is shorter than audit positions")
        rollout_positions = torch.linspace(
            0,
            token_count - 1,
            steps=args.positions,
            device=model.device,
        ).round().long().unique(sorted=True)

        base_logits = _selected_logits(
            model,
            tokenizer,
            prompt=_base_prompt(tokenizer, sample.problem),
            completion_ids=completion_ids,
            rollout_positions=rollout_positions,
        )
        true_logits = _selected_logits(
            model,
            tokenizer,
            prompt=_render_prompt(tokenizer, sample.problem, sample.guide),
            completion_ids=completion_ids,
            rollout_positions=rollout_positions,
        )
        generic_logits = _selected_logits(
            model,
            tokenizer,
            prompt=_render_prompt(tokenizer, sample.problem, GENERIC_GUIDE),
            completion_ids=completion_ids,
            rollout_positions=rollout_positions,
        )
        shuffled_logits = _selected_logits(
            model,
            tokenizer,
            prompt=_render_prompt(tokenizer, sample.problem, shuffled_guide),
            completion_ids=completion_ids,
            rollout_positions=rollout_positions,
        )

        probability = base_logits.softmax(dim=-1)
        true_direction = _center(probability, true_logits - base_logits)
        generic_direction = _center(probability, generic_logits - base_logits)
        shuffled_direction = _center(
            probability, shuffled_logits - base_logits
        )
        true_style_residual = _fisher_residual(
            probability, true_direction, generic_direction
        )
        shuffled_style_residual = _fisher_residual(
            probability, shuffled_direction, generic_direction
        )
        true_energy = _fisher_inner(
            probability, true_direction, true_direction
        )
        actual_tokens = completion_ids[0].index_select(
            0, rollout_positions
        ).cpu()
        for offset, rollout_position in enumerate(
            rollout_positions.detach().cpu().tolist()
        ):
            top_values, top_indices = true_direction[offset].topk(8)
            records.append(
                {
                    "source_index": sample.source_index,
                    "category": sample.category,
                    "rollout_position": rollout_position,
                    "normalized_position": (
                        rollout_position / max(token_count - 1, 1)
                    ),
                    "true_energy": float(true_energy[offset]),
                    "true_generic_cosine": float(
                        _cosine(
                            probability[offset : offset + 1],
                            true_direction[offset : offset + 1],
                            generic_direction[offset : offset + 1],
                        )[0]
                    ),
                    "true_shuffled_cosine": float(
                        _cosine(
                            probability[offset : offset + 1],
                            true_direction[offset : offset + 1],
                            shuffled_direction[offset : offset + 1],
                        )[0]
                    ),
                    "style_residual_shuffled_cosine": float(
                        _cosine(
                            probability[offset : offset + 1],
                            true_style_residual[offset : offset + 1],
                            shuffled_style_residual[offset : offset + 1],
                        )[0]
                    ),
                    "generic_residual_fraction": float(
                        _residual_fraction(
                            probability[offset : offset + 1],
                            true_direction[offset : offset + 1],
                            generic_direction[offset : offset + 1],
                        )[0]
                    ),
                    "shuffled_residual_fraction": float(
                        _residual_fraction(
                            probability[offset : offset + 1],
                            true_direction[offset : offset + 1],
                            shuffled_direction[offset : offset + 1],
                        )[0]
                    ),
                    "actual_token": tokenizer.decode(
                        [int(actual_tokens[offset])]
                    ),
                    "actual_token_shift": float(
                        true_direction[offset, actual_tokens[offset]]
                    ),
                    "actual_token_style_residual_shift": float(
                        true_style_residual[
                            offset, actual_tokens[offset]
                        ]
                    ),
                    "top_positive_tokens": [
                        {
                            "token": tokenizer.decode([int(token)]),
                            "shift": float(value),
                        }
                        for value, token in zip(
                            top_values.tolist(),
                            top_indices.tolist(),
                            strict=True,
                        )
                    ],
                }
            )

    aggregate: dict[str, Any] = {}
    for category in ("all", "boxed", "unfinished"):
        selected_records = (
            records
            if category == "all"
            else [record for record in records if record["category"] == category]
        )
        aggregate[category] = {
            "positions": len(selected_records),
            **{
                key: _mean(
                    [float(record[key]) for record in selected_records]
                )
                for key in (
                    "true_energy",
                    "true_generic_cosine",
                    "true_shuffled_cosine",
                    "style_residual_shuffled_cosine",
                    "generic_residual_fraction",
                    "shuffled_residual_fraction",
                    "actual_token_shift",
                    "actual_token_style_residual_shift",
                )
            },
        }
    return {
        "schema_version": 1,
        "model": args.model,
        "model_revision": args.model_revision,
        "samples": [asdict(sample) | {"completion": "<omitted>"} for sample in samples],
        "aggregate": aggregate,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-jsonl", type=Path, required=True)
    parser.add_argument("--generations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--positions", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--model-revision",
        default="1cfa9a7208912126459214e8b04321603b3df60c",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_group < 1 or args.positions < 2:
        raise SystemExit("samples-per-group must be positive and positions >= 2")
    report = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
