"""Launch an isolated experimental OPSD/GRAF candidate.

The historical launcher intentionally accepts only the reproduced 200-step,
1,024-token recipe.  This launcher preserves all of its reproducibility and
memory patches while validating the separately versioned GRAF candidate
contract.
"""

from __future__ import annotations

import os
import runpy
import sys
import json
from pathlib import Path

from .config import load_config


def _argument(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _validate_invocation() -> dict[str, object]:
    raw_config = os.environ.get("GRAF_CONFIG")
    if not raw_config:
        raise SystemExit("GRAF_CONFIG must name an isolated graf_train config")
    config = load_config(raw_config).data
    if config["kind"] != "graf_train":
        raise SystemExit("GRAF_CONFIG must have kind=graf_train")
    expected = {
        "--model_name_or_path": config["model"],
        "--model_revision": config["model_revision"],
        "--student_model_revision": config["model_revision"],
        "--max_completion_length": str(config["max_completion_length"]),
    }
    for flag, value in expected.items():
        if _argument(flag) != value:
            raise SystemExit(f"{flag} must match GRAF_CONFIG ({value!r})")
    requested_steps = _argument("--max_steps")
    smoke_steps = os.environ.get("GRAF_SMOKE_MAX_STEPS")
    full_steps = os.environ.get("GRAF_FULL_MAX_STEPS")
    if requested_steps != str(config["max_steps"]):
        is_smoke = requested_steps == "5" and smoke_steps == "5"
        is_full_confirmation = requested_steps == "200" and full_steps == "200"
        if not is_smoke and not is_full_confirmation:
            raise SystemExit("--max_steps must match config except explicit five-step smoke")
    for flag in ("--student_thinking", "--teacher_thinking", "--fixed_teacher", "--use_peft"):
        if flag not in sys.argv:
            raise SystemExit(f"required flag is missing: {flag}")
    # GRAF-Lite routing is enabled only after a graph-cache manifest exists.
    # C0 uses the same reliable upstream objective with a longer rollout.
    if config["graph_mode"] != "disabled":
        manifest = os.environ.get("GRAF_GRAPH_CACHE_MANIFEST")
        if not manifest:
            raise SystemExit("graph-routed candidates require GRAF_GRAPH_CACHE_MANIFEST")
        from .graf_cache import validate_graph_cache_manifest
        try:
            validate_graph_cache_manifest(manifest)
        except ValueError as error:
            raise SystemExit(f"invalid GRAF graph cache: {error}") from error
    return config


def main() -> None:
    config = _validate_invocation()
    # Reuse the tested compatibility and memory hooks without loosening the
    # historical reproduction's validation contract.
    from . import launch_official_opsd as official

    os.environ.setdefault("OPSD_DATASET_REVISION", str(config["dataset_revision"]))
    heldout_fraction = float(config.get("heldout_diagnostic_fraction", 0.0))
    os.environ["OPSD_HELDOUT_DIAGNOSTIC_FRACTION"] = str(heldout_fraction)
    from .training_data import (
        load_math_cot_20k,
        partition_indices,
        write_partition_manifest,
    )
    raw_dataset = load_math_cot_20k(
        str(config["dataset_revision"]), heldout_fraction=0.0
    )["train"]
    train_indices, heldout_indices = partition_indices(raw_dataset, heldout_fraction)
    manifest_path = os.environ.get("OPSD_TRAINING_PARTITION_MANIFEST")
    is_primary = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"
    if manifest_path and is_primary:
        manifest = write_partition_manifest(Path(manifest_path), raw_dataset, heldout_fraction)
    else:
        manifest = {
            "source_examples": len(raw_dataset),
            "train_examples": len(train_indices),
            "heldout_diagnostic_examples": len(heldout_indices),
            "heldout_diagnostic_fraction": heldout_fraction,
        }
    if is_primary:
        print(
            json.dumps({"event": "training_partition", **manifest}, sort_keys=True),
            flush=True,
        )
    train_index_set = set(train_indices)
    routed_targets = None
    if config["graph_mode"] == "scaffold_graph":
        from .graf_scaffold_dataset import (
            accepted_scaffolds,
            install_graph_scaffold_dataset_redirect,
        )

        available = accepted_scaffolds(os.environ["GRAF_GRAPH_CACHE_MANIFEST"])
        eligible = sorted(set(available).intersection(train_index_set))
        records = install_graph_scaffold_dataset_redirect(
            os.environ["GRAF_GRAPH_CACHE_MANIFEST"], source_indices=eligible
        )
        print(
            '{"event":"graf_answer_masked_scaffold_enabled",'
            f'"accepted_graph_records":{records}}}',
            flush=True,
        )
    elif config["graph_mode"] == "viability_routed":
        from .graf_routing import load_routing_targets
        from .graf_scaffold_dataset import install_graph_scaffold_dataset_redirect

        raw_viability_manifest = os.environ.get("GRAF_VIABILITY_MANIFEST")
        if not raw_viability_manifest:
            raise SystemExit("viability-routed candidates require GRAF_VIABILITY_MANIFEST")
        routed_targets = load_routing_targets(
            os.environ["GRAF_GRAPH_CACHE_MANIFEST"],
            raw_viability_manifest,
            min_target_margin=float(config.get("fork_threshold", 0.0)),
            min_target_information=float(config.get("fork_information_threshold", 0.0)),
            target_information_quantile=float(config.get("fork_information_quantile", 0.0)),
            information_weighting=bool(config.get("information_weighted_routing", False)),
            viability_beta_prior=float(config.get("viability_beta_prior", 0.0)),
        )
        routed_targets = {
            index: forks for index, forks in routed_targets.items()
            if index in train_index_set
        }
        active_examples = sum(bool(forks) for forks in routed_targets.values())
        if not routed_targets or not active_examples:
            raise SystemExit("viability-routed candidates require at least one active joined target")
        # Train only on identities with measured continuation viability.  This
        # makes every distributed batch exercise the GRAF loss instead of
        # silently reducing it to a shuffle-dependent sparse regularizer.
        records = install_graph_scaffold_dataset_redirect(
            os.environ["GRAF_GRAPH_CACHE_MANIFEST"],
            source_indices=routed_targets.keys(),
        )
        print(
            '{"event":"graf_viability_routing_enabled",'
            f'"accepted_graph_records":{records},'
            f'"routed_examples":{len(routed_targets)},'
            f'"active_routed_examples":{active_examples},'
            f'"fork_threshold":{float(config.get("fork_threshold", 0.0))},'
            f'"fork_information_threshold":{float(config.get("fork_information_threshold", 0.0))},'
            f'"fork_information_quantile":{float(config.get("fork_information_quantile", 0.0))},'
            f'"information_weighted_routing":{bool(config.get("information_weighted_routing", False))},'
            f'"viability_beta_prior":{float(config.get("viability_beta_prior", 0.0))}' + "}",
            flush=True,
        )
    else:
        official._install_dataset_redirect()
    upstream = Path(__file__).resolve().parents[1] / "third_party" / "opsd"
    sys.path.insert(0, str(upstream))
    official._install_structured_dataset_compat()
    official._install_exact_jsd_chunking()
    official._install_tail_logits_loss()
    if routed_targets is not None:
        official._install_nonreentrant_gradient_checkpointing()
        official._install_graf_source_index_collator()
        official._install_graf_routed_loss(
            routing_targets=routed_targets,
            branch_loss_weight=float(config["branch_loss_weight"]),
            entropy_floor_fraction=float(config["entropy_floor_weight"]),
        )
    official._install_final_generation_flush()
    runpy.run_path(str(upstream / "opsd_train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
