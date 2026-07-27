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
        is_smoke = requested_steps in {"1", "5"} and smoke_steps == requested_steps
        is_full_confirmation = requested_steps == "200" and full_steps == "200"
        if not is_smoke and not is_full_confirmation:
            raise SystemExit(
                "--max_steps must match config except an explicit one- or five-step smoke"
            )
    for flag in ("--student_thinking", "--teacher_thinking", "--fixed_teacher", "--use_peft"):
        if flag not in sys.argv:
            raise SystemExit(f"required flag is missing: {flag}")
    if config["graph_mode"] in {
        "context_dossier",
        "fluid_viability_routed",
    }:
        manifest = os.environ.get("CH_DOSSIER_MANIFEST")
        if not manifest:
            raise SystemExit(
                "dossier candidates require CH_DOSSIER_MANIFEST"
            )
        from .ch_dossier import accepted_teacher_dossiers

        try:
            accepted_teacher_dossiers(manifest)
        except ValueError as error:
            raise SystemExit(f"invalid CH dossier cache: {error}") from error
    # GRAF-Lite routing is enabled only after a graph-cache manifest exists.
    # C0 uses the same reliable upstream objective with a longer rollout.
    if config["graph_mode"] not in {
        "disabled",
        "context_dossier",
    }:
        manifest = os.environ.get("GRAF_GRAPH_CACHE_MANIFEST")
        if not manifest:
            raise SystemExit("graph-routed candidates require GRAF_GRAPH_CACHE_MANIFEST")
        from .graf_cache import validate_graph_cache_manifest
        try:
            graph_manifest = validate_graph_cache_manifest(manifest)
        except ValueError as error:
            raise SystemExit(f"invalid GRAF graph cache: {error}") from error
        if bool(config.get("teacher_graph_critique", False)) and not bool(
            graph_manifest.get("teacher_critique", False)
        ):
            raise SystemExit(
                "teacher_graph_critique requires a graph cache built with teacher critique"
            )
    return config


def _configure_token_divergence(config: dict[str, object]) -> None:
    """Select a canonical objective without changing historical GRAF runs."""
    divergence = config.get("token_divergence")
    if divergence is None:
        os.environ.pop("OPSD_TOKEN_DIVERGENCE", None)
        return
    os.environ["OPSD_TOKEN_DIVERGENCE"] = str(divergence)


def main() -> None:
    config = _validate_invocation()
    _configure_token_divergence(config)
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
    if config["graph_mode"] == "context_dossier":
        from .ch_dossier import accepted_teacher_dossiers
        from .ch_dossier_dataset import install_ch_dossier_dataset_redirect

        available = accepted_teacher_dossiers(os.environ["CH_DOSSIER_MANIFEST"])
        eligible = sorted(set(available).intersection(train_index_set))
        records = install_ch_dossier_dataset_redirect(
            os.environ["CH_DOSSIER_MANIFEST"], source_indices=eligible
        )
        print(
            '{"event":"ch_teacher_only_dossier_enabled",'
            f'"accepted_dossier_records":{records},'
            '"student_answer_context":false,'
            '"teacher_reference_context":true}',
            flush=True,
        )
    elif config["graph_mode"] == "scaffold_graph":
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
    elif config["graph_mode"] in {
        "viability_routed",
        "fluid_viability_routed",
    }:
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
            recovery_conditioned=bool(config.get("recovery_conditioned_routing", False)),
        )
        # An empty fork tuple means the measured viability target was uniform
        # and consequently supplies no routing gradient.  Excluding those
        # identities is important: otherwise a batch can silently perform
        # only the base OPSD update, making the routed objective depend on
        # shuffle order rather than its measured graph signal.
        routed_targets = {
            index: forks
            for index, forks in routed_targets.items()
            if index in train_index_set and forks
        }
        fluid_mode = config["graph_mode"] == "fluid_viability_routed"
        if fluid_mode:
            from .ch_dossier import accepted_teacher_dossiers

            dossier_indices = set(
                accepted_teacher_dossiers(
                    os.environ["CH_DOSSIER_MANIFEST"]
                )
            ).intersection(train_index_set)
            routed_targets = {
                index: forks
                for index, forks in routed_targets.items()
                if index in dossier_indices
            }
        if not routed_targets:
            raise SystemExit("viability-routed candidates require at least one active joined target")
        if fluid_mode:
            from .ch_dossier_dataset import (
                install_fluid_dossier_dataset_redirect,
            )

            # Coverage is defined by the natural teacher dossiers, not by
            # whether a graph fork happened to be informative. Every dossier
            # identity retains base OPSD; routing is a sparse auxiliary.
            records = install_fluid_dossier_dataset_redirect(
                os.environ["CH_DOSSIER_MANIFEST"],
                source_indices=sorted(dossier_indices),
            )
        else:
            # Historical fixed-cache ablations intentionally train only on
            # identities with measured continuation viability.
            records = install_graph_scaffold_dataset_redirect(
                os.environ["GRAF_GRAPH_CACHE_MANIFEST"],
                source_indices=routed_targets.keys(),
            )
        print(
            '{"event":"graf_viability_routing_enabled",'
            f'"base_opsd_records":{records},'
            f'"routed_examples":{len(routed_targets)},'
            f'"active_routed_examples":{len(routed_targets)},'
            f'"coverage_preserving":{str(fluid_mode).lower()},'
            f'"student_answer_context":false,'
            f'"teacher_reference_context":{str(fluid_mode).lower()},'
            f'"fork_threshold":{float(config.get("fork_threshold", 0.0))},'
            f'"fork_information_threshold":{float(config.get("fork_information_threshold", 0.0))},'
            f'"fork_information_quantile":{float(config.get("fork_information_quantile", 0.0))},'
            f'"information_weighted_routing":{bool(config.get("information_weighted_routing", False))},'
            f'"viability_beta_prior":{float(config.get("viability_beta_prior", 0.0))},'
            f'"recovery_conditioned_routing":{bool(config.get("recovery_conditioned_routing", False))}' + "}",
            flush=True,
        )
    else:
        official._install_dataset_redirect()
    upstream = Path(__file__).resolve().parents[1] / "third_party" / "opsd"
    sys.path.insert(0, str(upstream))
    official._install_structured_dataset_compat()
    official._install_exact_jsd_chunking()
    official._install_tail_logits_loss()
    official._install_adapter_stability_callback()
    if routed_targets is not None:
        official._install_nonreentrant_gradient_checkpointing()
        official._install_graf_source_index_collator()
        official._install_graf_routed_loss(
            routing_targets=routed_targets,
            branch_loss_weight=float(config["branch_loss_weight"]),
            entropy_floor_fraction=float(config["entropy_floor_weight"]),
        )
    official._install_auto_resume()
    official._install_final_generation_flush()
    runpy.run_path(str(upstream / "opsd_train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
