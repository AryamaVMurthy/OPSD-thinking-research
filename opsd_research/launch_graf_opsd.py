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
    if requested_steps != str(config["max_steps"]):
        if requested_steps != "5" or smoke_steps != "5":
            raise SystemExit("--max_steps must match config except explicit five-step smoke")
    for flag in ("--student_thinking", "--teacher_thinking", "--fixed_teacher", "--use_peft"):
        if flag not in sys.argv:
            raise SystemExit(f"required flag is missing: {flag}")
    # GRAF-Lite routing is enabled only after a graph-cache manifest exists.
    # C0 uses the same reliable upstream objective with a longer rollout.
    if config["graph_mode"] != "disabled":
        manifest = os.environ.get("GRAF_GRAPH_CACHE_MANIFEST")
        if not manifest or not Path(manifest).is_file():
            raise SystemExit("graph-routed candidates require GRAF_GRAPH_CACHE_MANIFEST")
    return config


def main() -> None:
    config = _validate_invocation()
    # Reuse the tested compatibility and memory hooks without loosening the
    # historical reproduction's validation contract.
    from . import launch_official_opsd as official

    os.environ.setdefault("OPSD_DATASET_REVISION", str(config["dataset_revision"]))
    official._install_dataset_redirect()
    upstream = Path(__file__).resolve().parents[1] / "third_party" / "opsd"
    sys.path.insert(0, str(upstream))
    official._install_structured_dataset_compat()
    official._install_exact_jsd_chunking()
    official._install_tail_logits_loss()
    official._install_final_generation_flush()
    runpy.run_path(str(upstream / "opsd_train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
