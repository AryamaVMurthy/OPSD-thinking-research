"""Small, auditable command-line controller for GRAF candidate registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .graf_contract import Candidate, append_ledger, validate_candidate


def _register(args: argparse.Namespace) -> None:
    policy = load_config(args.policy).data
    candidate_config = load_config(args.config).data
    candidate = Candidate(
        candidate_id=args.candidate_id,
        parent_id=args.parent_id,
        stage=args.stage,
        hypothesis=args.hypothesis,
        config=candidate_config,
        changed_fields=tuple(args.changed_field),
    )
    validate_candidate(candidate, set(policy["allowed_mutations"]))
    event = {
        "event": "candidate_registered",
        "candidate": json.loads(candidate.canonical_payload()),
        "candidate_sha256": candidate.sha256,
        "policy": str(args.policy.resolve()),
    }
    digest = append_ledger(args.ledger, event)
    print(json.dumps({"candidate_sha256": candidate.sha256, "ledger_event_sha256": digest}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Register immutable GRAF autoresearch candidates.")
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--policy", type=Path, required=True)
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--ledger", type=Path, required=True)
    register.add_argument("--candidate-id", required=True)
    register.add_argument("--parent-id")
    register.add_argument("--stage", required=True)
    register.add_argument("--hypothesis", required=True)
    register.add_argument("--changed-field", action="append", required=True)
    register.set_defaults(func=_register)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
