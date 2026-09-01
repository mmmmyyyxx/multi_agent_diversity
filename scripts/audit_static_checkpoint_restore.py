from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.checkpoint import restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.solver_headroom_multimodel_support import (
    RUN_ROOT, entrants, read_json, run_dir, sha256_file, write_json,
)


def audit_one(entry: dict) -> dict:
    source = run_dir(entry["key"], "STATIC")
    meta = read_json(source / "run_meta.json")
    checkpoint_path = source / "training_checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    before_hash = sha256_file(checkpoint_path)
    values = dict(meta["config"])
    values.update({
        "out_dir": str((RUN_ROOT / "offline_restore" / entry["key"]).resolve()),
        "shared_solver_cache_path": str(
            (RUN_ROOT / "offline_restore" / entry["key"] / "unused.sqlite").resolve()
        ),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "preserve_final_checkpoint": False,
    })
    system = PromptEnsembleOptimizationSystem(Config.from_flat(**values))
    train = _load(
        system.cfg.data.train_path,
        system.cfg.data.train_size,
        system.cfg.data.dataset_format,
    )
    system.set_run_identity(RunIdentity(**checkpoint["run_identity"]))
    system.proposal_memory_run_id = str(checkpoint["proposal_memory_run_id"])
    system.fixed_probe = system.build_probe(
        train[: min(len(train), system.cfg.evaluation.candidate_eval_pool_size)]
    )
    restore_checkpoint(system, checkpoint)
    expected_hash = str(
        meta["final_state_selection"]["selected_team_prompt_state_hash"]
    )
    checks = {
        "routing_disabled": system.protocol.service_routing_enabled is False,
        "saved_eligibility_empty": checkpoint["cached_responsibility_eligibility"] == {},
        "restored_eligibility_empty": system.cached_responsibility_eligibility == {},
        "restored_assignments_empty": all(
            not rows for rows in system.cached_responsibility_assignments.values()
        ),
        "team_hash_match": system.team_prompt_state_hash() == expected_hash,
        "active_profiles_match": [
            [asdict(answer) for answer in profile] for profile in system.active_profiles
        ] == checkpoint["active_profiles"],
        "checkpoint_unchanged": sha256_file(checkpoint_path) == before_hash,
    }
    return {
        "key": entry["key"],
        "solver_model": entry["model"],
        "checks": checks,
        "pass": all(checks.values()),
        "saved_eligibility_count": len(checkpoint["cached_responsibility_eligibility"]),
        "restored_eligibility_count": len(system.cached_responsibility_eligibility),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=RUN_ROOT / "offline_restore_audit.json",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("fresh audit output required")
    rows = [audit_one(entry) for entry in entrants()]
    payload = {
        "audit_version": "static_checkpoint_restore_semantics_v1",
        "gate": "PASS" if len(rows) == 6 and all(row["pass"] for row in rows) else "FAIL",
        "run_count": len(rows),
        "rows": rows,
        "api_calls": 0,
        "validation_evaluations": 0,
        "test_calls": 0,
        "static_reruns": 0,
        "generic_runs": 0,
    }
    write_json(args.out, payload)
    print(json.dumps({
        "gate": payload["gate"], "run_count": len(rows),
        "api_calls": 0, "test_calls": 0,
    }, indent=2))
    raise SystemExit(0 if payload["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
