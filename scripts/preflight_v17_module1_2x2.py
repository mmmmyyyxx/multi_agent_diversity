from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multi_dataset_diverse_rl.tcs import (
    AccuracyDiagnosisContext,
    PeerStateDiagnosisContext,
)
from v17_module1_2x2_support import (
    CELLS,
    context_hashes,
    immutable_state_hash,
    probe_system,
    targets_for,
)


def preflight(registry: dict[str, Any], scratch: Path) -> dict[str, Any]:
    errors: list[str] = []
    context_rows = []
    for case in registry.get("cases", []):
        parent_hashes = set()
        for cell in CELLS:
            for target in targets_for(case, cell):
                system = probe_system(
                    case, cell,
                    out_dir=scratch / case["case_id"] / cell / str(target),
                    cache_path="",
                    target=target,
                )
                before = immutable_state_hash(system)
                hashes = context_hashes(case, cell, target)
                context, _ = system._proposal_context(
                    target, system.agents[target].current_prompt, hashes
                )
                after = immutable_state_hash(system)
                parent_hashes.add(system.team_prompt_state_hash())
                expected = AccuracyDiagnosisContext if cell in {"A", "B"} else PeerStateDiagnosisContext
                if not isinstance(context, expected):
                    errors.append(f"context_type:{case['case_id']}:{cell}:{target}")
                if before != after:
                    errors.append(f"state_mutation:{case['case_id']}:{cell}:{target}")
                if (cell in {"A", "B"}) != (len(hashes) == 0):
                    errors.append(f"context_hash_contract:{case['case_id']}:{cell}:{target}")
                context_rows.append({
                    "case_id": case["case_id"], "cell": cell, "target": target,
                    "context_type": type(context).__name__,
                    "assigned_hash_count": len(hashes),
                })
        if parent_hashes != {case["parent_team_hash"]}:
            errors.append(f"parent_hash:{case['case_id']}")
        if case["w1_target_ids"] != case["w1_independent_replay_target_ids"]:
            errors.append(f"w1_replay:{case['case_id']}")
    if len({row["parent_team_hash"] for row in registry.get("cases", [])}) != 6:
        errors.append("distinct_parent_hashes")
    for key, expected in (
        ("case_count", 6), ("cell_count", 24), ("branch_count", 48),
        ("source_candidate_budget", 96),
    ):
        if int(registry.get(key, -1)) != expected:
            errors.append(key)
    if any(int(registry.get(key, -1)) for key in (
        "phase_a_api_calls", "phase_a_validation_calls"
    )) or registry.get("final_test_enabled") is not False:
        errors.append("phase_a_isolation")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "api_calls": 0, "model_calls": 0, "validation_calls": 0,
        "test_calls": 0, "case_count": 6, "cell_count": 24,
        "branch_count": 48, "context_checks": context_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    result = preflight(registry, args.scratch.resolve())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
