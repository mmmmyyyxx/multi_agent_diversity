from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    gate = json.loads((ROOT / "protocol_gate.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "variant_metrics.json").read_text(encoding="utf-8"))
    cost = json.loads((ROOT / "cost_summary.json").read_text(encoding="utf-8"))
    with (ROOT / "case_metrics.csv").open(encoding="utf-8", newline="") as handle:
        cases = list(csv.DictReader(handle))
    assert gate["gate"] == "PASS" and gate["blocker_count"] == 0
    assert gate["observed_cell_count"] == gate["complete_cell_count"] == 9
    assert gate["commit_count"] == gate["validation_calls"] == gate["test_calls"] == 0
    assert len(cases) == 9 and all(row["parent_state_equal"] == "true" for row in cases)
    for variant, row in metrics["variants"].items():
        selected = [case for case in cases if case["variant"] == variant]
        assert len(selected) == 3
        assert sum(int(case["valid_candidates"]) for case in selected) == row["valid_candidates"]
        assert sum(int(case["feasible_candidates"]) for case in selected) == row["feasible_candidates"]
    assert cost["provider_attempt_count"] == sum(cost["role_attempt_counts"].values())
    assert cost["total_tokens"] == cost["prompt_tokens"] + cost["completion_tokens"]
    assert cost["terminal_infrastructure_failure_count"] == 0
    print("v16 fixed-parent sanitized report assertions: PASS")


if __name__ == "__main__":
    main()
