from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--report_dir", default="reports/v17_failure_decomposition_20260820")
    args = parser.parse_args()
    repo = Path(args.workspace).resolve()
    report = (repo / args.report_dir).resolve()
    protocol = json.loads((report / "protocol_gate.json").read_text(encoding="utf-8"))
    diagnosis = json.loads((report / "diagnosis_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((report / "sanitized_manifest.json").read_text(encoding="utf-8"))
    assert protocol["audit_protocol_gate"] == "PASS"
    assert protocol["row_level_reconstruction_gate"] == "PASS"
    assert all(value == 0 for value in protocol["api_model_solver_optimizer_evaluator_call_counts"].values())
    assert diagnosis["S1_S2_test_transition_net"] == diagnosis["S1_S2_test_vote_net"]
    assert diagnosis["low_api_escalation_required"] is False
    assert diagnosis["method_changed"] is False
    assert manifest["sanitization"] == "PASS"
    print("PASS deterministic V17 failure-decomposition smoke")


if __name__ == "__main__":
    main()
