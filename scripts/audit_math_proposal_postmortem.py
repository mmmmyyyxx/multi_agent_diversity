"""Zero-provider forensic audit, independent of production experiment execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.benchmarks.legacy.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.evaluation.proposal_contract_audit import audit_proposals, paired_identity_audit
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("audit output must stay under ignored runs/")
    protocol = read_json(args.protocol)
    if protocol["real_api_authorized"] is not False or protocol["provider_call_budget"] != 0:
        raise ValueError("postmortem is strictly zero API")
    parent = args.parent_root.resolve()
    files = {}
    for row in protocol["inputs"]:
        path = (parent / row["relative_path"]).resolve()
        if not path.is_relative_to(parent) or digest(path) != row["sha256"]:
            raise ValueError("frozen parent input hash mismatch")
        files[row["name"]] = path
    refs = {}
    for name, row in protocol["tracked_inputs"].items():
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != row["sha256"]:
            raise ValueError("frozen tracked dependency hash mismatch")
        refs[name] = path
    contract = read_json(refs["parent_binding"])
    binding = execution_binding(ROOT, contract)
    optimize = binding.examples("optimize")
    examples = {x.item.input_id: LocalEvidenceExample(
        x.item.input_id, binding.benchmark().format_input(x.item), x.reference,
    ) for x in optimize}
    broker = RequestBroker(contract=contract, transport=None, arm="A1", seed=81)
    result = audit_proposals(
        read_jsonl(files["provider_trace"]), read_jsonl(files["trajectory"]),
        read_jsonl(files["search_ledger"]), examples,
        read_json(refs["owner_annotations"])["rows"], broker,
    )
    receipt = read_json(files["search_receipt"])
    if (receipt["search_closed_forever"] is not True
            or digest(files["trajectory"]) != receipt["trajectory_sha256"]
            or digest(files["initial_state"]) != digest(files["final_state"])):
        raise ValueError("closed search or unchanged-team evidence mismatch")
    paired = read_jsonl(files["paired_scores"])
    paired_identity = paired_identity_audit(paired, 100)
    validation = read_json(files["validation_summary"])
    result["zero_intervention"] = {
        "initial_final_state_bytes_equal": True, "changed_members": 0, **paired_identity,
        "validation_final_new_physical": sum(r["kind"] == "SUCCESS" and r["stage"] == "validation_final"
                                             for r in read_jsonl(files["validation_ledger"])),
        "historical_signal": validation["signal"], "search_restart_authorized": False,
        "baseline_coverage_scope": "DESCRIPTIVE_POST_SEARCH_NO_METHOD_DESIGN_FEEDBACK",
    }
    result["input_hashes"] = {name: digest(path) for name, path in files.items()}
    result["tracked_dependency_hashes"] = {name: digest(path) for name, path in refs.items()}
    result["scope"] = {"optimize_examples_read": len(examples), "validation_score_rows_read": len(paired),
                       "validation_raw_content_read": False, "test_raw_reads": 0, "test_model_calls": 0,
                       "new_provider_calls": 0, "new_provider_tokens": 0, "runtime_policies_changed": False}
    # Check again after replay: all previous evidence and the global accounting
    # ledger are read-only inputs, including their original immutable charges.
    if any(digest(files[r["name"]]) != r["sha256"] for r in protocol["inputs"]):
        raise ValueError("parent evidence changed during audit")
    atomic_write_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "logical_rows"}}, sort_keys=True))


if __name__ == "__main__":
    main()
