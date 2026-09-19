"""Layer-1-only pilot harness; preparation has no API authorization.

The callable parent harness takes an already frozen LocalOptimizationTask and
injected evaluator/model. It never constructs a team or obtains new parents.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import import_frozen_gepa, verify_frozen_gepa
from multi_dataset_diverse_rl.local_optimizers.proposal_telemetry import (
    ExactProposalStopper, MetricCeilingReached, ProposalTelemetry, descriptive_rate, wilson,
)

EXPERIMENT_ID = "local_gepa_acceptance_rate_pilot_v1"
MANIFEST = ROOT / "experiments/manifests/local_gepa_acceptance_rate_pilot_v1.yaml"
FORMAL_RUN = ROOT / "runs/local_gepa_acceptance_rate_pilot_v1_attempt1"


async def run_parent(*, task, evaluator, reflection_lm, accounting_reader,
                     run_root: Path, proposal_quota: int, skip_allowance: int):
    """Execute only official local optimization; dependencies own transport.

    This is also the fake-provider regression entrypoint. No provider is created
    by this module. The execution gate must be satisfied by any real caller.
    """
    captured = []

    def factory(*args, **kwargs):
        callback = ProposalTelemetry(*args, **kwargs, search_examples=task.search_examples,
                                     metric_ceiling=task.budget.max_metric_calls)
        captured.append(callback)
        return callback

    def optimize(**kwargs):
        kwargs["stop_callbacks"] = ExactProposalStopper(kwargs["callbacks"][0], proposal_quota, skip_allowance)
        return import_frozen_gepa().optimize(**kwargs)

    optimizer = GEPALocalPromptOptimizer(
        evaluator=evaluator, reflection_lm=reflection_lm, accounting_reader=accounting_reader,
        run_root=run_root, optimize_fn=optimize, callback_factory=factory,
    )
    termination = "official_stop"
    try:
        await optimizer.optimize(task)
    except MetricCeilingReached:
        termination = "metric_ceiling_before_batch"
    callback = captured[0]
    summary = callback.scientific_summary(proposal_quota)
    summary["parent_task_id"] = task.task_id
    member = task.task_id.rsplit("member", 1)[-1]
    summary["target_member"] = int(member) if member.isdigit() else None
    for proposal in summary["proposals"]:
        proposal["parent_task_id"] = task.task_id
        proposal["target_member"] = summary["target_member"]
    if termination == "metric_ceiling_before_batch":
        summary["quota_status"] = "PARENT_PROPOSAL_QUOTA_INCOMPLETE"
    summary["termination"] = termination
    # Preserve evaluations even if safety termination occurred before a child batch.
    summary["evaluation_batches"] = callback.evaluations
    return summary


def pooled_summary(parents):
    keys = parents[0]["funnel"] if parents else []
    accepted = sum(p["primary"]["numerator"] for p in parents)
    evaluated = sum(p["primary"]["denominator"] for p in parents)
    parent_accepted = sum(p["at_least_one_accepted"] for p in parents)
    return {"parents": parents,
            "funnel": {key: sum(p["funnel"][key] for p in parents) for key in keys},
            "primary": descriptive_rate(accepted, evaluated),
            "per_parent_acceptance_rates": {p.get("parent_task_id", str(index)): p["primary"]
                                             for index, p in enumerate(parents)},
            "parent_level_at_least_one_accepted": descriptive_rate(parent_accepted, len(parents)),
            "naive_iid_reference_only": wilson(accepted, evaluated),
            "iid_inference_permitted": False,
            "complete": bool(parents) and all(p["quota_status"] == "COMPLETE" for p in parents)}


def preflight():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    errors = validate_manifest(manifest, schema)
    if manifest["api_authorization"]["authorized"] is not False:
        errors.append("preparation must remain unauthorized")
    if manifest["artifacts"]["preregistration"]["sha256"] != preregistration_hash(manifest):
        errors.append("preregistration mismatch")
    return {"preparation_gate": "PASS" if not errors else "FAIL", "errors": errors,
            "execution_gate": "PARENT_CATALOG_INSUFFICIENT" if not manifest["design"]["selected_parents"] else "AUTHORIZATION_REQUIRED",
            "formal_run_root_absent": not FORMAL_RUN.exists(), "engine": verify_frozen_gepa(),
            "real_api_calls": 0, "heldout_Validation50_calls": 0, "Test50_calls": 0}


def require_execution_ready(manifest):
    # Check before creating a directory, loading data, or constructing providers.
    if manifest["api_authorization"].get("authorized") is not True:
        raise PermissionError("API_AUTHORIZATION_REQUIRED")
    if not manifest["design"].get("selected_parents"):
        raise RuntimeError("PARENT_CATALOG_INSUFFICIENT")
    raise RuntimeError("FROZEN_EXECUTION_HANDOFF_REQUIRED")


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        require_execution_ready(yaml.safe_load(MANIFEST.read_text(encoding="utf-8")))
    else:
        result = preflight()
        print(json.dumps(result, indent=2))
        raise SystemExit(result["preparation_gate"] != "PASS")


if __name__ == "__main__":
    main()
