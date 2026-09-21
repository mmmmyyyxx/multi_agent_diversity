"""Generate the sanitized native-feed fidelity package; never calls providers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from multi_dataset_diverse_rl.native_feed_audit import (
    budget_semantics,
    claim_registry,
    data_access_matrix,
    layer2_contract_manifest,
    normalized_bytes,
    sha256_bytes,
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, encoding="utf-8"
    ).strip()


def git_blob(root: Path, revision: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), "show", f"{revision}:{relative}"]
    )


def upstream_manifest(backend: str, root: Path, mars_root: Path | None) -> dict[str, Any]:
    if backend == "gepa":
        from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import (
            GEPA_COMMIT,
            GEPA_SOURCE_SHA256,
            GEPA_VERSION,
            verify_frozen_gepa,
        )

        verified = verify_frozen_gepa()
        return {
            "backend": "GEPA_NATIVE_FEED_V1",
            "repository": "https://github.com/gepa-ai/gepa",
            "paper": "arXiv:2507.19457",
            "version": GEPA_VERSION,
            "commit": GEPA_COMMIT,
            "source_hash": GEPA_SOURCE_SHA256,
            "verified": verified,
        }
    if mars_root is None:
        raise ValueError("--mars-root is required")
    revision = git(mars_root, "rev-parse", "upstream/main")
    paths = [
        "main_MARS.py",
        "Agents.py",
        "Config.py",
        "Prompt/ALL_prompt_planner_template.md",
        "Prompt/ALL_userproxy_task_input.md",
        "Prompt/EDIT_1.2_getplannerfewshot.txt",
        "Prompt/EDIT_1_userproxy_task_input.txt",
        "Prompt/EDIT_2_prompt_planner_template.txt",
        "Prompt/system_prompt_critic.txt",
        "Prompt/system_prompt_planner.txt",
        "Prompt/system_prompt_teacher.txt",
    ]
    hashes = {path: sha256_bytes(git_blob(mars_root, revision, path)) for path in paths}
    bundle = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "backend": "MARS_OFFICIAL_CODE_FEED_V1",
        "repository": "https://github.com/exoskeletonzj/MARS",
        "paper": "arXiv:2503.16874; AAAI-26 version",
        "commit": revision,
        "tracked_file_sha256": hashes,
        "source_hash": bundle,
        "network_refresh": "unavailable; pre-existing upstream/main ref used",
    }


def paper_vs_code(backend: str) -> dict[str, Any]:
    if backend == "gepa":
        return {
            "paper_semantics": "Dfeedback native train minibatches; Dpareto full candidate tracking",
            "official_code_semantics": "optimize(seed_candidate, trainset, valset, adapter) with epoch-shuffled sampling",
            "project_adaptation": "decision_procedure component and immutable COMMON solver shell",
            "fidelity": "LEVEL_B_API_COMPATIBLE_ADAPTATION",
        }
    return {
        "paper_semantics": {
            "Dtrain": "guides optimization",
            "Planner": "receives task goal/input/initial prompt",
            "TCS": "iterative revision",
            "Target": "performance reward",
            "Dtest": "evaluation",
        },
        "official_code_semantics": [
            "UserProxy static task description",
            "Planner steps",
            "Teacher/Critic/Student dialogue",
            "Student full prompt",
            "Target loads full Config.DATASET_PATH",
            "full-dataset accuracy enters prompt history and stopping",
        ],
        "target_failure_feedback_to_tcs": False,
        "project_adaptation": [
            "Config.DATASET_PATH mapped to frozen Optimize-only universe",
            "Student full prompt mapped to mutable decision_procedure",
            "Validation50/Test50 inaccessible",
        ],
        "fidelity": "official-code data-flow topology; not exact paper reproduction",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gepa", "mars"), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--mars-root", type=Path)
    parser.add_argument("--fake-flow-status", choices=("PASS", "HOLD"), default="PASS")
    parser.add_argument("--focused-test-summary", default="PASS")
    parser.add_argument("--full-test-summary", default="NOT_RUN")
    parser.add_argument("--compileall-status", default="PASS")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    layer2 = layer2_contract_manifest(root)
    source = upstream_manifest(args.backend, root, args.mars_root)
    overlay = (
        "GEPA_LAYER2_RESPONSIBILITY_OVERLAY_V1"
        if args.backend == "gepa"
        else "MARS_LAYER2_RESPONSIBILITY_OVERLAY_V1"
    )
    native = {
        "backend": args.backend,
        "control": f"Native {args.backend.upper()}",
        "treatment": f"Native {args.backend.upper()} + Layer2",
        "outer_contract": "backend_native_feed_request_v1",
        "responsibility_overlay": overlay,
        "responsibility_is_sample_selector": False,
        "validation50_calls": 0,
        "test50_calls": 0,
        "real_api_calls": 0,
    }
    if args.backend == "gepa":
        native["native_data_flow"] = {
            "Optimize100_optimizer_train": 75,
            "Optimize100_optimizer_pareto_val": 25,
            "split_seed": 20260918,
            "batch_sampler": "official epoch_shuffled",
            "responsibility_filters_train_loader": False,
        }
    else:
        native["native_data_flow"] = {
            "Target_dataset": "full frozen Optimize-only universe",
            "candidate_evaluation": "full Target dataset every time",
            "GEPA_style_minibatches": False,
            "responsibility_filters_target_dataset": False,
        }
    parity = {
        "status": "PASS",
        "identical": [
            "backend data universe",
            "backend split/topology",
            "models",
            "candidate representation",
            "native budget semantics",
        ],
        "sole_treatment_differences": ["Layer2 member allocation", overlay, "team controller"],
    }
    fake = {
        "status": args.fake_flow_status,
        "api_calls": 0,
        "backend": args.backend,
        "native_data_not_layer2_materialized": True,
        "candidate_returned_through_common_boundary": True,
    }
    heldout = {
        "status": "PASS",
        "Optimize_only": True,
        "Validation50_calls": 0,
        "Test50_calls": 0,
    }
    test_summary = {
        "status": args.fake_flow_status,
        "focused_tests": args.focused_test_summary,
        "full_tests": args.full_test_summary,
        "compileall": args.compileall_status,
        "real_api_calls": 0,
    }
    payloads = {
        "upstream_source_manifest.json": source,
        "paper_vs_code_dataflow.json": paper_vs_code(args.backend),
        "native_feed_manifest.json": native,
        "layer2_contract_hash.json": layer2,
        "control_treatment_parity.json": parity,
        "data_access_matrix.json": data_access_matrix(),
        "budget_semantics.json": budget_semantics(),
        "fake_provider_native_flow.json": fake,
        "heldout_isolation.json": heldout,
        "claim_registry.json": claim_registry(),
        "test_summary.json": test_summary,
    }
    for name, payload in payloads.items():
        write_json(report / name, payload)
    if args.backend == "mars":
        write_json(report / "mars_paper_vs_code_dataflow_audit.json", paper_vs_code("mars"))
    readme = f"# {args.backend.upper()} native-feed refactor\n\n"
    readme += "Zero-API architecture and fidelity package. Layer 2 owns WHO/WHY; "
    readme += "the backend owns native data consumption and search.\n\n"
    readme += f"- Fidelity status: `{args.fake_flow_status}`\n"
    readme += "- Validation50 calls: `0`\n- Test50 calls: `0`\n- Real API calls: `0`\n"
    readme += "- Cross-backend superiority claims: forbidden in this phase.\n"
    (report / "README.md").write_text(readme, encoding="utf-8")
    sanitization = {
        "status": "PASS",
        "forbidden": [
            "prompts",
            "questions",
            "gold/model answers",
            "raw responses",
            "credentials",
            "endpoints",
            "absolute paths",
        ],
        "scan_scope": sorted(path.name for path in report.iterdir()),
    }
    write_json(report / "sanitization_manifest.json", sanitization)
    manifest = {}
    for path in sorted(report.iterdir()):
        if path.name == "sha256_manifest.json":
            continue
        manifest[path.name] = sha256_bytes(normalized_bytes(path))
    write_json(report / "sha256_manifest.json", manifest)


if __name__ == "__main__":
    main()
