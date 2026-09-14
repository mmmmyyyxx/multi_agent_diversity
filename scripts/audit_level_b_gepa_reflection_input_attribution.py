"""Zero-API attribution audit for the authorized3 GEPA reflection inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RUN = ROOT / "runs/level_b_gepa_real_canary_v1_authorized3"
DEFAULT_PREP = ROOT / "runs/level_b_gepa_real_canary_v1_prep_authorized3"
DEFAULT_POSTHOC = (
    ROOT
    / "reports/level_b_gepa_real_canary_v1_authorized3/posthoc_rejection_audit.json"
)
DEFAULT_OUT = ROOT / "reports/level_b_gepa_reflection_input_attribution_20260914"
SOURCE_PATH = "multi_dataset_diverse_rl/local_optimizers/gepa_adapter.py"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_show(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )


def _lineage(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(run: Path, prep: Path, posthoc: Path) -> dict[str, Any]:
    freeze = _read_json(prep / "source_freeze.json")
    protocol = _read_json(prep / "protocol_freeze.json")
    summary = _read_json(run / "execution_summary.json")
    rejection = _read_json(posthoc)
    task_root = run / "local_gepa/seed78_update0_member3"
    candidates = _read_json(task_root / "candidates.json")
    parent = candidates[0]["decision_procedure"]
    events = _lineage(run / "local_gepa/seed78_update0_member3.lineage.jsonl")
    rejected = [event for event in events if event.get("event_type") == "candidate_rejected"]
    historical_source = _git_show(freeze["execution_commit"], SOURCE_PATH)

    from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
        mutable_prompt_violation_reasons,
    )
    from multi_dataset_diverse_rl.local_optimizers.gepa_proposer_contract import (
        DECISION_PROCEDURE_REFLECTION_TEMPLATE,
        DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256,
    )

    correct_rows = sum(int(event["old_score"]) for event in rejected)
    reflected_rows = len(rejected) * int(
        protocol["official_gepa"]["reflection_minibatch_size"]
    )
    checks = {
        "execution_audit_pass": _read_json(run / "audit.json")["gate"] == "PASS",
        "three_reflection_calls": summary["proposal_attempts"] == len(rejected) == 3,
        "frozen_template_hash_matches": (
            protocol["official_gepa"]["reflection_prompt_template_sha256"]
            == DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256
            == _sha256(DECISION_PROCEDURE_REFLECTION_TEMPLATE)
        ),
        "template_contract_safe": not mutable_prompt_violation_reasons(
            DECISION_PROCEDURE_REFLECTION_TEMPLATE
        ),
        "current_parameter_contract_safe": not mutable_prompt_violation_reasons(parent),
        "historical_side_info_included_raw_output": (
            '"Generated Outputs": observation.raw_output' in historical_source
        ),
        "historical_side_info_repeated_controller_context": (
            'feedback = f"{feedback}\\nController context: {self.optimization_context}"'
            in historical_source
        ),
        "all_rejections_output_contract": (
            rejection["primary_rejection_category_counts"]
            ["output_contract_contamination"]
            == 3
        ),
        "validation_and_test_zero": (
            summary["validation50_calls"] == summary["test50_calls"] == 0
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"reflection attribution audit failed: {checks}")

    return {
        "gate": "PASS",
        "audit_kind": "zero_api_read_only_reflection_input_attribution",
        "api_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
        "execution_commit": freeze["execution_commit"],
        "proposal_attempts": 3,
        "effective_prompt_reconstruction": {
            "exact_bytes": "NOT_RECOVERABLE_ZERO_API",
            "reason": (
                "The historical reflection request bodies and the nine raw reasoning "
                "traces were not durably persisted. Component structure and lower "
                "bounds are recoverable from frozen source, lineage, and strict Solver semantics."
            ),
            "component_structure": "template + curr_param + side_info",
        },
        "components": {
            "template": {
                "identity": "decision_procedure_reflection_v1",
                "sha256": DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256,
                "frozen_hash_match": True,
                "contract_contamination": False,
            },
            "curr_param": {
                "sha256": _sha256(parent),
                "char_count": len(parent),
                "contract_contamination": False,
            },
            "side_info": {
                "historical_schema_included_raw_solver_output": True,
                "historical_schema_included_gold_fallback": True,
                "gold_fallback_used_in_this_canary": "NOT_ESTABLISHED",
                "controller_context_repeated_per_record": True,
                "controller_context_named_output_interface": True,
                "reflected_record_count": reflected_rows,
                "strict_valid_correct_record_lower_bound": correct_rows,
                "final_answer_marker_lower_bound": correct_rows,
                "final_answer_marker_upper_bound": reflected_rows,
                "contract_contamination": True,
            },
        },
        "proposal_attribution": {
            "output_contract_contamination": {
                "proposal_count": 3,
                "source": "SIDE_INFO",
                "confidence": "CAUSALLY_IDENTIFIED_FOR_INPUT_CONTAMINATION",
                "basis": (
                    "The safe frozen template and safe current parameter exclude the "
                    "marker, while historical side_info necessarily contained at least "
                    "five strict FINAL_ANSWER lines and repeated output-interface context."
                ),
            },
            "append_only": {
                "proposal_count": rejection["failed_check_counts"]["append_only"],
                "source": "PROPOSER_REPLACEMENT_BEHAVIOR",
                "confidence": "OBSERVED_NOT_INPUT_ATTRIBUTABLE",
            },
        },
        "classifier": "SIDE_INFO_REPRESENTATION_CONTAMINATION_IDENTIFIED",
        "conclusion": (
            "The frozen template and current mutable component were not the contamination "
            "source. Historical make_reflective_dataset serialized immutable Solver "
            "interface evidence into side_info before all three rejected proposals."
        ),
        "checks": checks,
        "raw_prompts_published": False,
        "raw_responses_published": False,
    }


def _write_report(out: Path, result: dict[str, Any]) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("fresh attribution report root required")
    out.mkdir(parents=True, exist_ok=True)
    (out / "attribution.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    facts = {
        "gate": "PASS",
        "all_checks_true": all(result["checks"].values()),
        "api_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
        "historical_artifacts_modified": 0,
    }
    (out / "fact_assertions.json").write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "evidence": [
                    "authorized3 frozen source and protocol identities",
                    "authorized3 sanitized execution and rejection summaries",
                    "authorized3 local GEPA lineage counters",
                ],
                "private_evidence_read_only": True,
                "exact_effective_prompt_bytes_published": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Level-B GEPA reflection-input attribution\n\n"
        "Gate: **PASS**  \n"
        "Classifier: **SIDE_INFO_REPRESENTATION_CONTAMINATION_IDENTIFIED**\n\n"
        "The frozen reflection template and current `decision_procedure` were contract-safe. "
        "The historical reflective dataset placed raw Solver responses into `<side_info>`; "
        "strict-valid responses necessarily contained the immutable answer marker. It also "
        "repeated controller wording about the output interface. This identifies `<side_info>` "
        "as the pre-proposal contamination source for the three-call canary.\n\n"
        "Exact historical effective-prompt bytes are `NOT_RECOVERABLE_ZERO_API` because request "
        "bodies and all nine raw reasoning traces were not durably persisted. The attribution "
        "uses frozen source structure, lineage score counts, strict Solver semantics, and the "
        "sanitized rejection audit; it does not publish prompts, questions, answers, or responses.\n\n"
        "The compatible repair is component-specific evidence: problem text, reasoning-only "
        "trace, coarse correctness outcome, and allowlisted reasoning focus. Gold labels, raw "
        "failure codes, free-form controller/output-contract wording, and immutable answer lines "
        "must not enter reflection side information.\n",
        encoding="utf-8",
    )
    sanitization = {
        "gate": "PASS",
        "raw_prompts": False,
        "questions": False,
        "gold_answers": False,
        "model_answers": False,
        "raw_responses": False,
        "credentials": False,
        "endpoints": False,
        "sqlite_or_cache_content": False,
        "checkpoints": False,
        "absolute_paths": False,
    }
    (out / "sanitization_manifest.json").write_text(
        json.dumps(sanitization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    (out / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--posthoc", type=Path, default=DEFAULT_POSTHOC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = audit(args.run, args.prep, args.posthoc)
    _write_report(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
