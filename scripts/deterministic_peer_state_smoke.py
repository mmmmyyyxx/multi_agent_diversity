from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem


async def solver(question: str, agent_id: int, prompt: str) -> PromptAnswer:
    if "repair-uncovered" in prompt and question == "q0":
        answer = "A"
    elif question == "q1" and agent_id < 3:
        answer = "A"
    elif question == "q2" and agent_id < 2:
        answer = "A"
    else:
        answer = "B"
    return PromptAnswer(answer, f"verify FINAL_ANSWER: {answer}", True)


async def optimizer(system_prompt: str, _user_prompt: str, _temperature: float, _max_tokens: int) -> str:
    if "Audit the Teacher" in system_prompt:
        facts = json.loads(
            system_prompt.split("DERIVED_CASE_FACTS:\n", 1)[1].split(
                "\nProposalContext:", 1,
            )[0]
        )
        return json.dumps({
            "case_fact_restatements": facts,
            "context_consistent": True,
            "sample_memorization_free": True,
            "executable_change": True,
            "internally_consistent": True,
            "preservation_rule_present": True,
            "output_contract_safe": True,
            "peer_copying_free": True,
            "stereotype_forcing_free": True,
            "non_generic_change": True,
            "blocking_reasons": [],
            "soft_concerns": [],
            "score": 1.0,
            "feedback": "approved",
        })
    if system_prompt == "Return strict JSON only.":
        return json.dumps({"candidates": [{
            "candidate_prompt": "repair-uncovered",
            "observed_failure_pattern": "uncovered ambiguity",
            "generalizable_mechanism": "premature referent selection",
            "decision_rule": "compare interpretations and verify the referent",
            "uncertainty_or_abstention_rule": "retain ambiguity when neither interpretation is excluded",
            "preservation_conditions": "retain established correct decisions",
            "evidence_summary": "uncovered cases need a referent check",
        }]})
    return json.dumps({
        "observed_failure_pattern": "uncovered ambiguity",
        "generalizable_mechanism": "premature referent selection",
        "decision_rule": "compare interpretations and verify the selected referent",
        "uncertainty_or_abstention_rule": "retain ambiguity when neither interpretation is excluded",
        "preservation_conditions": "retain established correct decisions",
        "evidence_summary": "uncovered cases need a referent check",
    })


async def run(out_dir: Path) -> dict:
    cfg = Config.from_flat(
        out_dir=str(out_dir), answer_format="option_letter", num_candidates_per_parent=1,
        stage_a_channel_top_k=1, stage_b_candidate_budget=2,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=solver, optimizer_chat=optimizer)
    system.set_run_identity(RunIdentity(
        method_version="peer_state_counterfactual_v2",
        experiment_setting="shared_peer_state_full",
        git_commit="deterministic-smoke",
        git_dirty=False,
        config_fingerprint="deterministic",
        manifest_sha256="deterministic",
        train_file_sha256="deterministic",
        val_file_sha256="deterministic",
        test_file_sha256="deterministic",
        train_question_set_hash="deterministic",
        val_question_set_hash="deterministic",
        test_question_set_hash="deterministic",
    ))
    data = [{"question": f"q{index}", "answer": "A"} for index in range(3)]
    await system.initialize_fixed_probe(data)
    before, _, _ = system.current_states_and_opportunities()
    changed = await system.update_once(0)
    after, _, _ = system.current_states_and_opportunities()
    system.history.append({"epoch": 1, "accepted": changed})
    system.flush_artifacts()
    return {
        "accepted": changed,
        "before_g": before[0].gold_vote_count,
        "after_g": after[0].gold_vote_count,
        "candidate_decisions": len(system.candidate_decisions),
        "responsibility_assignments": len(system.responsibility_assignments),
        "probe_cache_hits": system.fixed_probe.cache_hits,
        "probe_cache_misses": system.fixed_probe.cache_misses,
        "outputs": sorted(path.name for path in out_dir.iterdir()),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="peer_state_smoke_") as temporary:
        report = asyncio.run(run(Path(temporary)))
    if not report["accepted"] or report["before_g"] != 0 or report["after_g"] != 1:
        raise SystemExit(f"deterministic smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
