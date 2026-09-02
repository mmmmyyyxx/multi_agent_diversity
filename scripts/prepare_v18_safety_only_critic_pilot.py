from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.tcs import PreviousUpdateOutcome, serialize_context
from scripts.generic_m20_probe_support import profile
from scripts.prepare_v18_m2f_trigger_extension_pilot import (
    answers_for_prompt, profile_block, question_rows, reconstruct_parent_prompts,
    stable_correct_hashes,
)
from scripts.v18_safety_only_critic_pilot_support import (
    HISTORICAL_ROOT, canonical_hash, read_json, read_jsonl, sha256_file, write_json,
)
from scripts.generic_m20_probe_support import system_for


SOURCE_FILES = (
    "scripts/v18_safety_only_critic_pilot_support.py",
    "scripts/prepare_v18_safety_only_critic_pilot.py",
    "scripts/run_v18_safety_only_critic_pilot.py",
    "scripts/audit_v18_safety_only_critic_pilot.py",
    "scripts/analyze_v18_safety_only_critic_pilot.py",
    "tests/test_v18_safety_only_critic_pilot.py",
    "experiments/v18_safety_only_critic_pilot_20260902/DESIGN_SPEC.md",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def select_cases() -> list[tuple[int, str, int, int]]:
    selected = []
    for seed in (59, 60, 61):
        decisions = read_jsonl(HISTORICAL_ROOT / f"seed{seed}" / "HYBRID_BASE" / "candidate_decisions.jsonl")
        for status in ("blocked", "passed"):
            choices = []
            for decision in decisions:
                for branch in decision["branches"]:
                    funnel = branch["funnel"]
                    matches = (int(funnel["student_calls"]) == 0 and int(funnel["critic_semantic_rejections"]) > 0) if status == "blocked" else int(funnel["student_calls"]) > 0
                    if matches:
                        choices.append((int(decision["update_index"]), int(branch["target_agent_id"])))
            if not choices:
                raise ValueError(f"no {status} case for seed {seed}")
            update, target = min(choices)
            selected.append((seed, status, update, target))
    return selected


def prepare(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh preparation root required")
    if git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must be clean")
    out.mkdir(parents=True)
    cases = []
    for seed, status, update, target in select_cases():
        run = HISTORICAL_ROOT / f"seed{seed}" / "HYBRID_BASE"
        meta = read_json(run / "run_meta.json")
        decisions = read_jsonl(run / "candidate_decisions.jsonl")
        decision = next(row for row in decisions if int(row["update_index"]) == update)
        branch = next(row for row in decision["branches"] if int(row["target_agent_id"]) == target)
        prompts = reconstruct_parent_prompts(decisions, update_index=update, shared_prompt=str(meta["config"]["shared_prompt"]))
        cache = run.parent / "_shared_solver_cache.sqlite"
        skeleton = {"base_config": meta["config"], "parent_prompts": prompts, "source_seed": seed, "questions": [], "initial_profiles": [], "active_profiles": [], "accepted_state_count": 0, "team_state_version": 0, "stable_correct_question_hashes_by_agent": {}, "previous_update_outcome_by_agent": {}, "target_agent_id": target, "active_lane": branch["active_lane"]}
        system = system_for(skeleton, setting="experimental_v16_efficacy_g_matched", out_dir=out / "read_only", cache_path=out / "unused.sqlite")
        train = system.build_probe(_load(system.cfg.data.train_path, system.cfg.data.train_size, system.cfg.data.dataset_format))
        validation = system.build_probe(_load(system.cfg.data.val_path, system.cfg.data.val_size, system.cfg.data.dataset_format))
        system.fixed_probe = train
        initial = [answers_for_prompt(system=system, cache_path=cache, prompt=str(meta["config"]["shared_prompt"]), examples=train.examples, historical_meta=meta) for _ in range(5)]
        active = [answers_for_prompt(system=system, cache_path=cache, prompt=prompt, examples=train.examples, historical_meta=meta) for prompt in prompts]
        system.initial_profiles = list(initial)
        system.active_profiles = list(active)
        stable = stable_correct_hashes(system=system, examples=train.examples, profiles=initial)
        system.stable_correct_question_hashes_by_agent = {int(agent): set(values) for agent, values in stable.items()}
        system.previous_update_outcomes[target] = PreviousUpdateOutcome()
        context, _ = system._proposal_context(target, prompts[target], set(branch["assigned_question_hashes"]))
        historical_context = next(row for row in read_jsonl(run / "tcs_context_history.jsonl") if int(row["update_index"]) == update and int(row["target_agent_id"]) == target)
        context_hash = canonical_hash(json.loads(serialize_context(context)))
        if context_hash != historical_context["proposal_context_hash"]:
            raise ValueError(f"historical context reconstruction mismatch: seed{seed} {status}")
        validation_profiles = [answers_for_prompt(system=system, cache_path=cache, prompt=prompt, examples=validation.examples, historical_meta=meta) for prompt in prompts]
        case_id = f"seed{seed}_{status}_u{update}_t{target}"
        cases.append({
            "case_id": case_id, "source_seed": seed, "historical_status": status,
            "source_update_index": update, "target_agent_id": target,
            "parent_team_hash": decision["parent_team_hash"], "parent_prompts": prompts,
            "assigned_question_hashes": sorted(branch["assigned_question_hashes"]),
            "active_lane": branch["active_lane"], "base_config": meta["config"],
            "questions": question_rows(train.examples), "validation_questions": question_rows(validation.examples),
            "initial_profiles": profile_block(train.examples, initial), "active_profiles": profile_block(train.examples, active),
            "validation_parent_profiles": profile_block(validation.examples, validation_profiles),
            "stable_correct_question_hashes_by_agent": stable, "accepted_state_count": sum(bool(row.get("accepted_prompt_hash")) for row in decisions if int(row["update_index"]) < update),
            "team_state_version": sum(bool(row.get("accepted_prompt_hash")) for row in decisions if int(row["update_index"]) < update),
            "previous_update_outcome_by_agent": {}, "historical_context_hash": historical_context["proposal_context_hash"],
        })
    registry = {"registry_version": "v18_safety_only_critic_pilot_v1", "execution_commit": git("rev-parse", "HEAD"), "case_selection_rule": "per_seed_earliest_hybrid_blocked_and_passed", "case_count": 6, "arms": ["canonical_llm", "deterministic_safety_only"], "source_candidates_per_branch": 2, "revision_per_valid_source": 1, "validation_after_train_freeze": True, "test_enabled": False, "cases": cases}
    registry["registry_content_hash"] = canonical_hash({k: v for k, v in registry.items() if k != "registry_content_hash"})
    write_json(out / "private_registry.json", registry)
    freeze = {"execution_commit": registry["execution_commit"], "registry_sha256": sha256_file(out / "private_registry.json"), "files": [{"path": item, "sha256": sha256_file(ROOT / item)} for item in SOURCE_FILES]}
    write_json(out / "source_freeze.json", freeze)
    gate = {"gate": "PASS", "case_count": 6, "blocked_case_count": 3, "passed_case_count": 3, "context_reconstruction_match_count": 6, "api_calls": 0, "test_calls": 0}
    write_json(out / "preflight.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    if ROOT.resolve() not in args.out.resolve().parents: raise SystemExit("project-local output required")
    print(json.dumps(prepare(args.out.resolve()), indent=2))


if __name__ == "__main__": main()
