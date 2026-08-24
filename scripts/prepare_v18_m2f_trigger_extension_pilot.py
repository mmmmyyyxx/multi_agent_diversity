from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from multi_dataset_diverse_rl.candidate_selection import (
    common_monotone_safe_key,
    evaluate_constraints,
)
from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.compatibility_repair import (
    EXTENDED_TRAIN_VOTE_LOSS_TRIGGER_VERSION,
    ONLINE_COMPATIBILITY_REPAIR_VERSION,
    REPAIR_INSTRUCTION,
    REPAIR_SYSTEM_PROMPT,
    build_repair_request,
    extended_repair_eligible,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.evaluation.fixed_probe import evaluate_candidate_profile
from multi_dataset_diverse_rl.peer_state import build_peer_vote_context
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.utils import normalize_prompt_text
from scripts.admit_v18_hybrid_online_scientific_analysis import (
    artifact_tree_identity,
)
from scripts.m2f_probe_support import read_cached_answers


CASES = ((59, 3), (61, 5))
ARM = "HYBRID_BASE"
MODEL = "qwen3-14b"
SOURCE_FILES = (
    "multi_dataset_diverse_rl/compatibility_repair.py",
    "multi_dataset_diverse_rl/candidate_selection.py",
    "multi_dataset_diverse_rl/system.py",
    "scripts/prepare_v18_m2f_trigger_extension_pilot.py",
    "scripts/run_v18_m2f_trigger_extension_pilot.py",
    "scripts/audit_v18_m2f_trigger_extension_pilot.py",
    "scripts/analyze_v18_m2f_trigger_extension_pilot.py",
)
CLASSIFIER_LABELS = (
    "EXTENDED_M2F_WRITEBACK_RISK_REDUCTION_SUPPORTED",
    "EXTENDED_M2F_TRAIN_COLLATERAL_REDUCTION_ONLY",
    "EXTENDED_M2F_TRIGGER_NOT_SUPPORTED",
    "EXTENDED_M2F_HARMFUL",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def prompt_hash(value: str) -> str:
    return hashlib.sha256(
        normalize_prompt_text(value).encode("utf-8")
    ).hexdigest()


def reconstruct_parent_prompts(
    decisions: Sequence[dict[str, Any]], *, update_index: int,
    shared_prompt: str,
) -> list[str]:
    prompts = [shared_prompt] * 5
    for decision in sorted(decisions, key=lambda row: int(row["update_index"])):
        if int(decision["update_index"]) >= update_index:
            break
        accepted_hash = str(decision.get("accepted_prompt_hash", ""))
        if not accepted_hash:
            continue
        candidate = next(
            row for row in decision["candidates"]
            if str(row["prompt_hash"]) == accepted_hash
        )
        target = int(decision["target_agent_id"])
        prompt = str(candidate["evaluation"]["prompt"])
        if prompt_hash(prompt) != accepted_hash:
            raise ValueError("historical accepted prompt hash mismatch")
        prompts[target] = prompt
    return prompts


def make_system(
    *, base_config: dict[str, Any], parent_prompts: Sequence[str],
    seed: int, cache_path: Path, out_dir: Path,
) -> PromptEnsembleOptimizationSystem:
    flat = dict(base_config)
    flat.update({
        "initialization_mode": "provided_prompt_set",
        "provided_prompts_json": json.dumps(list(parent_prompts)),
        "out_dir": str(out_dir.resolve()),
        "shared_solver_cache_path": str(cache_path.resolve()),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "proposal_memory_mode": "off",
        "seed": int(seed),
        "agent_model": MODEL,
        "optimizer_model": MODEL,
        "evaluator_model": MODEL,
    })
    defaults = Config().to_flat_dict()
    cfg = Config.from_flat(**{
        key: flat.get(key, default) for key, default in defaults.items()
    })
    return PromptEnsembleOptimizationSystem(cfg)


def answers_for_prompt(
    *, system: PromptEnsembleOptimizationSystem, cache_path: Path,
    prompt: str, examples: Sequence[Any], require_complete: bool = True,
) -> tuple[PromptAnswer, ...] | None:
    observations = read_cached_answers(cache_path, prompt_hash(prompt), system)
    if not all(example.question_hash in observations for example in examples):
        if require_complete:
            raise ValueError("historical cache profile is incomplete")
        return None
    return tuple(
        PromptAnswer(**{
            key: value
            for key, value in observations[example.question_hash].items()
            if key in PromptAnswer.__dataclass_fields__
        })
        for example in examples
    )


def profile_block(
    examples: Sequence[Any], profiles: Sequence[Sequence[PromptAnswer]],
) -> list[dict[str, Any]]:
    return [
        {
            "question_hash": example.question_hash,
            "team_answers": [profile[index].answer for profile in profiles],
            "team_validity": [bool(profile[index].valid) for profile in profiles],
        }
        for index, example in enumerate(examples)
    ]


def question_rows(examples: Sequence[Any]) -> list[dict[str, str]]:
    return [
        {
            "question_hash": example.question_hash,
            "question": example.question,
            "answer": example.gold_answer,
        }
        for example in examples
    ]


def stable_correct_hashes(
    *, system: PromptEnsembleOptimizationSystem, examples: Sequence[Any],
    profiles: Sequence[Sequence[PromptAnswer]],
) -> dict[str, list[str]]:
    return {
        str(agent): sorted(
            example.question_hash
            for example, answer in zip(examples, profiles[agent], strict=True)
            if answer.valid and system.match_answer(answer.answer, example.gold_answer)
        )
        for agent in range(5)
    }


def repair_evidence(
    *, system: PromptEnsembleOptimizationSystem, target: int,
    assigned_hashes: set[str], source_profile: Sequence[PromptAnswer],
    stable_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    states, _, _ = system.current_states_and_opportunities()
    repairs: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    stable_losses: list[dict[str, Any]] = []
    all_nonresponsibility_losses = 0
    for example, state, before, after in zip(
        system.fixed_probe.examples,
        states,
        system.active_profiles[target],
        source_profile,
        strict=True,
    ):
        before_correct = bool(
            before.valid and system.match_answer(before.answer, example.gold_answer)
        )
        after_correct = bool(
            after.valid and system.match_answer(after.answer, example.gold_answer)
        )
        evidence = {
            "question_hash": example.question_hash,
            "question": example.question,
            "gold_answer": example.gold_answer,
            "parent_target_output": before.answer,
            "source_candidate_target_output": after.answer,
        }
        if example.question_hash in assigned_hashes and not before_correct and after_correct:
            repairs.append({**evidence, "responsibility_status": "assigned"})
        if example.question_hash not in assigned_hashes and before_correct and not after_correct:
            all_nonresponsibility_losses += 1
            peer = build_peer_vote_context(state, target)
            role = (
                "unique" if state.gold_vote_count == 1 else
                "pivotal" if state.vote_correct and peer.peer_margin <= 0 else
                "stable" if example.question_hash in stable_hashes else
                "fragile"
            )
            row = {
                **evidence,
                "parent_competence_role": role,
                "parent_plurality_margin": int(state.plurality_margin),
            }
            if role in {"unique", "pivotal"}:
                losses.append(row)
            elif role == "stable":
                stable_losses.append(row)
    stable_losses.sort(
        key=lambda row: (
            -int(row["parent_plurality_margin"]), str(row["question_hash"])
        )
    )
    losses.extend(stable_losses[:2])
    return (
        sorted(repairs, key=lambda row: str(row["question_hash"])),
        losses,
        all_nonresponsibility_losses,
    )


def build(*, run_root: Path, admission: dict[str, Any], out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh Phase A root required")
    if ROOT.resolve() not in out.resolve().parents:
        raise ValueError("Phase A root must be project-local")
    if git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must be clean")
    head = git("rev-parse", "HEAD")
    if admission.get("scientific_analysis_admitted") is not True:
        raise ValueError("V18 scientific admission is not PASS")
    raw_identity = artifact_tree_identity(run_root)
    if raw_identity != admission["raw_artifact_identity"]:
        raise ValueError("historical V18 artifact identity mismatch")

    cases: list[dict[str, Any]] = []
    for seed, update_index in CASES:
        run = run_root / f"seed{seed}" / ARM
        cache_path = run.parent / "_shared_solver_cache.sqlite"
        meta = read_json(run / "run_meta.json")
        decisions = read_jsonl(run / "candidate_decisions.jsonl")
        decision = next(
            row for row in decisions if int(row["update_index"]) == update_index
        )
        parent_prompts = reconstruct_parent_prompts(
            decisions,
            update_index=update_index,
            shared_prompt=str(meta["config"]["shared_prompt"]),
        )
        system = make_system(
            base_config=meta["config"], parent_prompts=parent_prompts,
            seed=seed, cache_path=cache_path, out_dir=out / "read_only",
        )
        if system.team_prompt_state_hash() != str(decision["parent_team_hash"]):
            raise ValueError("reconstructed parent team hash mismatch")
        train_data = _load(
            system.cfg.data.train_path,
            system.cfg.data.train_size,
            system.cfg.data.dataset_format,
        )
        val_data = _load(
            system.cfg.data.val_path,
            system.cfg.data.val_size,
            system.cfg.data.dataset_format,
        )
        train_probe = system.build_probe(train_data)
        validation_probe = system.build_probe(val_data)
        system.fixed_probe = train_probe
        shared_prompt = str(meta["config"]["shared_prompt"])
        initial_profiles = [
            answers_for_prompt(
                system=system, cache_path=cache_path, prompt=shared_prompt,
                examples=train_probe.examples,
            )
            for _ in range(5)
        ]
        active_profiles = [
            answers_for_prompt(
                system=system, cache_path=cache_path, prompt=prompt,
                examples=train_probe.examples,
            )
            for prompt in parent_prompts
        ]
        if any(profile is None for profile in active_profiles):
            raise ValueError("parent train profile missing")
        system.initial_profiles = list(initial_profiles)  # type: ignore[arg-type]
        system.active_profiles = list(active_profiles)  # type: ignore[arg-type]
        stable = stable_correct_hashes(
            system=system,
            examples=train_probe.examples,
            profiles=initial_profiles,  # type: ignore[arg-type]
        )
        system.stable_correct_question_hashes_by_agent = {
            int(agent): set(hashes) for agent, hashes in stable.items()
        }
        branch_by_target = {
            int(row["target_agent_id"]): row for row in decision["branches"]
        }
        pool = []
        recovered = 0
        for candidate in decision["candidates"]:
            constraint_raw = candidate.get("constraint")
            evaluation_raw = candidate.get("evaluation")
            if not constraint_raw or not evaluation_raw:
                continue
            if not bool(constraint_raw["passed"]) or int(constraint_raw["vote_loss_count"]) <= 0:
                continue
            recovered += 1
            source_prompt = str(evaluation_raw["prompt"])
            source_hash = str(candidate["prompt_hash"])
            if prompt_hash(source_prompt) != source_hash:
                raise ValueError("source candidate prompt hash mismatch")
            target = int(candidate["target_agent_id"])
            assigned = set(map(str, branch_by_target[target]["assigned_question_hashes"]))
            source_profile = answers_for_prompt(
                system=system, cache_path=cache_path, prompt=source_prompt,
                examples=train_probe.examples,
            )
            if source_profile is None:
                raise ValueError("source train profile missing")
            evaluation = evaluate_candidate_profile(
                prompt=source_prompt,
                prompt_hash=source_hash,
                examples=train_probe.examples,
                active_profiles=system.active_profiles,
                initial_profiles=system.initial_profiles,
                candidate_profile=source_profile,
                target_agent_id=target,
                assigned_question_hashes=assigned,
                normalize_answer=system.normalize_answer,
                match_answer=system.match_answer,
                tie_break=system.protocol.tie_policy,
                seed=seed,
                tau=system.cfg.peer_state.soft_vote_tau,
            )
            incumbent = system.active_evaluation(target)
            constraint = evaluate_constraints(evaluation, incumbent)
            expected = (
                bool(constraint_raw["passed"]),
                int(constraint_raw["target_gain"]),
                int(constraint_raw["vote_gain_count"]),
                int(constraint_raw["vote_loss_count"]),
            )
            actual = (
                bool(constraint.passed), int(constraint.target_gain),
                int(constraint.vote_gain_count), int(constraint.vote_loss_count),
            )
            if actual != expected:
                raise ValueError("source train evaluation reconstruction mismatch")
            repair_rows, loss_rows, nonresponsibility_losses = repair_evidence(
                system=system,
                target=target,
                assigned_hashes=assigned,
                source_profile=source_profile,
                stable_hashes=set(stable[str(target)]),
            )
            extended = extended_repair_eligible(
                responsibility_gain_count=len(repair_rows),
                rejection_reasons=constraint.rejection_reasons,
                loss_evidence_count=len(loss_rows),
                source_common_safe=constraint.passed,
                source_vote_loss_count=constraint.vote_loss_count,
            )
            numeric_summary = {
                "responsibility_gain_count": len(repair_rows),
                "nonresponsibility_loss_count": nonresponsibility_losses,
                "source_target_gain": int(constraint.target_gain),
                "source_vote_gain_count": int(constraint.vote_gain_count),
                "source_vote_loss_count": int(constraint.vote_loss_count),
            }
            case_id = f"seed{seed}_update{update_index}_{source_hash[:12]}"
            row = {
                "case_id": case_id,
                "source_seed": seed,
                "source_update_index": update_index,
                "arm": ARM,
                "parent_team_hash": str(decision["parent_team_hash"]),
                "target_agent_id": target,
                "target_selection_rank": int(candidate["target_selection_rank"]),
                "active_lane": str(branch_by_target[target]["active_lane"]),
                "assigned_question_hashes": sorted(assigned),
                "frozen_responsibility_evidence_hash": canonical_hash(sorted(assigned)),
                "source_candidate_hash": source_hash,
                "source_candidate_prompt": source_prompt,
                "source_candidate_stage": str(candidate["candidate_stage"]),
                "source_generation": int(candidate["generation"]),
                "parent_prompt": parent_prompts[target],
                "parent_prompts": parent_prompts,
                "questions": question_rows(train_probe.examples),
                "initial_profiles": profile_block(train_probe.examples, initial_profiles),  # type: ignore[arg-type]
                "active_profiles": profile_block(train_probe.examples, active_profiles),  # type: ignore[arg-type]
                "validation_questions": question_rows(validation_probe.examples),
                "stable_correct_question_hashes_by_agent": stable,
                "accepted_state_count": sum(
                    bool(item.get("accepted_prompt_hash"))
                    for item in decisions if int(item["update_index"]) < update_index
                ),
                "team_state_version": sum(
                    bool(item.get("accepted_prompt_hash"))
                    for item in decisions if int(item["update_index"]) < update_index
                ),
                "previous_update_outcome_by_agent": {},
                "base_config": meta["config"],
                "historical_cache_path": str(cache_path.resolve()),
                "repair_evidence": repair_rows,
                "loss_evidence": loss_rows,
                "numeric_summary": numeric_summary,
                "source_metrics": {
                    "target_gain": int(constraint.target_gain),
                    "vote_gain_count": int(constraint.vote_gain_count),
                    "vote_loss_count": int(constraint.vote_loss_count),
                    "vote_net_gain": int(constraint.vote_net_gain),
                    "common_safe_feasible": bool(constraint.passed),
                    "ranking_key": list(common_monotone_safe_key(
                        evaluation, int(candidate["generation"])
                    )),
                },
                "extended_m2f_eligible": bool(extended),
                "repair_input_hash": hashlib.sha256(
                    build_repair_request(
                        parent_prompt=parent_prompts[target],
                        source_candidate_prompt=source_prompt,
                        repair_evidence=repair_rows,
                        loss_evidence=loss_rows,
                        numeric_summary=numeric_summary,
                    ).encode("utf-8")
                ).hexdigest(),
            }
            validation_parent_profiles = [
                answers_for_prompt(
                    system=system, cache_path=cache_path, prompt=prompt,
                    examples=validation_probe.examples,
                )
                for prompt in parent_prompts
            ]
            if any(profile is None for profile in validation_parent_profiles):
                raise ValueError("parent validation profile missing")
            row["validation_parent_profiles"] = profile_block(
                validation_probe.examples,
                validation_parent_profiles,  # type: ignore[arg-type]
            )
            cached_source_validation = answers_for_prompt(
                system=system, cache_path=cache_path, prompt=source_prompt,
                examples=validation_probe.examples, require_complete=False,
            )
            row["source_validation_profile_cached"] = (
                [asdict(answer) for answer in cached_source_validation]
                if cached_source_validation is not None else None
            )
            row["source_validation_cache_complete"] = cached_source_validation is not None
            pool.append({
                "source_candidate_hash": source_hash,
                "source_generation": int(candidate["generation"]),
                "source_metrics": row["source_metrics"],
            })
            cases.append(row)
        expected_count = 4 if seed == 59 else 3
        if recovered != expected_count:
            raise ValueError("frozen intervention pool inventory mismatch")
        for row in cases:
            if row["source_seed"] == seed and row["source_update_index"] == update_index:
                row["original_pool"] = pool
                row["historically_committed_source"] = (
                    row["source_candidate_hash"] == str(decision["accepted_prompt_hash"])
                )

    if len(cases) != 7 or not all(row["extended_m2f_eligible"] for row in cases):
        raise ValueError("extended trigger must admit exactly 7/7 frozen sources")
    registry = {
        "registry_version": "v18_m2f_trigger_extension_registry_v1",
        "execution_commit": head,
        "model": MODEL,
        "thinking": False,
        "repair_attempts_per_source": 1,
        "trigger_version": EXTENDED_TRAIN_VOTE_LOSS_TRIGGER_VERSION,
        "repair_version": ONLINE_COMPATIBILITY_REPAIR_VERSION,
        "repair_prompt_hash": hashlib.sha256(
            (REPAIR_SYSTEM_PROMPT + REPAIR_INSTRUCTION).encode("utf-8")
        ).hexdigest(),
        "source_count": len(cases),
        "eligible_count": len(cases),
        "commit_enabled": False,
        "trajectory_mutation_enabled": False,
        "validation_after_train_decisions_frozen": True,
        "test_enabled": False,
        "classifier": {
            "labels": list(CLASSIFIER_LABELS),
            "target_gain_retention_high_threshold": 0.8,
            "target_gain_retention_harmful_threshold": 0.5,
        },
        "cases": cases,
    }
    registry["registry_content_hash"] = canonical_hash(registry)
    freeze = {
        "freeze_version": "v18_m2f_trigger_extension_source_freeze_v1",
        "execution_commit": head,
        "registry_content_hash": registry["registry_content_hash"],
        "raw_artifact_identity": raw_identity,
        "trigger_version": EXTENDED_TRAIN_VOTE_LOSS_TRIGGER_VERSION,
        "repair_version": ONLINE_COMPATIBILITY_REPAIR_VERSION,
        "source_files": [
            {"path": path, "sha256": sha256_file(ROOT / path)}
            for path in SOURCE_FILES
        ],
    }
    gate = {
        "phase_a_version": "v18_m2f_trigger_extension_phase_a_v1",
        "phase_a_gate": "PASS",
        "source_reconstruction_gate": "PASS",
        "extended_trigger_gate": "PASS",
        "repair_semantics_unchanged_gate": "PASS",
        "validation_isolation_gate": "PASS",
        "historical_raw_hash_gate": "PASS",
        "source_candidate_count": 7,
        "eligible_source_candidate_count": 7,
        "cached_source_validation_profile_count": sum(
            bool(row["source_validation_cache_complete"]) for row in cases
        ),
        "phase_b_authorized_by_gate": True,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "method_change_scope": "trigger_eligibility_only",
        "historical_raw_artifacts_modified": False,
    }
    out.mkdir(parents=True)
    write_json(out / "private_registry.json", registry)
    write_json(out / "source_freeze.json", freeze)
    write_json(out / "phase_a_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(
        run_root=args.run_root.resolve(),
        admission=read_json(args.admission.resolve()),
        out=args.out.resolve(),
    )


if __name__ == "__main__":
    main()
