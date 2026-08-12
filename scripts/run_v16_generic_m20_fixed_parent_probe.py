from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generic_m20_probe_support import (
    AUTHORIZATION_ENV,
    FROZEN_DEFINITION_SHA256,
    G0,
    M20,
    M2E,
    diagnostic_payload,
    generation_hashes,
    generation_system,
    evaluation_system,
    project_local,
    responsibility_evidence_hash,
    state_hash,
    team_prompt_hash,
    tracked_source_dirty,
)
from multi_dataset_diverse_rl.system import CandidateFunnel
from multi_dataset_diverse_rl.peer_state import build_peer_vote_context
from multi_dataset_diverse_rl.tcs import (
    AccuracyDiagnosisContext,
    SingleLaneDiagnosisContext,
)


RESULT_VERSION = "v16_generic_m20_fixed_parent_cell_v1"
PROBE_VERSION = "v16_generic_m20_fixed_parent_probe_v1"


class FirstSuccessfulCallFreeze:
    """Latch the tracked-source freeze immediately after the first success."""

    def __init__(
        self,
        registry: dict[str, Any],
        source_freeze: dict[str, Any],
        marker_path: Path,
        *,
        registry_path: Path | None = None,
        definition_root: Path | None = None,
    ) -> None:
        self.registry = registry
        self.source_freeze = source_freeze
        self.marker_path = marker_path
        self.registry_path = registry_path
        self.definition_root = definition_root
        self.started = False

    def attach(self, system: Any) -> None:
        original = system.llm.chat_result

        async def guarded(*args: Any, **kwargs: Any) -> Any:
            result = await original(*args, **kwargs)
            if not self.started:
                blockers = source_freeze_gate(
                    self.registry,
                    self.source_freeze,
                    registry_path=self.registry_path,
                    definition_root=self.definition_root,
                )
                if blockers:
                    raise RuntimeError(
                        "tracked source changed at first successful provider call: "
                        + ",".join(blockers)
                    )
                atomic_write_json(self.marker_path, {
                    "marker_version": "v16_first_success_source_freeze_v1",
                    "status": "HARD",
                    "execution_commit": self.registry["execution_commit"],
                    "registry_content_hash": self.registry[
                        "registry_content_hash"
                    ],
                    "working_tree_source_hash": self.source_freeze[
                        "working_tree_source_hash"
                    ],
                    "registry_file_sha256": self.source_freeze[
                        "registry_file_sha256"
                    ],
                    "frozen_definition_sha256": self.source_freeze[
                        "frozen_definition_sha256"
                    ],
                })
                self.started = True
            return result

        system.llm.chat_result = guarded


def canonical_registry_hash(registry: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in registry.items()
        if key != "registry_content_hash"
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_fresh_cell_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"fixed-parent cell path must be fresh: {path}")


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def source_freeze_gate(
    registry: dict[str, Any],
    source_freeze: dict[str, Any],
    *,
    registry_path: Path | None = None,
    definition_root: Path | None = None,
    check_inventory: bool = True,
) -> list[str]:
    blockers: list[str] = []
    expected_commit = str(registry.get("execution_commit", ""))
    actual_commit = git_head()
    if actual_commit != expected_commit:
        blockers.append("execution_commit_mismatch")
    if source_freeze.get("execution_commit") != expected_commit:
        blockers.append("source_freeze_commit_mismatch")
    if source_freeze.get("registry_content_hash") != registry.get(
        "registry_content_hash"
    ):
        blockers.append("source_freeze_registry_mismatch")
    if source_freeze.get("source_freeze_status") != "PASS":
        blockers.append("source_freeze_status")
    if source_freeze.get("repo_dirty") is not False:
        blockers.append("source_freeze_repo_dirty")
    if registry_path is not None:
        if not registry_path.is_file():
            blockers.append("registry_file_missing")
        else:
            registry_file_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
            if registry_file_hash != source_freeze.get("registry_file_sha256"):
                blockers.append("registry_file_hash_mismatch")
            try:
                disk_registry = json.loads(registry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                blockers.append("registry_file_invalid")
            else:
                if disk_registry != registry:
                    blockers.append("registry_file_content_mismatch")
    frozen_definitions = source_freeze.get("frozen_definition_sha256")
    variants = tuple(registry.get("variants", ()))
    if not isinstance(frozen_definitions, dict) or not frozen_definitions:
        blockers.append("frozen_definition_manifest_mismatch")
    elif variants == (G0, M20) and frozen_definitions != FROZEN_DEFINITION_SHA256:
        blockers.append("frozen_definition_manifest_mismatch")
    if definition_root is not None:
        for name, expected_hash in frozen_definitions.items():
            path = definition_root / name
            if not path.is_file():
                blockers.append(f"frozen_definition_missing:{name}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                blockers.append(f"frozen_definition_hash:{name}")
    expected_source_roots = {"multi_dataset_diverse_rl", "scripts", "tests"}
    frozen_files = source_freeze.get("files")
    if not isinstance(frozen_files, list) or not frozen_files:
        blockers.append("source_freeze_file_inventory_missing")
    else:
        if len(frozen_files) != int(source_freeze.get("source_file_count", -1)):
            blockers.append("source_freeze_file_count_mismatch")
        current_tracked = (
            {
                relative.replace("\\", "/")
                for relative in subprocess.check_output(
                    ["git", "ls-files", *sorted(expected_source_roots)],
                    cwd=ROOT,
                    text=True,
                ).splitlines()
            }
            if check_inventory else None
        )
        frozen_inventory = {
            str(row.get("path", "")).replace("\\", "/")
            for row in frozen_files
        }
        if current_tracked is not None and current_tracked != frozen_inventory:
            blockers.append("source_freeze_inventory_mismatch")
        combined = hashlib.sha256()
        for row in sorted(frozen_files, key=lambda item: str(item.get("path"))):
            relative = str(row.get("path", "")).replace("\\", "/")
            path = (ROOT / relative).resolve()
            if not relative or ROOT.resolve() not in path.parents or not path.is_file():
                blockers.append(f"source_file_missing:{relative}")
                continue
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != str(row.get("sha256", "")):
                blockers.append(f"source_file_hash:{relative}")
            combined.update(
                relative.encode() + b"\0" + actual_hash.encode() + b"\n"
            )
        if combined.hexdigest() != str(
            source_freeze.get("working_tree_source_hash", "")
        ):
            blockers.append("source_tree_hash_mismatch")
    dirty = tracked_source_dirty()
    if dirty:
        blockers.append("tracked_source_dirty")
    return blockers


def validate_registry_contract(registry: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if registry.get("registry_content_hash") != canonical_registry_hash(registry):
        blockers.append("registry_content_hash")
    if registry.get("registry_version") not in {
        "v16_generic_m20_fixed_parent_registry_v1",
        "v16_m20_m2e_fixed_parent_registry_v1",
    }:
        blockers.append("registry_version")
    variants = tuple(registry.get("variants", ()))
    if variants not in {(G0, M20), (M20, M2E)}:
        blockers.append("variant_inventory")
    if registry.get("model") != "qwen3-14b" or registry.get("thinking") is not False:
        blockers.append("model_or_thinking")
    if int(registry.get("case_count", -1)) != 8 or len(
        registry.get("cases", [])
    ) != 8:
        blockers.append("case_count")
    if int(registry.get("cell_count", -1)) != 16:
        blockers.append("cell_count")
    if int(registry.get("candidate_count_per_cell", -1)) != 2 or int(
        registry.get("maximum_planned_candidates", -1)
    ) != 32:
        blockers.append("candidate_budget")
    for flag in (
        "commit_enabled", "parent_mutation_enabled",
        "optimizer_state_update_enabled", "validation_enabled",
        "final_test_enabled",
    ):
        if registry.get(flag) is not False:
            blockers.append(f"isolation_flag:{flag}")
    for index, case in enumerate(registry.get("cases", [])):
        expected_order = list(variants) if index % 2 == 0 else list(reversed(variants))
        if list(case.get("cell_order", ())) != expected_order:
            blockers.append(f"cell_order:{case.get('case_id')}")
    return blockers


def _cost(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in calls),
        "completion_tokens": sum(
            int(row.get("completion_tokens", 0)) for row in calls
        ),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in calls),
        "teacher_calls": sum(row.get("role") == "teacher" for row in calls),
        "critic_calls": sum(row.get("role") == "critic" for row in calls),
        "student_calls": sum(row.get("role") == "student" for row in calls),
        "solver_calls": sum(row.get("role") == "solver" for row in calls),
        "failed_calls": sum(not bool(row.get("success")) for row in calls),
    }


def _context_diagnostics(
    system: Any, variant: str, frozen_hashes: set[str]
) -> dict[str, Any]:
    row = dict(system.tcs_context_history[-1])
    selected = row.get("selected_context_pattern_question_hashes", {})
    item_hashes = sorted({
        str(question_hash)
        for hashes in selected.values()
        for question_hash in hashes
    })
    recursive_paths = [
        str(value).lower()
        for value in row.get("serialized_recursive_field_paths", [])
    ]
    # Generic evidence is selected from target errors across the same fixed probe.
    # Coincidental hash overlap with the private M20 portfolio is retained only as
    # a diagnostic; the causal leakage guard is the empty assigned-hash argument.
    overlap = sorted(set(item_hashes) & set(frozen_hashes))
    responsibility_markers = (
        "responsibility", "assigned", "repair_lane",
        "active_residual_count", "dominant_target_role",
        "dominant_pattern_case_count", "pattern_summary", "repair_cases",
        "preservation_case",
    )
    coverage_conversion_markers = (
        "coverage_residual", "conversion_residual", "repair_lane",
        "coverage_failure", "conversion_failure",
    )
    repair_distance_markers = (
        "repair_distance", "gold_vote_count", "plurality_margin",
        "vote_flip_gain", "margin_gain",
    )
    branch = dict(system.residual_diagnosis_branch_diagnostics[-1])
    assigned_to_generator = list(
        branch.get("responsibility_question_hashes", [])
    )
    return {
        "context_type": row["context_type"],
        "context_mode": row["context_mode"],
        "proposal_context_hash": row["proposal_context_hash"],
        "context_char_count": int(row["context_characters"]),
        "context_item_count": int(row["selected_case_count"]),
        "generator_evidence_item_count": len(item_hashes),
        "frozen_responsibility_hash_overlap_count": len(overlap),
        "responsibility_question_hashes_exposed_to_generator": len(
            assigned_to_generator
        ),
        "generator_assigned_responsibility_hash_count": len(
            assigned_to_generator
        ),
        "member_specific_responsibility_summary_exposed": (
            any(
                marker in path
                for marker in responsibility_markers
                for path in recursive_paths
            )
        ),
        "coverage_conversion_responsibility_labels_exposed": any(
            marker in path
            for marker in coverage_conversion_markers
            for path in recursive_paths
        ),
        "repair_distance_responsibility_metadata_exposed": any(
            marker in path
            for marker in repair_distance_markers
            for path in recursive_paths
        ),
        "forbidden_field_violations": list(
            row.get("forbidden_field_violations", [])
        ),
    }


def _responsibility_effects(
    evaluator: Any,
    *,
    target: int,
    frozen_hashes: set[str],
    candidate_profile: Any,
) -> dict[str, Any]:
    if evaluator.fixed_probe is None:
        raise RuntimeError("fixed probe is not initialized")
    states, _, _ = evaluator.current_states_and_opportunities()
    state_by_hash = {row.question_hash: row for row in states}
    profile_by_hash = {
        example.question_hash: answer
        for example, answer in zip(
            evaluator.fixed_probe.examples, candidate_profile, strict=True
        )
    }
    gain = loss = coverage_gain = conversion_gain = 0
    nonresponsibility_gain = nonresponsibility_loss = 0
    stable_loss = pivotal_loss = unique_loss = fragile_loss = 0
    stable = evaluator.stable_correct_question_hashes_by_agent[target]
    for question_hash in sorted(state_by_hash):
        state = state_by_hash[question_hash]
        candidate = profile_by_hash[question_hash]
        before_correct = bool(state.team_correctness[target])
        after_correct = bool(
            candidate.valid
            and evaluator.match_answer(candidate.answer, state.gold_answer)
        )
        gained = not before_correct and after_correct
        lost = before_correct and not after_correct
        if question_hash in frozen_hashes:
            gain += int(gained)
            loss += int(lost)
            coverage_gain += int(gained and state.gold_vote_count == 0)
            conversion_gain += int(gained and state.gold_vote_count > 0)
        else:
            nonresponsibility_gain += int(gained)
            nonresponsibility_loss += int(lost)
            if lost:
                peer = build_peer_vote_context(state, target)
                if state.gold_vote_count == 1:
                    unique_loss += 1
                elif state.vote_correct and peer.peer_margin <= 0:
                    pivotal_loss += 1
                elif question_hash in stable:
                    stable_loss += 1
                else:
                    fragile_loss += 1
    return {
        "responsibility_residual_gain_count": gain,
        "responsibility_residual_loss_count": loss,
        "responsibility_repair_rate": gain / max(1, len(frozen_hashes)),
        "coverage_responsibility_gain": coverage_gain,
        "conversion_responsibility_gain": conversion_gain,
        "responsibility_portfolio_size": len(frozen_hashes),
        "nonresponsibility_gain_count": nonresponsibility_gain,
        "nonresponsibility_loss_count": nonresponsibility_loss,
        "stable_loss_count": stable_loss,
        "pivotal_loss_count": pivotal_loss,
        "unique_loss_count": unique_loss,
        "fragile_loss_count": fragile_loss,
    }


def _candidate_payload(
    row: Any,
    *,
    responsibility_effects: dict[str, Any],
) -> dict[str, Any]:
    evaluation = row.final_evaluation
    if evaluation is None or row.constraint is None:
        raise ValueError("probe candidate lacks final common-safe evaluation")
    diagnostics = dict(row.module2_diagnostics)
    student = row.student_candidate
    trigger = str(getattr(student, "trigger_condition", ""))
    behavior = str(getattr(student, "localized_behavior", ""))
    parent_prompt = evaluation.prompt[: -len(
        "\n\n[Responsibility-specific conditional refinement]\n"
        + f"When {trigger}:\n    {behavior}\n\n"
        + "Outside this condition, follow the original procedure unchanged."
    )] if trigger and behavior else ""
    return {
        "prompt_hash": row.prompt_hash,
        "evaluation": asdict(evaluation),
        "constraint": asdict(row.constraint),
        "candidate_geometry": diagnostics.get("candidate_geometry"),
        "target_gain": diagnostics.get("target_gain"),
        "vote_gain_count": diagnostics.get("vote_gain_count"),
        "vote_loss_count": diagnostics.get("vote_loss_count"),
        "vote_net_gain": diagnostics.get("vote_net_gain"),
        **responsibility_effects,
        "scoped_patch_mechanism": {
            "enabled": bool(trigger and behavior),
            "parent_prefix_byte_identical": bool(
                trigger and evaluation.prompt.startswith(parent_prompt)
                and evaluation.prompt[: len(parent_prompt)] == parent_prompt
            ),
            "parent_prompt_sha256": hashlib.sha256(parent_prompt.encode()).hexdigest(),
            "trigger_condition_sha256": hashlib.sha256(trigger.encode()).hexdigest() if trigger else "",
            "localized_behavior_sha256": hashlib.sha256(behavior.encode()).hexdigest() if behavior else "",
            "trigger_character_count": len(trigger),
            "localized_behavior_character_count": len(behavior),
            "unconditional_marker_count": sum(
                token in trigger.lower()
                for token in ("always", "every problem", "every question", "before every answer")
            ),
        },
    }


async def run_cell(
    registry: dict[str, Any],
    case: dict[str, Any],
    variant: str,
    out_dir: Path,
    cache_path: Path,
    freeze_latch: FirstSuccessfulCallFreeze | None = None,
) -> dict[str, Any]:
    require_fresh_cell_path(out_dir)
    frozen = set(map(str, case["assigned_question_hashes"]))
    if responsibility_evidence_hash(frozen) != str(
        case["frozen_responsibility_evidence_hash"]
    ):
        raise RuntimeError("frozen responsibility evidence hash mismatch")
    generator = generation_system(
        case,
        variant,
        out_dir=out_dir / "generation",
        cache_path=cache_path,
    )
    evaluator = evaluation_system(
        case,
        out_dir=out_dir / "evaluation",
        cache_path=cache_path,
    )
    if freeze_latch is not None:
        freeze_latch.attach(generator)
        freeze_latch.attach(evaluator)
    target = int(case["target_agent_id"])
    if team_prompt_hash(generator) != case["parent_team_hash"]:
        raise RuntimeError("generation parent team hash mismatch")
    if team_prompt_hash(evaluator) != case["parent_team_hash"]:
        raise RuntimeError("evaluation parent team hash mismatch")
    if (
        [agent.current_prompt for agent in generator.agents]
        != [agent.current_prompt for agent in evaluator.agents]
        or generator.active_profiles != evaluator.active_profiles
        or generator.initial_profiles != evaluator.initial_profiles
        or generator.team_state_version != evaluator.team_state_version
        or generator.accepted_state_count != evaluator.accepted_state_count
    ):
        raise RuntimeError("generation and evaluation parent prompts/profiles differ")
    before_generator = state_hash(generator)
    before_evaluator = state_hash(evaluator)
    before_prompts = [agent.current_prompt for agent in evaluator.agents]
    allowed_generator_logs_before = diagnostic_payload(generator)
    allowed_evaluator_logs_before = diagnostic_payload(evaluator)
    atomic_write_json(out_dir / "cell_status.json", {
        "status": "started",
        "case_id": case["case_id"],
        "variant": variant,
        "commit_enabled": False,
        "validation_enabled": False,
        "final_test_enabled": False,
    })
    funnel = CandidateFunnel()
    try:
        candidates = await generator.propose_candidates(
            target,
            generation_hashes(variant, frozen),
            funnel,
            int(case["source_update_index"]),
        )
        context_diag = _context_diagnostics(generator, variant, frozen)
        if variant == G0:
            if context_diag["context_type"] != AccuracyDiagnosisContext.__name__:
                raise RuntimeError("G0 did not use current generic accuracy context")
            if context_diag[
                "responsibility_question_hashes_exposed_to_generator"
            ]:
                raise RuntimeError("G0 responsibility evidence leaked to generator")
        elif context_diag["context_type"] != SingleLaneDiagnosisContext.__name__:
            raise RuntimeError("conditioned variant did not use current single-lane context")

        winner = incumbent = None
        evaluated = []
        if candidates:
            winner, incumbent, evaluated = await evaluator.evaluate_candidates(
                target,
                candidates,
                frozen,
                funnel,
                int(case["source_update_index"]),
            )
        terminal = str(funnel.terminal_failure_class or "")
        if terminal in {"transport_failure", "persistence_failure"}:
            raise RuntimeError(f"probe infrastructure failure: {terminal}")
        after_generator = state_hash(generator)
        after_evaluator = state_hash(evaluator)
        if before_generator != after_generator or before_evaluator != after_evaluator:
            raise RuntimeError("fixed-parent probe mutated parent state")
        if before_prompts != [agent.current_prompt for agent in evaluator.agents]:
            raise RuntimeError("fixed-parent probe committed a prompt")
        allowed_generator_logs_after = diagnostic_payload(generator)
        allowed_evaluator_logs_after = diagnostic_payload(evaluator)
        calls = [*generator.llm.calls, *evaluator.llm.calls]
        payload = {
            "result_version": RESULT_VERSION,
            "case_id": case["case_id"],
            "seed": int(case["source_seed"]),
            "update_index": int(case["source_update_index"]),
            "variant": variant,
            "target_agent_id": target,
            "parent_team_hash": case["parent_team_hash"],
            "responsibility_evidence_hash": case[
                "frozen_responsibility_evidence_hash"
            ],
            "generation_context": context_diag,
            "generation_policy": generator.protocol.tcs_context_policy,
            "evaluation_policy": {
                "experiment_setting": evaluator.protocol.name,
                "sample_pool_policy": evaluator.protocol.sample_pool_policy,
                "stage_a_policy": evaluator.protocol.stage_a_policy,
                "candidate_acceptance_policy": (
                    evaluator.protocol.candidate_acceptance_policy
                ),
                "candidate_ranking_policy": (
                    evaluator.protocol.candidate_ranking_policy
                ),
            },
            "requested_candidate_count": int(
                generator.protocol.candidates_per_target_branch
            ),
            "generated_candidate_count": len(candidates),
            "evaluated_candidate_count": len(evaluated),
            "funnel": asdict(funnel),
            "incumbent": asdict(incumbent) if incumbent else None,
            "candidates": [
                _candidate_payload(
                    row,
                    responsibility_effects=_responsibility_effects(
                        evaluator,
                        target=target,
                        frozen_hashes=frozen,
                        candidate_profile=row.profile,
                    ),
                )
                for row in evaluated
            ],
            "winner_prompt_hash_diagnostic_only": (
                winner.prompt_hash if winner else ""
            ),
            "cost": _cost(calls),
            "team_prompt_commit_count": 0,
            "optimizer_state_update_count": 0,
            "parent_state_mutation_count": 0,
            "commit_performed": False,
            "parent_state_hash_before": before_evaluator,
            "parent_state_hash_after": after_evaluator,
            "generation_parent_state_hash_before": before_generator,
            "generation_parent_state_hash_after": after_generator,
            "allowed_generation_diagnostic_lengths_before": (
                allowed_generator_logs_before
            ),
            "allowed_generation_diagnostic_lengths_after": (
                allowed_generator_logs_after
            ),
            "allowed_evaluation_diagnostic_lengths_before": (
                allowed_evaluator_logs_before
            ),
            "allowed_evaluation_diagnostic_lengths_after": (
                allowed_evaluator_logs_after
            ),
            "validation_calls": (
                generator.validation_evaluation_count
                + evaluator.validation_evaluation_count
            ),
            "test_calls": (
                generator.test_evaluation_count
                + evaluator.test_evaluation_count
            ),
            "execution_commit": registry["execution_commit"],
            "registry_content_hash": registry["registry_content_hash"],
        }
        atomic_write_json(out_dir / "cell_result.json", payload)
        atomic_write_json(out_dir / "cell_status.json", {
            "status": "complete",
            "case_id": case["case_id"],
            "variant": variant,
            "cell_result_present": True,
        })
        return payload
    except Exception as exc:
        atomic_write_json(out_dir / "cell_status.json", {
            "status": "failed",
            "case_id": case["case_id"],
            "variant": variant,
            "failure_class": type(exc).__name__,
            "cell_result_present": (out_dir / "cell_result.json").exists(),
        })
        raise
    finally:
        atomic_write_jsonl(
            out_dir / "llm_calls.jsonl",
            [*generator.llm.calls, *evaluator.llm.calls],
        )


async def main_async(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    authorization_env = (
        "M2E_FIXED_PARENT_PROBE_AUTHORIZED"
        if tuple(registry.get("variants", ())) == (M20, M2E)
        else AUTHORIZATION_ENV
    )
    if os.environ.get(authorization_env) != "1":
        raise SystemExit(
            f"API execution blocked: {authorization_env}=1 required"
        )
    source_freeze = json.loads(args.source_freeze.read_text(encoding="utf-8"))
    registry_file_hash = hashlib.sha256(args.registry.read_bytes()).hexdigest()
    if source_freeze.get("registry_file_sha256") != registry_file_hash:
        raise SystemExit("source freeze gate failed: registry_file_sha256")
    blockers = [
        *validate_registry_contract(registry),
        *source_freeze_gate(
            registry,
            source_freeze,
            registry_path=args.registry,
            definition_root=args.source_freeze.parent,
        ),
    ]
    if blockers:
        raise SystemExit("source freeze gate failed: " + ",".join(blockers))
    out_root = args.out_root.resolve()
    if not project_local(out_root) or out_root.exists():
        raise SystemExit("out_root must be a fresh project-local directory")

    if tuple(registry.get("variants", ())) == (M20, M2E):
        from preflight_v16_m20_m2e_probe import preflight
    else:
        from preflight_v16_generic_m20_probe import preflight

    preflight_result = preflight(
        registry,
        scratch=ROOT / "runs/v16_generic_m20_runtime_preflight_only",
    )
    if preflight_result.get("status") != "PASS":
        raise SystemExit(
            "offline semantic preflight failed: "
            + ",".join(preflight_result.get("errors", []))
        )
    out_root.mkdir(parents=True)
    cache_path = out_root / "_shared_solver_cache.sqlite"
    freeze_latch = FirstSuccessfulCallFreeze(
        registry,
        source_freeze,
        out_root / "first_success_source_freeze.json",
        registry_path=args.registry,
        definition_root=args.source_freeze.parent,
    )
    results = []
    for case in registry["cases"]:
        for variant in case["cell_order"]:
            if freeze_latch.started:
                blockers = source_freeze_gate(
                    registry,
                    source_freeze,
                    registry_path=args.registry,
                    definition_root=args.source_freeze.parent,
                )
                if blockers:
                    raise RuntimeError(
                        "tracked source changed after provider call: "
                        + ",".join(blockers)
                    )
            result = await run_cell(
                registry,
                case,
                variant,
                out_root / case["case_id"] / variant,
                cache_path,
                freeze_latch,
            )
            results.append(result)
            print(json.dumps({
                "status": "cell_complete",
                "completed_cells": len(results),
                "case_id": result["case_id"],
                "variant": result["variant"],
                "generated_candidates": result["generated_candidate_count"],
                "evaluated_candidates": result["evaluated_candidate_count"],
            }, sort_keys=True), flush=True)
    final_blockers = source_freeze_gate(
        registry,
        source_freeze,
        registry_path=args.registry,
        definition_root=args.source_freeze.parent,
    )
    if final_blockers:
        raise RuntimeError(
            "tracked source changed before final persistence: "
            + ",".join(final_blockers)
        )
    summary = {
        "probe_version": PROBE_VERSION,
        "execution_commit": registry["execution_commit"],
        "registry_hash": registry["registry_content_hash"],
        "cell_count": len(results),
        "requested_candidate_count": sum(
            row["requested_candidate_count"] for row in results
        ),
        "commit_count": 0,
        "parent_state_mutation_count": 0,
        "optimizer_state_update_count": 0,
        "validation_calls": sum(row["validation_calls"] for row in results),
        "test_calls": sum(row["test_calls"] for row in results),
        "tracked_source_freeze_hard": freeze_latch.started,
        "first_success_source_freeze": (
            json.loads(freeze_latch.marker_path.read_text(encoding="utf-8"))
            if freeze_latch.marker_path.is_file() else None
        ),
        "cells": results,
    }
    atomic_write_json(out_root / "probe_summary.json", summary)
    print(json.dumps({
        "status": "probe_complete",
        "cell_count": summary["cell_count"],
        "requested_candidate_count": summary["requested_candidate_count"],
        "validation_calls": summary["validation_calls"],
        "test_calls": summary["test_calls"],
    }, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
