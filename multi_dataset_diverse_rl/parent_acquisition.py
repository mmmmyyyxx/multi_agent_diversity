"""Optimize-only parent reconstruction; no provider, optimizer, or write-back."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from types import SimpleNamespace
from typing import Any

from infrastructure.common_solver_contract_v1.contract import COMMON_SOLVER_CONTRACT_ID, parse_solver_output, request_identity
from .evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION
from .local_optimizers.gepa_adapter import validate_complete_compact_prompt
from .local_optimizers.schemas import LocalEvidenceExample, LocalOptimizationTask, LocalOptimizerBudget
from .peer_state import build_peer_vote_context, build_team_vote_state
from .responsibility import ResponsibilityState, build_service_routing, compute_member_aware_repair_opportunity, compute_repair_eligibility_sets
from .team_search.primary_responsibility_scheduler import build_primary_responsibility_summaries
from .team_search.schemas import TeamSearchRequest
from .team_search.system_runtime import FrozenResponsibilitySnapshot, SystemResponsibilityAssignmentFactory
from .team_search.task_builder import LocalTaskBuilder


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def restore_task(payload: dict) -> LocalOptimizationTask:
    values = dict(payload)
    for key in ("search_examples", "local_validation_examples"):
        values[key] = tuple(LocalEvidenceExample(**{**row, "tags": tuple(row["tags"])}) for row in values[key])
    values["budget"] = LocalOptimizerBudget(**values["budget"])
    return LocalOptimizationTask(**values)


def select_parents(catalog: list[dict], count: int = 4) -> list[str]:
    """Exact maximum coverage; stable subset hash breaks all equal-coverage ties."""
    eligible = [row for row in catalog if row["eligible"]]
    if len(eligible) < count:
        return []
    def key(group):
        ids = sorted(row["parent_task_id"] for row in group)
        return (-len({row["source_state_hash"] for row in group}),
                -len({row["target_member"] for row in group}),
                -len({row["primary_responsibility_lane"] for row in group}), digest(ids), tuple(ids))
    chosen = min(combinations(eligible, count), key=key)
    return sorted(row["parent_task_id"] for row in chosen)


def task_integrity(task: LocalOptimizationTask, examples: list[dict], target: int, lane: str) -> dict:
    source = {row["example_id"]: row for row in examples}
    validate_complete_compact_prompt(task.parent_prompt, parent_prompt=task.parent_prompt,
                                    examples=(*task.search_examples, *task.local_validation_examples), max_chars=3000)
    checks = {
        "target_member_explicit": target in range(5),
        "primary_lane_explicit": lane in {"coverage", "direct_flip", "near_margin"},
        "search_nonempty": bool(task.search_examples),
        "local_optimizer_validation_size_12": len(task.local_validation_examples) == 12,
        "optimize_only_payloads": all(row.example_id in source and row.input_payload == source[row.example_id]["question"]
                                     and row.gold == source[row.example_id]["gold"]
                                     for row in (*task.search_examples, *task.local_validation_examples)),
        "ordered_ids_unique": all(len(rows) == len({r.example_id for r in rows})
                                  for rows in (task.search_examples, task.local_validation_examples)),
        "validation_subset_search": {r.example_id for r in task.local_validation_examples} <= {r.example_id for r in task.search_examples},
        "local_validation_4_4_4": all(sum(group in row.tags for row in task.local_validation_examples) == 4
                                        for group in ("responsibility", "coalition", "preservation")),
        "context_reconstructable": task.optimization_context.startswith(f"primary_responsibility_lane={lane}\n"),
        "roundtrip_exact": asdict(restore_task(json.loads(json.dumps(asdict(task))))) == asdict(task),
        "current_solver_contract": task.solver_contract_id == COMMON_SOLVER_CONTRACT_ID,
        "current_output_contract": task.output_contract_id == SOLVER_OUTPUT_CONTRACT_VERSION,
        "backend_state_absent": task.backend_state is None,
    }
    if not all(checks.values()):
        raise ValueError("parent_task_integrity_failed:" + ",".join(k for k, v in checks.items() if not v))
    return checks


def build_catalog(inputs: dict, observations: list[dict], *, seed: int, local_metric_budget: int) -> dict:
    """Use baseline realizations only, with the unchanged current domain modules."""
    examples = inputs["examples"]
    prompt = inputs["parent_prompt"]
    if len(examples) != 100 or len(observations) != 100 or len({r["example_id"] for r in examples}) != 100:
        raise ValueError("Optimize100_complete_baseline_required")
    states, profile = {}, []
    for example, observation in zip(examples, observations, strict=True):
        if observation["example_id"] != example["example_id"]:
            raise ValueError("baseline_order_mismatch")
        if example["example_id"] != text_hash(example["question"]):
            raise ValueError("example_payload_identity_mismatch")
        if observation["request_identity"] != request_identity(decision_procedure=prompt, question=example["question"]):
            raise ValueError("baseline_request_identity_mismatch")
        if observation["response_sha256"] != text_hash(observation["raw_output"]):
            raise ValueError("baseline_response_identity_mismatch")
        parsed = parse_solver_output(observation["raw_output"], question=example["question"])
        answer = parsed.answer if parsed.valid else ""
        profile.append(SimpleNamespace(answer=answer, valid=parsed.valid))
        states[example["example_id"]] = build_team_vote_state(
            question_hash=example["example_id"], gold_answer=example["gold"],
            answers=[answer] * 5, valid_vector=[parsed.valid] * 5, tie_break="abstain", seed=seed)
    opportunities = {ident: tuple(compute_member_aware_repair_opportunity(
        team_state=state, peer_context=build_peer_vote_context(state, member), tau=1.0)
        for member in range(5)) for ident, state in states.items()}
    responsibility = ResponsibilityState(updates_since_selected_by_agent={i: 0 for i in range(5)},
                                       accepted_updates_by_agent={i: 0 for i in range(5)},
                                       target_attempt_count_by_agent={i: 0 for i in range(5)})
    eligible, _, _ = compute_repair_eligibility_sets(team_states=states, opportunities=opportunities, state=responsibility)
    routing = build_service_routing(team_states=states, opportunities=opportunities,
                                   eligible_agents_by_question=eligible, state=responsibility, seed=seed)
    margins = {ident: state.plurality_margin for ident, state in states.items()}
    snapshot = FrozenResponsibilitySnapshot(assigned=routing.active_slices, state_by_question=states,
                                            current_margin_by_question=margins)
    summaries = build_primary_responsibility_summaries(assigned=snapshot.assigned,
                  current_margin_by_question=margins, failure_count_by_member={i: 0 for i in range(5)})
    source_state = digest({"seed": seed, "prompt_hash": text_hash(prompt),
                           "baseline": [{k: row[k] for k in ("example_id", "request_identity", "response_sha256")} for row in observations]})
    # Only the data attributes consumed by the production assignment factory.
    # No system constructor, Solver, controller, or evaluation stage is invoked.
    facade = SimpleNamespace(fixed_probe=SimpleNamespace(examples=[SimpleNamespace(
        question_hash=row["example_id"], question=row["question"], gold_answer=row["gold"]) for row in examples]),
        active_profiles=[tuple(profile)] * 5, agents=[SimpleNamespace(current_prompt=prompt) for _ in range(5)])
    builder = LocalTaskBuilder()
    factory = SystemResponsibilityAssignmentFactory(system=facade, snapshot_reader=lambda: snapshot, task_builder=builder)
    request = TeamSearchRequest(seed=seed, update_index=0, team_state_hash=source_state,
                               local_metric_budget=local_metric_budget, solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
                               output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION)
    catalog, private, audits = [], {}, []
    seen_content = set()
    for summary in summaries:
        member, lane = summary.member_id, summary.primary_lane
        identity = f"seed{seed}_update0_member{member}"
        row = {"parent_task_id": identity, "source_state_hash": source_state, "source_seed": seed,
               "update_index": 0, "target_member": member, "primary_responsibility_lane": lane,
               "parent_prompt_hash": text_hash(prompt), "eligible": False}
        if not snapshot.assigned[member] or lane == "fallback":
            row["exclusion_reason"] = "no_primary_responsibility"
        else:
            try:
                assignment = factory.build_from_member(request=request, member_id=member, primary_lane=lane,
                                                       responsibility_identity="phase_a_frozen_baseline")
            except ValueError as exc:
                if "requires exactly" not in str(exc):
                    raise
                row["exclusion_reason"] = "current_local_validation_quota_unavailable"
            else:
                task = builder.build(request, assignment)
                audit = task_integrity(task, examples, member, lane)
                payload = json.loads(json.dumps(asdict(task)))
                content_hash = digest({k: payload[k] for k in ("parent_prompt", "search_examples", "local_validation_examples", "optimization_context")})
                if content_hash in seen_content:
                    row["exclusion_reason"] = "duplicate_local_problem"
                else:
                    seen_content.add(content_hash)
                    row.update(eligible=True, task_payload_sha256=digest(payload), local_problem_sha256=content_hash,
                               search_example_ids=[r.example_id for r in task.search_examples],
                               local_validation_example_ids=[r.example_id for r in task.local_validation_examples],
                               search_payload_sha256=digest(payload["search_examples"]),
                               local_validation_payload_sha256=digest(payload["local_validation_examples"]),
                               optimization_context_sha256=text_hash(task.optimization_context),
                               search_example_identity_hash=digest([r.example_id for r in task.search_examples]),
                               local_validation_identity_hash=digest([r.example_id for r in task.local_validation_examples]),
                               task_seed=task.seed)
                    private[identity] = payload
                    audits.append({"parent_task_id": identity, "checks": audit})
        catalog.append(row)
    selected = select_parents(catalog)
    return {"parent_catalog": catalog, "eligible_parent_catalog": [r for r in catalog if r["eligible"]],
            "selected_parent_ids": selected, "tasks_private": private, "integrity_audit": audits,
            "status": "PARENTS_FROZEN_PHASE_B_NOT_AUTHORIZED" if len(selected) == 4 else "PARENT_CATALOG_INSUFFICIENT",
            "source_state_hash": source_state}
