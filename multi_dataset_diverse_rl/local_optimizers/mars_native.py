"""Official-code-topology MARS backend with an Optimize-only Target universe."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

from ..evaluation.mutable_prompt_contract import validate_mutable_decision_procedure
from ..native_feed import (
    Layer2OptimizationRequest,
    NativeOptimizationRequest,
    PacketEvidenceExample,
)
from ..versions import (
    MARS_LAYER2_EVIDENCE_BACKEND_VERSION,
    MARS_LAYER2_RESPONSIBILITY_OVERLAY_VERSION,
    MARS_NATIVE_FEED_VERSION,
)
from .base import LocalSolverEvaluator
from .schemas import (
    LocalEvidenceExample,
    LocalOptimizationResult,
    LocalPromptCandidate,
    OpaqueOptimizerState,
)


@dataclass(frozen=True)
class MARSRoleResponse:
    payload: Mapping[str, Any] | str
    input_tokens: int = 0
    output_tokens: int = 0
    provider_called: bool = True


class MARSRoleClient(Protocol):
    async def complete(self, *, role: str, context: Mapping[str, Any]) -> MARSRoleResponse: ...


class MARSNativeDataBuilder:
    """Own the full Optimize Target dataset used for every prompt evaluation."""

    def __init__(
        self,
        *,
        optimize_examples: Sequence[LocalEvidenceExample],
        optimize_universe_id: str,
    ) -> None:
        self.examples = tuple(optimize_examples)
        self.optimize_universe_id = optimize_universe_id
        if not self.examples:
            raise ValueError("MARS Target Optimize universe cannot be empty")
        ids = [row.example_id for row in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("MARS Target example ids must be unique")

    def build(self, request: NativeOptimizationRequest) -> tuple[LocalEvidenceExample, ...]:
        if request.optimize_universe_id != self.optimize_universe_id:
            raise ValueError("MARS native Optimize-universe identity mismatch")
        return self.examples

    def identity(self) -> str:
        payload = {
            "backend": MARS_NATIVE_FEED_VERSION,
            "universe": self.optimize_universe_id,
            "ordered_example_ids": [row.example_id for row in self.examples],
            "target_evaluation": "full_dataset_every_candidate",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

def _payload(response: MARSRoleResponse) -> Mapping[str, Any]:
    value: Any = response.payload
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("MARS role response must be a JSON object")
    return value


class MARSNativeFeedOptimizer:
    """Preserve Planner→T/C/S→full-Target topology behind the shared boundary."""

    backend_name = "mars_native_feed"
    backend_version = MARS_NATIVE_FEED_VERSION

    def __init__(
        self,
        *,
        evaluator: LocalSolverEvaluator,
        role_client: MARSRoleClient,
        data_builder: MARSNativeDataBuilder,
        task_definition: str,
        layer2_overlay_enabled: bool,
        max_stable_rounds: int = 2,
        stability_threshold: float = 0.01,
    ) -> None:
        self.evaluator = evaluator
        self.role_client = role_client
        self.data_builder = data_builder
        self.task_definition = task_definition
        self.layer2_overlay_enabled = layer2_overlay_enabled
        self.max_stable_rounds = max_stable_rounds
        self.stability_threshold = stability_threshold

    def _overlay(self, request: NativeOptimizationRequest) -> Mapping[str, Any] | None:
        if not self.layer2_overlay_enabled:
            return None
        return {
            "version": MARS_LAYER2_RESPONSIBILITY_OVERLAY_VERSION,
            **request.responsibility.overlay_payload(),
        }

    def _target_evaluate(self, procedure: str, dataset):
        observations = [self.evaluator.evaluate(procedure, row) for row in dataset]
        score = sum(row.valid and row.correct for row in observations) / len(observations)
        per_example = {
            example.example_id: float(observation.valid and observation.correct)
            for example, observation in zip(dataset, observations, strict=True)
        }
        return float(score), per_example, observations

    @staticmethod
    def _student_procedure(payload: Mapping[str, Any]) -> str:
        if set(payload) != {"decision_procedure"}:
            raise ValueError("MARS Student adapter requires one decision_procedure field")
        value = payload["decision_procedure"]
        if not isinstance(value, str):
            raise ValueError("MARS Student decision_procedure must be text")
        procedure = value.strip()
        validate_mutable_decision_procedure(procedure)
        return procedure

    async def optimize_native(
        self, request: NativeOptimizationRequest
    ) -> LocalOptimizationResult:
        if self.evaluator.solver_contract_id != request.solver_contract_id:
            raise ValueError("MARS native Solver contract mismatch")
        if self.evaluator.output_contract_id != request.output_contract_id:
            raise ValueError("MARS native output contract mismatch")
        dataset = self.data_builder.build(request)
        overlay = self._overlay(request)
        roles: list[MARSRoleResponse] = []
        planner = await self.role_client.complete(
            role="planner",
            context={"task_definition": self.task_definition, "responsibility_overlay": overlay},
        )
        roles.append(planner)
        planner_payload = _payload(planner)
        steps = planner_payload.get("steps")
        if not isinstance(steps, list) or not steps or not all(
            isinstance(step, str) and step.strip() for step in steps
        ):
            raise ValueError("MARS Planner must return non-empty steps")

        parent_score, _, parent_observations = self._target_evaluate(
            request.parent_decision_procedure, dataset
        )
        history = [
            {
                "prompt_sha256": hashlib.sha256(
                    request.parent_decision_procedure.encode("utf-8")
                ).hexdigest(),
                "accuracy": parent_score,
            }
        ]
        observations = list(parent_observations)
        candidates: list[LocalPromptCandidate] = []
        current = request.parent_decision_procedure
        previous_score = parent_score
        stable_rounds = 0
        max_rounds = min(request.budget.native_unit_limit, len(steps))
        for step_index, step in enumerate(steps[:max_rounds], start=1):
            if len(roles) + 3 > request.budget.optimizer_call_limit:
                break
            teacher = await self.role_client.complete(
                role="teacher",
                context={
                    "task_definition": self.task_definition,
                    "previous_prompt": current,
                    "planner_step": step,
                    "dialogue_history": list(history),
                    "responsibility_overlay": overlay,
                },
            )
            roles.append(teacher)
            critic = await self.role_client.complete(
                role="critic",
                context={"teacher_question": dict(_payload(teacher))},
            )
            roles.append(critic)
            critic_payload = _payload(critic)
            if critic_payload.get("socratic_valid") is not True:
                continue
            student = await self.role_client.complete(
                role="student",
                context={
                    "task_definition": self.task_definition,
                    "last_prompt": current,
                    "teacher_question": dict(_payload(teacher)),
                    "dialogue_history": list(history),
                    "responsibility_overlay": overlay,
                },
            )
            roles.append(student)
            candidate = self._student_procedure(_payload(student))
            if candidate == current:
                stable_rounds += 1
                continue
            if len(observations) + len(dataset) > request.budget.metric_call_limit:
                break
            score, per_example, candidate_observations = self._target_evaluate(
                candidate, dataset
            )
            observations.extend(candidate_observations)
            digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            history.append({"prompt_sha256": digest, "accuracy": score})
            if score > parent_score:
                candidates.append(
                    LocalPromptCandidate(
                        candidate_id=f"mars-{digest[:16]}",
                        prompt=candidate,
                        local_score=score,
                        per_example_scores=per_example,
                        parent_ids=(history[0]["prompt_sha256"],),
                        generation=step_index,
                        local_rank_metadata={
                            "parent_score": parent_score,
                            "full_target_dataset": True,
                        },
                        backend_metadata={
                            "planner_step": step_index,
                            "official_code_feed": True,
                        },
                    )
                )
            stable_rounds = (
                stable_rounds + 1
                if abs(score - previous_score) < self.stability_threshold
                else 0
            )
            previous_score = score
            current = candidate
            if stable_rounds >= self.max_stable_rounds:
                break

        candidates.sort(key=lambda row: (-float(row.local_score or 0), row.candidate_id))
        candidates = candidates[: request.budget.max_returned_candidates]
        input_tokens = sum(row.input_tokens for row in roles) + sum(
            row.input_tokens for row in observations
        )
        output_tokens = sum(row.output_tokens for row in roles) + sum(
            row.output_tokens for row in observations
        )
        state = OpaqueOptimizerState(
            self.backend_name,
            self.backend_version,
            {
                "feed_identity": self.data_builder.identity(),
                "history": history,
                "planner_steps": len(steps),
                "completed_rounds": len(history) - 1,
                "target_dataset_size": len(dataset),
                "responsibility_overlay_present": overlay is not None,
                "termination_reason": "candidate_returned" if candidates else "no_improving_candidate",
            },
        )
        return LocalOptimizationResult(
            candidates=tuple(candidates),
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            optimizer_state=state,
            solver_calls=sum(row.provider_called for row in observations),
            optimizer_calls=sum(row.provider_called for row in roles),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            termination_reason=state.payload["termination_reason"],
        )

    def parity_identity(self, request: NativeOptimizationRequest) -> str:
        payload = {
            "data_builder": self.data_builder.identity(),
            "task_definition_sha256": hashlib.sha256(
                self.task_definition.encode("utf-8")
            ).hexdigest(),
            "budget": request.budget.__dict__,
            "solver_contract": request.solver_contract_id,
            "output_contract": request.output_contract_id,
            "stopping": [self.max_stable_rounds, self.stability_threshold],
            "candidate_component": "decision_procedure",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _local_eval_row(row: PacketEvidenceExample) -> LocalEvidenceExample:
    return LocalEvidenceExample(
        example_id=row.packet_item_id,
        input_payload=row.input_payload,
        gold=row.gold,
        parent_output=row.parent_output,
        textual_feedback=row.textual_feedback,
        tags=(row.lane, row.responsibility_role),
    )


def _reasoning_evidence(row: PacketEvidenceExample) -> Mapping[str, Any]:
    """Provider-facing reasoning evidence without gold labels or raw reasoning."""

    return {
        "example_id": row.example_id,
        "problem": row.input_payload,
        "parent_output_present": row.parent_output is not None,
        "sanitized_feedback": row.textual_feedback,
        "responsibility_role": row.responsibility_role,
        "lane": row.lane,
        "metadata": dict(row.metadata),
    }


class MARSLayer2EvidenceOptimizer(MARSNativeFeedOptimizer):
    """Released MARS revision topology over an exact Layer-2 evidence packet."""

    backend_name = "mars_search_core_layer2_evidence"
    backend_version = MARS_LAYER2_EVIDENCE_BACKEND_VERSION

    async def optimize_layer2(
        self, request: Layer2OptimizationRequest
    ) -> LocalOptimizationResult:
        if self.evaluator.solver_contract_id != request.solver_contract_id:
            raise ValueError("MARS Layer-2 Solver contract mismatch")
        if self.evaluator.output_contract_id != request.output_contract_id:
            raise ValueError("MARS Layer-2 output contract mismatch")
        packet = request.packet
        before = packet.packet_hash
        dataset = tuple(_local_eval_row(row) for row in packet.local_eval_examples)
        evidence_context = {
            "packet_version": packet.packet_version,
            "packet_hash": packet.packet_hash,
            "target_member": packet.target_member,
            "primary_responsibility_lane": packet.primary_responsibility_lane,
            "responsibility_value": packet.responsibility_value,
            "responsibility_context": packet.responsibility_context,
            "TEAM RESPONSIBILITY EVIDENCE": [
                _reasoning_evidence(row) for row in packet.responsibility_examples
            ],
            "RECENT REGRESSION / FOCUS EVIDENCE": [
                _reasoning_evidence(row) for row in packet.focus_examples
            ],
            "RECENT GAIN / ANCHOR EVIDENCE": [
                _reasoning_evidence(row) for row in packet.anchor_examples
            ],
            "ordered_batch_schedule": [
                list(batch) for batch in packet.ordered_batch_schedule
            ],
        }
        roles: list[MARSRoleResponse] = []
        planner = await self.role_client.complete(
            role="planner",
            context={
                "task_definition": self.task_definition,
                "layer2_evidence_packet": evidence_context,
            },
        )
        roles.append(planner)
        steps = _payload(planner).get("steps")
        if not isinstance(steps, list) or not steps or not all(
            isinstance(step, str) and step.strip() for step in steps
        ):
            raise ValueError("MARS Planner must return non-empty steps")
        parent_score, _, parent_observations = self._target_evaluate(
            request.parent_decision_procedure, dataset
        )
        history = [
            {
                "prompt_sha256": hashlib.sha256(
                    request.parent_decision_procedure.encode("utf-8")
                ).hexdigest(),
                "accuracy": parent_score,
            }
        ]
        observations = list(parent_observations)
        candidates: list[LocalPromptCandidate] = []
        current = request.parent_decision_procedure
        previous_score = parent_score
        stable_rounds = 0
        max_rounds = min(packet.budget.native_unit_limit, len(steps))
        provenance = request.candidate_provenance(
            backend=self.backend_name, backend_version=self.backend_version
        )
        for step_index, step in enumerate(steps[:max_rounds], start=1):
            if len(roles) + 3 > packet.budget.optimizer_call_limit:
                break
            teacher = await self.role_client.complete(
                role="teacher",
                context={
                    "task_definition": self.task_definition,
                    "previous_prompt": current,
                    "planner_step": step,
                    "dialogue_history": list(history),
                    "layer2_evidence_packet": evidence_context,
                },
            )
            roles.append(teacher)
            critic = await self.role_client.complete(
                role="critic",
                context={
                    "teacher_question": dict(_payload(teacher)),
                    "packet_hash": packet.packet_hash,
                    "layer2_evidence_packet": evidence_context,
                },
            )
            roles.append(critic)
            if _payload(critic).get("socratic_valid") is not True:
                continue
            student = await self.role_client.complete(
                role="student",
                context={
                    "task_definition": self.task_definition,
                    "last_prompt": current,
                    "teacher_question": dict(_payload(teacher)),
                    "dialogue_history": list(history),
                    "layer2_evidence_packet": evidence_context,
                },
            )
            roles.append(student)
            candidate = self._student_procedure(_payload(student))
            if candidate == current:
                stable_rounds += 1
                continue
            if len(observations) + len(dataset) > packet.budget.metric_call_limit:
                break
            score, per_example, candidate_observations = self._target_evaluate(
                candidate, dataset
            )
            observations.extend(candidate_observations)
            digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            history.append({"prompt_sha256": digest, "accuracy": score})
            if score > parent_score:
                candidates.append(
                    LocalPromptCandidate(
                        candidate_id=f"mars-layer2-{digest[:16]}",
                        prompt=candidate,
                        local_score=score,
                        per_example_scores=per_example,
                        parent_ids=(history[0]["prompt_sha256"],),
                        generation=step_index,
                        local_rank_metadata={
                            "parent_score": parent_score,
                            "layer2_local_eval": True,
                        },
                        backend_metadata={
                            "planner_step": step_index,
                            "official_code_search_core": True,
                            **provenance,
                        },
                    )
                )
            stable_rounds = (
                stable_rounds + 1
                if abs(score - previous_score) < self.stability_threshold
                else 0
            )
            previous_score = score
            current = candidate
            if stable_rounds >= self.max_stable_rounds:
                break
        if packet.packet_hash != before:
            raise RuntimeError("MARS mutated the immutable Layer-2 packet")
        candidates.sort(key=lambda row: (-float(row.local_score or 0), row.candidate_id))
        candidates = candidates[: packet.budget.max_returned_candidates]
        input_tokens = sum(row.input_tokens for row in roles) + sum(
            row.input_tokens for row in observations
        )
        output_tokens = sum(row.output_tokens for row in roles) + sum(
            row.output_tokens for row in observations
        )
        state = OpaqueOptimizerState(
            self.backend_name,
            self.backend_version,
            {
                "history": history,
                "planner_steps": len(steps),
                "completed_rounds": len(history) - 1,
                "target_dataset_size": len(dataset),
                "responsibility_packet_hash": packet.packet_hash,
                "backend_example_selection_calls": 0,
                "native_global_dataset_accessed": False,
                "official_mars_search_core_retained": {
                    "planner": True,
                    "teacher": True,
                    "critic": True,
                    "student": True,
                },
                "responsibility_count": len(packet.responsibility_examples),
                "focus_count": len(packet.focus_examples),
                "anchor_count": len(packet.anchor_examples),
                "transition_effect_hash": (
                    packet.latest_transition.transition_effect_hash
                    if packet.latest_transition is not None else None
                ),
                "termination_reason": (
                    "candidate_returned" if candidates else "no_improving_candidate"
                ),
            },
        )
        return LocalOptimizationResult(
            candidates=tuple(candidates),
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            optimizer_state=state,
            solver_calls=sum(row.provider_called for row in observations),
            optimizer_calls=sum(row.provider_called for row in roles),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            termination_reason=state.payload["termination_reason"],
        )
