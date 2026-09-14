"""Official-GEPA-backed implementation of the local optimizer interface."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .gepa_adapter import (
    GEPAAdapter,
    LocalSolverEvaluator,
    validate_complete_compact_prompt,
)
from .gepa_callbacks import GEPALineageCallback
from .gepa_proposer_contract import (
    DECISION_PROCEDURE_REFLECTION_TEMPLATE,
    DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256,
    validate_proposer_contract,
)
from .gepa_runtime import GEPA_COMMIT, GEPA_SOURCE_SHA256, GEPA_VERSION, import_frozen_gepa
from .schemas import (
    LocalOptimizationResult,
    LocalOptimizationTask,
    LocalPromptCandidate,
    OpaqueOptimizerState,
)
from ..versions import (
    LOCAL_GEPA_ENGINE_ACCEPTANCE_SEMANTICS,
    LOCAL_GEPA_CANDIDATE_COMPONENT,
    LOCAL_GEPA_PROPOSER_CONTRACT_VERSION,
    LOCAL_GEPA_REFLECTIVE_DATASET_VERSION,
    LOCAL_GEPA_RESULT_SEMANTICS_VERSION,
    LOCAL_OPTIMIZER_FIDELITY_LEVEL,
    MUTABLE_PROMPT_CONTRACT_VERSION,
)


class ReflectionLanguageModel(Protocol):
    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        ...


AccountingReader = Callable[[], Mapping[str, int]]


@dataclass(frozen=True)
class GEPAOptimizerConfig:
    optimizer_fidelity_level: str = LOCAL_OPTIMIZER_FIDELITY_LEVEL
    candidate_component_name: str = LOCAL_GEPA_CANDIDATE_COMPONENT
    mutable_component_contract_version: str = MUTABLE_PROMPT_CONTRACT_VERSION
    candidate_selection_strategy: str = "pareto"
    frontier_type: str = "instance"
    engine_acceptance_semantics: str = LOCAL_GEPA_ENGINE_ACCEPTANCE_SEMANTICS
    reflection_minibatch_size: int = 3
    skip_perfect_score: bool = True
    perfect_score: float = 1.0
    batch_sampler: str = "epoch_shuffled"
    val_evaluation_policy: str = "full_eval"
    use_merge: bool = False
    max_merge_invocations: int = 5
    merge_val_overlap_floor: int = 5
    module_selector: str = "round_robin"
    cache_evaluation: bool = False
    k_local_return: int = 4
    max_prompt_chars: int = 3000
    proposer_contract_version: str = LOCAL_GEPA_PROPOSER_CONTRACT_VERSION
    reflective_dataset_version: str = LOCAL_GEPA_REFLECTIVE_DATASET_VERSION
    reflection_prompt_template_sha256: str = DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256
    result_semantics: str = LOCAL_GEPA_RESULT_SEMANTICS_VERSION
    max_metric_calls_source: str = "LocalOptimizationTask.budget.max_metric_calls"
    seed_source: str = "LocalOptimizationTask.seed"

    def __post_init__(self) -> None:
        expected = (
            LOCAL_OPTIMIZER_FIDELITY_LEVEL, LOCAL_GEPA_CANDIDATE_COMPONENT,
            MUTABLE_PROMPT_CONTRACT_VERSION,
            "pareto", "instance", LOCAL_GEPA_ENGINE_ACCEPTANCE_SEMANTICS, 3,
            True, 1.0, "epoch_shuffled", "full_eval", False,
            5, 5, "round_robin", False, 4, 3000, LOCAL_GEPA_PROPOSER_CONTRACT_VERSION,
            LOCAL_GEPA_REFLECTIVE_DATASET_VERSION,
            DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256,
            LOCAL_GEPA_RESULT_SEMANTICS_VERSION,
            "LocalOptimizationTask.budget.max_metric_calls", "LocalOptimizationTask.seed",
        )
        actual = (
            self.optimizer_fidelity_level,
            self.candidate_component_name,
            self.mutable_component_contract_version,
            self.candidate_selection_strategy,
            self.frontier_type,
            self.engine_acceptance_semantics,
            self.reflection_minibatch_size,
            self.skip_perfect_score,
            self.perfect_score,
            self.batch_sampler,
            self.val_evaluation_policy,
            self.use_merge,
            self.max_merge_invocations,
            self.merge_val_overlap_floor,
            self.module_selector,
            self.cache_evaluation,
            self.k_local_return,
            self.max_prompt_chars,
            self.proposer_contract_version,
            self.reflective_dataset_version,
            self.reflection_prompt_template_sha256,
            self.result_semantics,
            self.max_metric_calls_source,
            self.seed_source,
        )
        if actual != expected:
            raise ValueError("two_layer_rg_gepa_v1 local GEPA contract changed")
        validate_proposer_contract()

    def identity(self) -> str:
        payload = {
            **asdict(self),
            "gepa_version": GEPA_VERSION,
            "gepa_commit": GEPA_COMMIT,
            "gepa_source_sha256": GEPA_SOURCE_SHA256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class LocalGEPABudgetCapacity:
    metric_budget: int
    validation_size: int
    reflection_minibatch_size: int
    seed_evaluation_calls: int
    proposal_attempt_calls: int
    accepted_full_evaluation_calls: int
    max_rejected_proposals: int
    max_accepted_children: int
    max_accepted_generations: int


def local_gepa_budget_capacity(
    *, metric_budget: int, validation_size: int, reflection_minibatch_size: int
) -> LocalGEPABudgetCapacity:
    """Return the no-overshoot capacity implied by pinned GEPA call arithmetic."""

    if min(metric_budget, validation_size, reflection_minibatch_size) <= 0:
        raise ValueError("GEPA budget arithmetic inputs must be positive")
    seed = validation_size
    proposal = 2 * reflection_minibatch_size
    full = validation_size
    remaining = max(0, metric_budget - seed)
    max_rejected = remaining // proposal
    max_accepted = remaining // (proposal + full)
    return LocalGEPABudgetCapacity(
        metric_budget=metric_budget,
        validation_size=validation_size,
        reflection_minibatch_size=reflection_minibatch_size,
        seed_evaluation_calls=seed,
        proposal_attempt_calls=proposal,
        accepted_full_evaluation_calls=full,
        max_rejected_proposals=max_rejected,
        max_accepted_children=max_accepted,
        max_accepted_generations=max_accepted,
    )


def verify_frozen_gepa_engine_contract() -> None:
    """Assert that the pinned engine exposes and implements the frozen semantics."""

    gepa = import_frozen_gepa()
    required = {
        "skip_perfect_score",
        "perfect_score",
        "batch_sampler",
        "val_evaluation_policy",
        "reflection_prompt_template",
    }
    if not required.issubset(inspect.signature(gepa.optimize).parameters):
        raise RuntimeError("pinned GEPA public API no longer exposes the frozen engine contract")
    from gepa.core.engine import GEPAEngine  # type: ignore[import-not-found]

    source = inspect.getsource(GEPAEngine.run)
    if "if new_sum <= old_sum:" not in source or "on_candidate_rejected" not in source:
        raise RuntimeError("pinned GEPA strict-improvement acceptance semantics changed")


class GEPALocalPromptOptimizer:
    backend_name = "gepa"
    backend_version = GEPA_VERSION
    GEPA_GROUNDED = True

    def __init__(
        self,
        *,
        evaluator: LocalSolverEvaluator,
        reflection_lm: ReflectionLanguageModel,
        accounting_reader: AccountingReader,
        run_root: Path,
        config: GEPAOptimizerConfig | None = None,
        optimize_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.reflection_lm = reflection_lm
        self.accounting_reader = accounting_reader
        self.run_root = Path(run_root)
        self.config = config or GEPAOptimizerConfig()
        self._optimize_fn = optimize_fn

    @staticmethod
    def _generation(index: int, parents: list[list[int | None]], memo: dict[int, int]) -> int:
        if index in memo:
            return memo[index]
        parent_indices = [int(value) for value in parents[index] if value is not None]
        memo[index] = 0 if not parent_indices else 1 + max(
            GEPALocalPromptOptimizer._generation(value, parents, memo) for value in parent_indices
        )
        return memo[index]

    def _run(self, task: LocalOptimizationTask) -> LocalOptimizationResult:
        if task.backend_state is not None:
            raise ValueError("two_layer_rg_gepa_v1 starts fresh GEPA state for every team update")
        if task.budget.reflection_minibatch_size != self.config.reflection_minibatch_size:
            raise ValueError("task/config reflection minibatch mismatch")
        if task.budget.max_returned_candidates != self.config.k_local_return:
            raise ValueError("task/config local return budget mismatch")
        verify_frozen_gepa_engine_contract()
        all_examples = tuple(
            {row.example_id: row for row in (*task.search_examples, *task.local_validation_examples)}.values()
        )
        if any(row.weight != 1.0 for row in all_examples):
            raise ValueError("official local GEPA currently requires unit-weight evidence")
        try:
            validate_complete_compact_prompt(
                task.parent_prompt,
                parent_prompt=task.parent_prompt,
                examples=all_examples,
                max_chars=self.config.max_prompt_chars,
            )
        except ValueError as exc:
            raise ValueError("PARENT_DECISION_PROCEDURE_CONTRACT_VIOLATION") from exc
        task_run = self.run_root / task.task_id
        lineage_path = task_run.parent / f"{task.task_id}.lineage.jsonl"
        if task_run.exists() or lineage_path.exists():
            raise FileExistsError("GEPA local search run root must be fresh")
        task_run.parent.mkdir(parents=True, exist_ok=True)
        adapter = GEPAAdapter(
            self.evaluator,
            parent_prompt=task.parent_prompt,
            all_examples=all_examples,
            optimization_context=task.optimization_context,
            output_contract_id=task.output_contract_id,
            max_prompt_chars=self.config.max_prompt_chars,
        )
        callback = GEPALineageCallback(
            lineage_path,
            parent_prompt=task.parent_prompt,
            examples=all_examples,
            max_prompt_chars=self.config.max_prompt_chars,
        )
        before = dict(self.accounting_reader())
        optimize = self._optimize_fn or import_frozen_gepa().optimize
        result = optimize(
            seed_candidate={self.config.candidate_component_name: task.parent_prompt},
            trainset=list(task.search_examples),
            valset=list(task.local_validation_examples),
            adapter=adapter,
            reflection_lm=self.reflection_lm,
            candidate_selection_strategy=self.config.candidate_selection_strategy,
            frontier_type=self.config.frontier_type,
            reflection_minibatch_size=self.config.reflection_minibatch_size,
            skip_perfect_score=self.config.skip_perfect_score,
            perfect_score=self.config.perfect_score,
            batch_sampler=self.config.batch_sampler,
            val_evaluation_policy=self.config.val_evaluation_policy,
            reflection_prompt_template=DECISION_PROCEDURE_REFLECTION_TEMPLATE,
            module_selector=self.config.module_selector,
            use_merge=self.config.use_merge,
            max_merge_invocations=self.config.max_merge_invocations,
            merge_val_overlap_floor=self.config.merge_val_overlap_floor,
            custom_candidate_proposer=None,
            max_metric_calls=task.budget.max_metric_calls,
            run_dir=str(task_run),
            callbacks=[callback],
            display_progress_bar=False,
            cache_evaluation=self.config.cache_evaluation,
            seed=task.seed,
            raise_on_exception=True,
        )
        after = dict(self.accounting_reader())
        frontier = sorted(
            {
                int(candidate)
                for members in result.per_val_instance_best_candidates.values()
                for candidate in members
            }
        )
        # GEPA index 0 is the seed/root baseline. The Layer-1 boundary returns
        # proposals, not baselines: unchanged prompts have no possible team
        # transition and must not consume TeamMiniBatch evaluation slots.
        changed_frontier = [
            index
            for index in frontier
            if int(index) != 0
            and result.candidates[index][self.config.candidate_component_name] != task.parent_prompt
        ]
        invalid_prompt_indices: list[int] = []
        duplicate_prompt_indices: list[int] = []
        valid_unique: list[int] = []
        seen_prompt_hashes: set[str] = set()
        for index in changed_frontier:
            prompt = result.candidates[index][self.config.candidate_component_name]
            try:
                validate_complete_compact_prompt(
                    prompt,
                    parent_prompt=task.parent_prompt,
                    examples=all_examples,
                    max_chars=self.config.max_prompt_chars,
                )
            except ValueError:
                invalid_prompt_indices.append(index)
                continue
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompt_hashes:
                duplicate_prompt_indices.append(index)
                continue
            seen_prompt_hashes.add(prompt_hash)
            valid_unique.append(index)
        ranked = sorted(
            valid_unique,
            key=lambda index: (-float(result.val_aggregate_scores[index]), index),
        )
        chosen = ranked[: self.config.k_local_return]
        id_by_index = {
            index: f"gepa:{index}:{hashlib.sha256(result.candidates[index][self.config.candidate_component_name].encode('utf-8')).hexdigest()[:12]}"
            for index in range(result.num_candidates)
        }
        generations: dict[int, int] = {}
        candidates: list[LocalPromptCandidate] = []
        validation_ids = [row.example_id for row in task.local_validation_examples]
        for index in chosen:
            prompt = result.candidates[index][self.config.candidate_component_name]
            raw_scores = result.val_subscores[index]
            per_example = {
                validation_ids[int(key)] if isinstance(key, int) and int(key) < len(validation_ids) else str(key): float(value)
                for key, value in raw_scores.items()
            }
            parent_indices = [int(value) for value in result.parents[index] if value is not None]
            candidates.append(
                LocalPromptCandidate(
                    candidate_id=id_by_index[index],
                    prompt=prompt,
                    local_score=float(result.val_aggregate_scores[index]),
                    per_example_scores=per_example,
                    parent_ids=tuple(id_by_index[parent] for parent in parent_indices),
                    generation=self._generation(index, result.parents, generations),
                    local_rank_metadata={
                        "local_gepa_frontier": True,
                        "local_frontier_rank": ranked.index(index) + 1,
                    },
                    backend_metadata={
                        "program_candidate_index": index,
                        "discovery_eval_count": int(result.discovery_eval_counts[index]),
                    },
                )
            )
        input_tokens = int(after.get("input_tokens", 0)) - int(before.get("input_tokens", 0))
        output_tokens = int(after.get("output_tokens", 0)) - int(before.get("output_tokens", 0))
        optimizer_calls = int(after.get("successful_calls", 0)) - int(before.get("successful_calls", 0))
        accepted_events = [
            row for row in callback.events if row["event_type"] == "candidate_accepted"
        ]
        rejected_events = [
            row for row in callback.events if row["event_type"] == "candidate_rejected"
        ]
        skipped_events = [
            row for row in callback.events if row["event_type"] == "evaluation_skipped"
        ]
        full_local_events = [
            row
            for row in callback.events
            if row["event_type"] == "valset_evaluated" and row["candidate_index"] != 0
        ]
        proposal_diagnostics = callback.proposal_diagnostics()
        materialized_candidates = max(0, result.num_candidates - 1)
        solver_reached = len(
            set(proposal_diagnostics["proposal_hashes"])
            & adapter.solver_reached_proposal_hashes
        )
        proposer_diagnostics = {
            **proposal_diagnostics,
            "materialized_candidates": materialized_candidates,
            "unmaterialized_proposals": max(
                0,
                proposal_diagnostics["proposal_attempts"] - materialized_candidates,
            ),
            "accepted_candidates": len(accepted_events),
            "local_frontier_candidates": len(changed_frontier),
            "returned_frontier_candidates": len(chosen),
            "solver_reached": solver_reached,
            "positive_minibatch_delta": sum(
                row["new_score"] > row["old_score"] for row in rejected_events
            ) + len(accepted_events),
            "accepted_mutation": len(accepted_events),
        }
        budget_capacity = asdict(local_gepa_budget_capacity(
            metric_budget=task.budget.max_metric_calls,
            validation_size=len(task.local_validation_examples),
            reflection_minibatch_size=self.config.reflection_minibatch_size,
        ))
        state_payload = {
            "gepa_result": result.to_dict(),
            "local_gepa_frontier_indices": frontier,
            "changed_frontier_indices": changed_frontier,
            "valid_unique_frontier_indices": ranked,
            "invalid_prompt_indices": invalid_prompt_indices,
            "duplicate_prompt_indices": duplicate_prompt_indices,
            "returned_candidate_indices": chosen,
            "callback_events": callback.events,
            "reflection_minibatches": callback.reflection_minibatches,
            "protocol_hash": self.config.identity(),
            "result_semantics": LOCAL_GEPA_RESULT_SEMANTICS_VERSION,
            "optimizer_fidelity_level": self.config.optimizer_fidelity_level,
            "candidate_component_name": self.config.candidate_component_name,
            "budget_capacity": budget_capacity,
            "telemetry": {
                "proposal_attempts": callback.proposal_count,
                "all_scores_perfect_skips": sum(
                    row["reason"] == "all_scores_perfect" for row in skipped_events
                ),
                "positive_minibatch_deltas": sum(
                    row["new_score"] > row["old_score"] for row in rejected_events
                ) + len(accepted_events),
                "accepted_mutations": len(accepted_events),
                "full_local_evaluations": len(full_local_events),
                "valid_changed_unique_frontier": len(ranked),
                "proposer_diagnostics": proposer_diagnostics,
            },
        }
        if candidates:
            termination_reason = "gepa_metric_budget_exhausted"
        elif not changed_frontier:
            termination_reason = "no_local_improvement"
        else:
            termination_reason = "no_valid_local_candidate"
        return LocalOptimizationResult(
            candidates=tuple(candidates),
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            optimizer_state=OpaqueOptimizerState(self.backend_name, self.backend_version, state_payload),
            solver_calls=adapter.solver_calls,
            optimizer_calls=optimizer_calls,
            input_tokens=adapter.input_tokens + input_tokens,
            output_tokens=adapter.output_tokens + output_tokens,
            total_tokens=adapter.input_tokens + adapter.output_tokens + input_tokens + output_tokens,
            termination_reason=termination_reason,
        )

    async def optimize(self, task: LocalOptimizationTask) -> LocalOptimizationResult:
        return await asyncio.to_thread(self._run, task)
