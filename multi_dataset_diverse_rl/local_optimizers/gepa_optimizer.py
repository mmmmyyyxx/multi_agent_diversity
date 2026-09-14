"""Official-GEPA-backed implementation of the local optimizer interface."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .gepa_adapter import GEPAAdapter, LocalSolverEvaluator, validate_complete_compact_prompt
from .gepa_callbacks import GEPALineageCallback
from .gepa_runtime import GEPA_COMMIT, GEPA_SOURCE_SHA256, GEPA_VERSION, import_frozen_gepa
from .schemas import (
    LocalOptimizationResult,
    LocalOptimizationTask,
    LocalPromptCandidate,
    OpaqueOptimizerState,
)
from ..versions import LOCAL_GEPA_RESULT_SEMANTICS_VERSION


class ReflectionLanguageModel(Protocol):
    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        ...


AccountingReader = Callable[[], Mapping[str, int]]


@dataclass(frozen=True)
class GEPAOptimizerConfig:
    candidate_selection_strategy: str = "pareto"
    frontier_type: str = "instance"
    acceptance_criterion: str = "strict_improvement"
    reflection_minibatch_size: int = 3
    use_merge: bool = False
    module_selector: str = "round_robin"
    cache_evaluation: bool = False
    k_local_return: int = 4
    max_prompt_chars: int = 3000
    result_semantics: str = LOCAL_GEPA_RESULT_SEMANTICS_VERSION

    def __post_init__(self) -> None:
        expected = (
            "pareto", "instance", "strict_improvement", 3, False,
            "round_robin", False, 4, LOCAL_GEPA_RESULT_SEMANTICS_VERSION,
        )
        actual = (
            self.candidate_selection_strategy,
            self.frontier_type,
            self.acceptance_criterion,
            self.reflection_minibatch_size,
            self.use_merge,
            self.module_selector,
            self.cache_evaluation,
            self.k_local_return,
            self.result_semantics,
        )
        if actual != expected:
            raise ValueError("two_layer_rg_gepa_v1 local GEPA contract changed")

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
        task_run = self.run_root / task.task_id
        lineage_path = task_run.parent / f"{task.task_id}.lineage.jsonl"
        if task_run.exists() or lineage_path.exists():
            raise FileExistsError("GEPA local search run root must be fresh")
        task_run.parent.mkdir(parents=True, exist_ok=True)
        all_examples = tuple({row.example_id: row for row in (*task.search_examples, *task.local_validation_examples)}.values())
        adapter = GEPAAdapter(
            self.evaluator,
            parent_prompt=task.parent_prompt,
            all_examples=all_examples,
            optimization_context=task.optimization_context,
            output_contract_id=task.output_contract_id,
            max_prompt_chars=self.config.max_prompt_chars,
        )
        callback = GEPALineageCallback(lineage_path)
        before = dict(self.accounting_reader())
        optimize = self._optimize_fn or import_frozen_gepa().optimize
        result = optimize(
            seed_candidate={"system_prompt": task.parent_prompt},
            trainset=list(task.search_examples),
            valset=list(task.local_validation_examples),
            adapter=adapter,
            reflection_lm=self.reflection_lm,
            candidate_selection_strategy=self.config.candidate_selection_strategy,
            frontier_type=self.config.frontier_type,
            reflection_minibatch_size=self.config.reflection_minibatch_size,
            module_selector=self.config.module_selector,
            use_merge=self.config.use_merge,
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
            and result.candidates[index]["system_prompt"] != task.parent_prompt
        ]
        ranked = sorted(
            changed_frontier,
            key=lambda index: (-float(result.val_aggregate_scores[index]), index),
        )
        chosen = ranked[: self.config.k_local_return]
        id_by_index = {
            index: f"gepa:{index}:{hashlib.sha256(result.candidates[index]['system_prompt'].encode('utf-8')).hexdigest()[:12]}"
            for index in range(result.num_candidates)
        }
        generations: dict[int, int] = {}
        candidates: list[LocalPromptCandidate] = []
        validation_ids = [row.example_id for row in task.local_validation_examples]
        for index in chosen:
            prompt = result.candidates[index]["system_prompt"]
            try:
                validate_complete_compact_prompt(
                    prompt,
                    parent_prompt=task.parent_prompt,
                    examples=all_examples,
                    max_chars=self.config.max_prompt_chars,
                )
            except ValueError:
                continue
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
        state_payload = {
            "gepa_result": result.to_dict(),
            "local_gepa_frontier_indices": frontier,
            "changed_frontier_indices": changed_frontier,
            "returned_candidate_indices": chosen,
            "callback_events": callback.events,
            "reflection_minibatches": callback.reflection_minibatches,
            "protocol_hash": self.config.identity(),
            "result_semantics": LOCAL_GEPA_RESULT_SEMANTICS_VERSION,
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
