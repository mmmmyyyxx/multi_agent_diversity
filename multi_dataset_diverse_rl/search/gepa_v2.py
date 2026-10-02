"""V2 GEPA public seams: same strict core, wider ephemeral candidate visibility."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

from .. import versions
from ..saturation import SaturationConfig, OptimizationUnitType
from ..local_optimizers.gepa_adapter import GEPAAdapter
from ..local_optimizers.gepa_callbacks import GEPALineageCallback
from ..local_optimizers.gepa_optimizer import GEPALocalPromptOptimizer
from ..local_optimizers.gepa_runtime import import_frozen_gepa
from ..local_optimizers.gepa_native import Layer2FrozenBatchSampler
from ..local_optimizers.schemas import LocalEvidenceExample, LocalOptimizationTask, LocalOptimizerBudget, LocalPromptCandidate, OpaqueOptimizerState
from .context import SearchContextComposer
from .gepa import GEPADerivedConfig
from .schemas import SearchCandidate, SearchResult, SearchContractError
from .variable_evidence import validation_capacity


@dataclass(frozen=True)
class GEPATeamExposureConfig(GEPADerivedConfig):
    optimizer_fidelity_level: str = "GEPA_DERIVED_TEAM_EXPOSURE_V2"
    result_semantics: str = "all_solver_evaluated_unique_proposals_v2"

    def identity(self):
        return hashlib.sha256((versions.UNIFIED_GEPA_EXPOSURE_V2_VERSION + ":" + super().identity()).encode()).hexdigest()


class EphemeralProposalCallback(GEPALineageCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ephemeral_proposals_by_iteration = {}
        self.solver_evaluated_hashes = set()
        self.invalid_response_hashes = set()
        self.metric_count = 0
        self.metric_limit = None
        self._parent_index_by_iteration = {}

    def on_proposal_end(self, event):
        super().on_proposal_end(event)
        instructions = event.get("new_instructions", {})
        if isinstance(instructions, dict) and set(instructions) == {"decision_procedure"} and isinstance(instructions["decision_procedure"], str):
            self._ephemeral_proposals_by_iteration[int(event["iteration"])] = instructions["decision_procedure"]

    def before_evaluation(self, size):
        if self.metric_limit is not None and self.metric_count + size > self.metric_limit:
            raise SearchContractError("GEPA_METRIC_CAPACITY_EXCEEDED_PRE_SOLVER")
        self.metric_count += size

    def observe_evaluation(self, event):
        # A cache realization is also a Solver evaluation. Contract-invalid
        # pseudo-evaluations are excluded by the proposal contract below.
        if event["example_ids"]:
            self.solver_evaluated_hashes.add(event["candidate_hash"])

    def on_evaluation_end(self, event):
        super().on_evaluation_end(event)
        if event.get("candidate_idx") is not None:
            self._parent_index_by_iteration[int(event["iteration"])] = int(event["candidate_idx"])

    def export(self, task, payload):
        outcomes = {r["iteration"]:r for r in self.proposal_diagnostics()["proposal_outcomes"]}
        raw = payload.get("gepa_result", {})
        frontier = set(payload.get("local_gepa_frontier_indices", ()))
        accepted = {r["iteration"]:r["candidate_index"] for r in self.events if r["event_type"] == "candidate_accepted"}
        selected = {}
        proposals = sorted(self._proposal_records, key=lambda r: (r["iteration"] not in accepted, r["iteration"]))
        for row in proposals:
            iteration = row["iteration"]; effect = outcomes[iteration]; h = row["proposal_hash"]
            if not row["changed"] or row["contract_invalid"] or h in selected or h not in self.solver_evaluated_hashes or not effect["evaluable"]:
                continue
            prompt = self._ephemeral_proposals_by_iteration.get(iteration)
            if prompt is None or hashlib.sha256(prompt.encode()).hexdigest() != h:
                raise SearchContractError("proposal ephemeral identity mismatch")
            survived = iteration in accepted
            idx = accepted.get(iteration)
            score = raw["val_aggregate_scores"][idx] if survived else None
            full_delta = score - raw["val_aggregate_scores"][0] if survived else None
            parents = tuple(f"gepa:index:{i}" for i in raw["parents"][idx] if i is not None) if survived else ()
            parent_idx = self._parent_index_by_iteration.get(iteration, 0)
            generation = (GEPALocalPromptOptimizer._generation(idx, raw["parents"], {}) if survived else
                          1 + GEPALocalPromptOptimizer._generation(parent_idx, raw["parents"], {}) if raw.get("parents") else 1)
            if not survived and raw.get("parents"):
                parents = (f"gepa:index:{parent_idx}",)
            selected[h] = LocalPromptCandidate(
                f"gepa:proposal:{iteration}:{h[:12]}", prompt, score, {}, parents, generation,
                backend_metadata=dict(gepa_search_survived=survived,
                    gepa_local_acceptance_status="ACCEPTED" if survived else "REJECTED",
                    proposal_iteration=iteration, proposal_hash=h,
                    local_effect_scope="minibatch", local_acceptance_delta=effect["delta_local"],
                    local_full_validation_delta=full_delta, gepa_frontier_member=idx in frontier if survived else False,
                    local_newly_fixed=effect["newly_fixed"], local_newly_broken=effect["newly_broken"],
                    local_preservation_loss=effect["preservation_loss"], opportunity_id=task.task_id,
                    changed=True, duplicate=False, contract_valid=True, solver_evaluated=True,
                    **({"program_candidate_index":idx} if survived else {})))
        # Each proposal costs at least one pair of minibatches; no score-based truncation.
        ceiling = (task.budget.max_metric_calls - len(task.local_validation_examples)) // (2 * task.budget.reflection_minibatch_size)
        if len(selected) > ceiling:
            raise SearchContractError("proposal count exceeds frozen metric-derived capacity")
        return tuple(selected.values())


class V2ReflectionAdapter(GEPAAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        result = super().evaluate(batch, candidate, capture_traces)
        if self.evaluation_observer is not None and any(not output.get("valid", False) for output in result.outputs):
            self.evaluation_observer.invalid_response_hashes.add(
                hashlib.sha256(candidate["decision_procedure"].encode()).hexdigest())
        return result

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = super().make_reflective_dataset(candidate, eval_batch, components_to_update)
        tags = {t.example.example_id: t.example.tags for t in eval_batch.trajectories}
        for records in rows.values():
            for row in records:
                purpose = tags[row["example_id"]]
                if "safety_boundary_v2" in purpose:
                    row["Reasoning Focus"] = (
                        "BOUNDARY ONLY: observe preservation, collateral risk and applicability. "
                        "Do not repair this row's different mechanism or add another repair objective, "
                        "even when its evaluation outcome is incorrect.")
                elif "focus_repair_v2" in purpose:
                    row["Reasoning Focus"] = "REPAIR: address only the single selected focus mechanism."
        if "\n" not in self.optimization_context:
            return rows
        optional = self.optimization_context.split("\n", 1)[1]
        return {component:[{**r, "Optional Search Context":optional} for r in records]
                for component,records in rows.items()}


class _SilentLogger:
    def log(self, message): pass


class GEPATeamExposureOptimizer(GEPALocalPromptOptimizer):
    """Inherited wrapper and pinned GEPA run, with callback and stopper seams only."""
    def __init__(self, **kwargs):
        self._active_callback = None
        self._delegate_optimize = kwargs.pop("optimize_fn", None)
        kwargs["config"] = kwargs.get("config") or GEPATeamExposureConfig()
        if not isinstance(kwargs["config"], GEPATeamExposureConfig):
            raise SearchContractError("V2 exposure config required")
        kwargs["adapter_factory"] = V2ReflectionAdapter
        kwargs["callback_factory"] = self._callback
        kwargs["optimize_fn"] = self._bounded_optimize
        super().__init__(**kwargs)

    def _callback(self, *args, **kwargs):
        self._active_callback = EphemeralProposalCallback(*args, **kwargs)
        return self._active_callback

    def _bounded_optimize(self, **kwargs):
        cb = self._active_callback
        limit = self._current_metric_budget
        kwargs["max_metric_calls"] = limit
        cb.metric_limit = limit
        n = len(kwargs["valset"]); m = self.config.reflection_minibatch_size
        # Reserve the next complete pair and possible full evaluation before
        # entering an iteration. Backend internals never see a partial budget.
        scientific_stop = kwargs.get("stop_callbacks")
        kwargs["stop_callbacks"] = lambda state: (state.total_num_evals + 2*m + n > limit or
                                                  bool(scientific_stop and scientific_stop(state)))
        # Pinned logger otherwise persists raw proposed text even when rejected.
        # Supported no-persistence/logger seams keep raw proposals ephemeral.
        kwargs["logger"] = _SilentLogger()
        kwargs["run_dir"] = None
        return (self._delegate_optimize or import_frozen_gepa().optimize)(**kwargs)

    def _run(self, task, **kwargs):
        if kwargs.get("saturation_config") is not None:
            raise SearchContractError("V2 metric-budget search cannot use legacy unbounded saturation")
        try:
            before_usage = dict(self.accounting_reader())
            self._current_metric_budget = task.budget.max_metric_calls
            # Reuse the existing complete-evidence-epoch saturation observer;
            # never replace its unit with individual rejected proposals.
            result = super()._run(task, **kwargs,
                saturation_config=SaturationConfig(enabled=True, scientific_budget_enabled=False,
                    local_no_update_patience=3, team_no_update_patience=2),
                saturation_unit_type=OptimizationUnitType.LAYER2_EVIDENCE_EPOCH,
                saturation_mode_name="unified_v2")
            cb = self._active_callback
            after_usage = dict(self.accounting_reader())
            meta_tokens = sum(int(after_usage.get(k, 0)) - int(before_usage.get(k, 0))
                              for k in ("input_tokens", "output_tokens"))
            if not 0 <= meta_tokens <= result.total_tokens:
                raise SearchContractError("V2 search token accounting inconsistency")
            payload = dict(result.optimizer_state.payload)
            candidates = cb.export(task, payload)
            telemetry = dict(proposal_count=cb.proposal_count, team_candidate_count=len(candidates),
                local_survival_update_count=len(cb._accepted_iterations), strict_accepted_count=len(cb._accepted_iterations),
                strict_rejected_exported_count=sum(not c.backend_metadata["gepa_search_survived"] for c in candidates),
                local_metric_evaluations=cb.metric_count, metric_limit=task.budget.max_metric_calls)
            # Persisted state contains hashes/scores/status only, including for accepted proposals.
            state = OpaqueOptimizerState(self.backend_name, versions.UNIFIED_GEPA_EXPOSURE_V2_VERSION,
                {"protocol_hash": self.config.identity(), "callback_events": cb.events,
                 "proposal_diagnostics": cb.proposal_diagnostics(), "telemetry": telemetry,
                 "saturation": payload.get("saturation", {}),
                 "operational_failure":bool(cb.invalid_response_hashes),
                 "token_accounting":{"solver_tokens":result.total_tokens-meta_tokens,
                                     "search_meta_tokens":meta_tokens}})
            return replace(result, candidates=candidates, optimizer_state=state)
        finally:
            if self._active_callback is not None:
                self._active_callback._ephemeral_proposals_by_iteration.clear()
            self._active_callback = None


class V2GEPABridge:
    identity = "unified_gepa_bridge_v2"
    responsibility_context = "Use general reasoning rules; preserve competence and the immutable output interface."

    def __init__(self, *, optimizer, history, seed, solver_contract_id, output_contract_id):
        self.optimizer = optimizer; self.history = history; self.seed = seed
        self.solver_contract_id = solver_contract_id; self.output_contract_id = output_contract_id
        self.raw_candidates = {}

    def make_task(self, opportunity, context):
        budget = int(opportunity.search_budget["metric_calls"])
        m = self.optimizer.config.reflection_minibatch_size
        mutation = opportunity.evidence.mutation_evidence; validation = opportunity.evidence.search_validation_evidence
        if len(mutation) < m or not 1 <= len(validation) <= validation_capacity(budget, m):
            raise SearchContractError("V2 backend evidence capacity mismatch")
        def local(row):
            s = row.signals
            return LocalEvidenceExample(row.example_id, s["input_payload"], s["gold"],
                s.get("target_output"), s.get("feedback"), tuple(s.get("legacy_tags", ())))
        return LocalOptimizationTask("unified_v2_" + hashlib.sha256(opportunity.opportunity_id.encode()).hexdigest(), opportunity.parent_prompt,
            tuple(map(local, mutation)), tuple(map(local, validation)),
            SearchContextComposer().compose(self.responsibility_context, context.pattern_view, context.memory_view),
            self.solver_contract_id, self.output_contract_id, self.seed,
            LocalOptimizerBudget(budget, m, self.optimizer.config.k_local_return),
            run_seed=self.seed, update_index=int(opportunity.opportunity_id.rsplit(":", 2)[1]),
            target_member=opportunity.target_member)

    async def optimize(self, task):
        ids = tuple(r.example_id for r in task.search_examples); m = task.budget.reflection_minibatch_size
        schedule = tuple(tuple(ids[(step*m+j) % len(ids)] for j in range(m))
                         for step in range(task.budget.max_metric_calls // (2*m)))
        sampler = Layer2FrozenBatchSampler(ordered_batch_schedule=schedule, ordered_example_ids=ids,
                                          replay_epochs=True)
        result = await self.optimizer.optimize_with_batch_sampler(task, sampler)
        self.raw_candidates = {r.candidate_id:r for r in result.candidates}
        return result


class GEPATeamCandidateExposureEngine:
    identity = versions.UNIFIED_GEPA_EXPOSURE_V2_VERSION

    def __init__(self, bridge): self.bridge = bridge

    async def search(self, opportunity, context):
        result = await self.bridge.optimize(self.bridge.make_task(opportunity, context))
        t = result.optimizer_state.payload["telemetry"]
        return SearchResult(tuple(SearchCandidate(r.candidate_id, r.prompt, r.local_score,
            {}, {**r.backend_metadata, "generation":r.generation, "opportunity_id":opportunity.opportunity_id}) for r in result.candidates),
            result.termination_reason, dict(result.optimizer_state.payload), result.solver_calls,
            result.optimizer_calls,
            solver_tokens=result.optimizer_state.payload["token_accounting"]["solver_tokens"],
            search_meta_tokens=result.optimizer_state.payload["token_accounting"]["search_meta_tokens"],
            proposal_count=t["proposal_count"], team_candidate_count=t["team_candidate_count"],
            local_survival_update_count=t["local_survival_update_count"], strict_accepted_count=t["strict_accepted_count"],
            strict_rejected_exported_count=t["strict_rejected_exported_count"])
