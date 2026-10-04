"""Benchmark-neutral fixed-peer ports for Boolean equivalence plurality.

Only Optimize records enter this module. A separate gate owns Shadow records.
The orchestrator, variable evidence, GEPA, Pattern and Memory remain shared.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from .. import versions
import hashlib
import json

from ..peer_state import build_team_vote_state, soft_vote_utility
from ..vote_aligned_scheduler import classify_opportunity_lane
from ..team_search.schemas import TeamMiniBatchMetrics
from ..candidate_selection import common_monotone_safe_measurement_key
from .evaluation import FixedPeerPromotion
from .binary_responsibility import binary_plurality_snapshot, observation_from_outputs
from .schemas import EvidenceItem, SearchContractError, TeamEvaluation


@dataclass(frozen=True)
class CorrectnessExample:
    item: object
    reference: str

    def __post_init__(self):
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise SearchContractError("REFERENCE_INVALID_NOT_SOLVER_WRONG")


class BinaryTeamStateStore:
    def __init__(self, *, benchmark, examples, prompts, solver, aggregation,
                 freeze_initial_competence=False):
        self.benchmark = benchmark
        self.examples = tuple(examples)
        self.prompts = tuple(prompts)
        self.solver = solver
        self.aggregation = aggregation
        if not callable(getattr(aggregation, "aggregate_sync", None)):
            raise SearchContractError("SYNCHRONOUS_BINARY_AGGREGATION_PORT_REQUIRED")
        self.profiles = {}
        self.full_profiles = {}
        self.initial_member_scores = None
        self.initial_state_id = None
        self.freeze_initial_competence = freeze_initial_competence
        if len(self.prompts) != 5 or not self.examples or len({e.item.input_id for e in self.examples}) != len(self.examples):
            raise SearchContractError("BINARY_TEAM_INITIALIZATION_INVALID")

    def initialize(self):
        if self.freeze_initial_competence and self.initial_state_id is not None:
            raise SearchContractError("INITIAL_COMPETENCE_CANNOT_REBASE")
        self.profiles = {}
        for i, p in enumerate(self.prompts):
            if hasattr(self.solver, "observe_member"):
                self.solver.observe_member(i)
            self.profiles[i] = tuple(self.solver.solve(p, e.item, stage="initial", split="optimize")
                                     for e in self.examples)
        initial = self.snapshot()  # Parse and audit the actual complete initial state.
        if self.freeze_initial_competence:
            self.initial_member_scores = initial.member_scores
            self.initial_state_id = initial.team_state_id

    def snapshot(self):
        if set(self.profiles) != set(range(5)):
            raise SearchContractError("BINARY_TEAM_NOT_INITIALIZED")
        aggregates = tuple(self.aggregation.aggregate_sync(item=e.item,
            member_outputs=tuple(self.profiles[i][j] for i in range(5)), benchmark=self.benchmark)
            for j, e in enumerate(self.examples))
        observations = tuple(observation_from_outputs(benchmark=self.benchmark, item=e.item,
            member_outputs=tuple(self.profiles[i][j] for i in range(5)), gold=e.reference)
            for j, e in enumerate(self.examples))
        identity = hashlib.sha256(json.dumps(self.prompts, separators=(",", ":")).encode()).hexdigest()
        state = binary_plurality_snapshot(state_id=identity, member_prompts=self.prompts,
            observations=observations, capabilities=self.benchmark.capabilities)
        states = state.diagnostics["team_states"]
        if any(s.vote_correct != (self.benchmark.score_member_output(a.parsed_output, e.reference) == 1)
               for s, a, e in zip(states, aggregates, self.examples, strict=True)):
            raise SearchContractError("RESPONSIBILITY_AGGREGATION_CONFORMANCE_MISMATCH")
        outputs = tuple(tuple(self.benchmark.parse_member_output(raw, e.item)
                        for raw, e in zip(self.profiles[i], self.examples, strict=True)) for i in range(5))
        return replace(state, member_outputs=outputs, aggregated_outputs=tuple(a.parsed_output for a in aggregates),
            member_scores=tuple(float(sum(o.member_success[i] for o in observations)) for i in range(5)),
            team_scores={"vote_correct_count": float(sum(s.vote_correct for s in states))},
            residuals=tuple(s.question_hash for s in states if not s.vote_correct),
            diagnostics={**state.diagnostics, "raw_profiles": tuple(self.profiles[i] for i in range(5)),
                **({"evaluation_support_identity":hashlib.sha256(json.dumps(
                    [e.item.input_id for e in self.examples],separators=(",", ":")).encode()).hexdigest(),
                    "member_metric":"binary_correct_count", "evaluator_identity":getattr(self.benchmark,"evaluator_identity",None)}
                   if self.freeze_initial_competence else {})})

    def restore(self, snapshot):
        self.prompts = snapshot.member_prompts
        self.profiles = dict(enumerate(snapshot.diagnostics["raw_profiles"]))

    def replace_member(self, member_id, prompt):
        key = (self.snapshot().team_state_id, member_id, prompt)
        if key not in self.full_profiles:
            raise SearchContractError("COMMIT_REQUIRES_CURRENT_PARENT_FULL_PROFILE")
        self.prompts = tuple(prompt if i == member_id else p for i, p in enumerate(self.prompts))
        self.profiles = {**self.profiles, member_id: self.full_profiles[key]}
        return self.snapshot()

    @staticmethod
    def transition_fields(parent, child, member):
        before = tuple((s.question_hash, s.team_correctness[member]) for s in parent.diagnostics["team_states"])
        after = tuple((s.question_hash, s.team_correctness[member]) for s in child.diagnostics["team_states"])
        return dict(parent_correctness=before, child_correctness=after,
                    newly_fixed_ids=tuple(i for (i, a), (_, b) in zip(before, after, strict=True) if not a and b),
                    newly_broken_ids=tuple(i for (i, a), (_, b) in zip(before, after, strict=True) if a and not b))


class BinaryEvidenceSource:
    def __init__(self, store, history):
        self.store = store
        self.history = history

    def for_member(self, state, diagnosis, member_id):
        if state.team_state_id != self.store.snapshot().team_state_id:
            raise SearchContractError("EVIDENCE_PARENT_MISMATCH")
        assigned = {r.question_hash: r for r in diagnosis.benchmark_signals["assigned"].get(member_id, ())}
        latest = self.history.latest_member_transition(member_id)
        focus = set(latest.newly_broken_ids) if latest else set()
        anchor = set(latest.newly_fixed_ids) if latest else set()
        result = []
        for index, (example, row) in enumerate(zip(self.store.examples, state.diagnostics["team_states"], strict=True)):
            opportunity = assigned.get(row.question_hash)
            pivotal = False
            if row.vote_correct:
                validity = list(row.team_validity)
                validity[member_id] = False
                pivotal = not build_team_vote_state(question_hash=row.question_hash, gold_answer=row.gold_answer,
                    answers=row.team_answers, valid_vector=validity).vote_correct
            if not row.vote_correct and opportunity is not None:
                lane = classify_opportunity_lane(opportunity, diagnosis.benchmark_signals["margins"]) or "coverage"
                lane = "coverage" if lane == "pure_coverage" else lane
                roles = {"REPAIR", "TEAM_HARD", lane}
                tags = ("repair", lane, "team_hard")
                feedback = f"Improve general reasoning on this legal residual while preserving competence; lane={lane}."
            elif not row.vote_correct:
                lane = "team_hard"
                roles = {"TEAM_HARD", "vote_wrong"}
                tags = ("team_hard", "vote_wrong")
                feedback = "Repair current team failure while preserving general competence."
            else:
                lane = "preservation"
                roles = {"PRESERVATION", "target_correct" if row.team_correctness[member_id] else "team_correct"}
                tags = ("preservation", "target_correct" if row.team_correctness[member_id] else "team_correct")
                feedback = "Preserve current correct team behavior."
            if row.question_hash in focus:
                roles.add("TRANSITION_FOCUS")
            if row.question_hash in anchor:
                roles.add("TRANSITION_ANCHOR")
            output = state.member_outputs[member_id][index]
            result.append(EvidenceItem(example.item.input_id, "optimize", frozenset(roles),
                dict(input_payload=example.item.problem, gold=example.reference,
                     correctness_signal_identity=versions.TARGET_CORRECTNESS_SIGNAL_VERSION,
                     target_member_correct=bool(row.team_correctness[member_id]),
                     target_member_valid=bool(output.valid),
                     responsibility_labels=tuple(name for name in ('direct_flip','near_margin','coverage') if name in roles),
                     target_output=output.answer if output.valid else None, feedback=feedback,
                     legacy_group="repair" if "REPAIR" in roles else "preservation" if "PRESERVATION" in roles else "team_hard",
                     legacy_tags=tags, lane=lane, team_disagreement=len({a for a, v in zip(row.team_answers, row.team_validity) if v}),
                     residual_frequency=sum(not c for c in row.team_correctness), team_margin=row.plurality_margin,
                     mutation_sensitive=pivotal or row.question_hash in focus | anchor)))
        return tuple(result)


class FixedPeerTeamEvaluationProvider:
    def __init__(self, store, transition=None):
        self.store = store
        self.transition = transition
        self.invalid_predictions_are_incorrect = bool(getattr(store.benchmark, "invalid_predictions_are_incorrect", False))
        self.probed = []
        self.fulled = []

    def _parent(self, opportunity):
        parent = self.store.snapshot()
        if parent.team_state_id != opportunity.parent_state_id:
            raise SearchContractError("EVALUATION_PARENT_MISMATCH")
        return parent

    @staticmethod
    def evaluation(states):
        return TeamEvaluation(float(sum(s.vote_correct for s in states)), None,
            tuple(float(sum(s.team_correctness[i] for s in states)) for i in range(5)),
            aggregation_diagnostics={"terminal_invalid_count": sum(not all(s.team_validity) for s in states)})

    def _evaluate(self, opportunity, candidate, ids, stage):
        parent = self._parent(opportunity)
        selected = [(i, e) for i, e in enumerate(self.store.examples) if e.item.input_id in ids]
        if len(selected) != len(ids):
            raise SearchContractError("EVALUATION_SCOPE_NOT_OPTIMIZE")
        target = opportunity.target_member
        if hasattr(self.store.solver, "observe_member"):
            self.store.solver.observe_member(target)
        profile = tuple(self.store.solver.solve(candidate.prompt, e.item, stage=stage, split="optimize") for _, e in selected)
        aggregates = tuple(self.store.aggregation.aggregate_sync(item=e.item,
            member_outputs=tuple(profile[j] if m == target else self.store.profiles[m][i] for m in range(5)), benchmark=self.store.benchmark)
            for j, (i, e) in enumerate(selected))
        observations = tuple(observation_from_outputs(benchmark=self.store.benchmark, item=e.item,
            member_outputs=tuple(profile[j] if m == target else self.store.profiles[m][i] for m in range(5)),
            gold=e.reference) for j, (i, e) in enumerate(selected))
        states = binary_plurality_snapshot(state_id=parent.team_state_id, member_prompts=parent.member_prompts,
            observations=observations, capabilities=self.store.benchmark.capabilities).diagnostics["team_states"]
        if any(s.vote_correct != (self.store.benchmark.score_member_output(a.parsed_output, e.reference) == 1)
               for s, a, (_, e) in zip(states, aggregates, selected, strict=True)):
            raise SearchContractError("EVALUATION_AGGREGATION_CONFORMANCE_MISMATCH")
        before = tuple(parent.diagnostics["team_states"][i] for i, _ in selected)
        if stage == "full":
            self.store.full_profiles[(parent.team_state_id, target, candidate.prompt)] = profile
        return before, states

    async def active(self, opportunity):
        return self.evaluation(self._parent(opportunity).diagnostics["team_states"])

    async def team_probe(self, opportunity, candidate):
        self.probed.append(candidate.candidate_id)
        ids = {r.example_id for r in opportunity.evidence.team_probe_evidence}
        before, after = self._evaluate(opportunity, candidate, ids, "team_probe")
        old, new = self.evaluation(before), self.evaluation(after)
        t = opportunity.target_member
        legal = {r.question_hash for r in opportunity.diagnosis.benchmark_signals["assigned"].get(t, ())}
        target = int(new.member_scores[t] - old.member_scores[t])
        vote = int(new.aggregate_score - old.aggregate_score)
        metrics = TeamMiniBatchMetrics(target_delta=target, vote_delta=vote, team_net_vote_delta=vote,
            responsibility_delta=sum(not a.team_correctness[t] and b.team_correctness[t] and a.question_hash in legal
                                     for a, b in zip(before, after, strict=True)),
            broad_delta=target, invalid_delta=sum(not b.team_validity[t] for b in after)-sum(not a.team_validity[t] for a in before))
        catastrophe = (metrics.invalid_delta > 0 and not self.invalid_predictions_are_incorrect) or metrics.vote_delta <= -2 or metrics.team_net_vote_delta <= -3
        return replace(new, aggregation_diagnostics={"team_probe_metrics": metrics,
            **({"scientific_risk_code": "TEAM_PROBE_REJECTION"} if catastrophe else {}),
            **({"operational_failure": True} if not self.invalid_predictions_are_incorrect and any(not s.team_validity[t] for s in after) else {})})

    async def full(self, opportunity, candidate):
        self.fulled.append(candidate.candidate_id)
        before, after = self._evaluate(opportunity, candidate, {e.item.input_id for e in self.store.examples}, "full")
        old, new = self.evaluation(before), self.evaluation(after)
        t = opportunity.target_member
        invalid_delta = sum(not b.team_validity[t] for b in after)-sum(not a.team_validity[t] for a in before)
        fixed = sum(not a.vote_correct and b.vote_correct for a, b in zip(before, after, strict=True))
        broken = sum(a.vote_correct and not b.vote_correct for a, b in zip(before, after, strict=True))
        safe = (new.member_scores[t] >= old.member_scores[t] and new.aggregate_score >= old.aggregate_score
                and (new.member_scores[t] > old.member_scores[t] or new.aggregate_score > old.aggregate_score) and invalid_delta <= 0)
        measurement = replace(new, aggregation_diagnostics=dict(terminal_invalid_delta=invalid_delta,
            team_newly_fixed_count=fixed, team_newly_broken_count=broken,
            mean_soft_vote_utility=sum(soft_vote_utility(s.gold_vote_count, s.plurality_margin) for s in after)/len(after),
            target_invalid_count=sum(not s.team_validity[t] for s in after),
            **({"operational_failure": True} if not self.invalid_predictions_are_incorrect and any(not s.team_validity[t] for s in after) else {})))
        if self.transition is not None:
            safe = self.transition.allows(old, measurement, t)
            measurement = replace(measurement, aggregation_diagnostics={**measurement.aggregation_diagnostics,
                "team_score_delta": new.aggregate_score - old.aggregate_score,
                "target_score_delta": new.member_scores[t] - old.member_scores[t],
                "target_initial_margin": new.member_scores[t] - self.transition.initial_scores[t]})
        return replace(measurement, aggregation_diagnostics={**measurement.aggregation_diagnostics,
            **({"scientific_risk_code": "COMMON_SAFE_REJECTION"} if not safe else {})})


class FixedPeerCommonSafe:
    """The frozen Common-Safe feasibility and S0-S2 key on neutral records."""
    identity = "common_safe_v1"
    def select(self, parent, candidates):
        from .schemas import TransitionDecision
        feasible = []
        for r in candidates:
            f = r.full
            if not r.promoted or f is None:
                continue
            t = r.diagnostics["target_member"]
            if (f.member_scores[t] >= parent.member_scores[t] and f.aggregate_score >= parent.aggregate_score
                    and (f.member_scores[t] > parent.member_scores[t] or f.aggregate_score > parent.aggregate_score)
                    and f.aggregation_diagnostics["terminal_invalid_delta"] <= 0):
                d = f.aggregation_diagnostics
                key = common_monotone_safe_measurement_key(vote_correct=f.aggregate_score, target_correct=f.member_scores[t],
                    soft_vote_utility=d["mean_soft_vote_utility"], vote_loss_count=d["team_newly_broken_count"],
                    invalid_count=d["target_invalid_count"], generation=r.candidate.backend_details.get("generation", 0),
                    prompt_hash=hashlib.sha256(r.candidate.prompt.encode()).hexdigest())
                feasible.append((key, r))
        return TransitionDecision(max(feasible, key=lambda x: x[0])[1], "COMMON_SAFE_WINNER") if feasible else TransitionDecision(None, "NO_COMMON_SAFE_WINNER")
