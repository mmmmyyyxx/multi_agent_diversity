"""Current BBH/plurality state diagnosis without the old controller graph."""

from __future__ import annotations

from typing import Any

from ..peer_state import build_team_vote_state
from ..vote_aligned_scheduler import classify_opportunity_lane
from .history import HistoryState
from .schemas import (
    Diagnosis, EvidenceItem, ParsedOutput, SearchContractError,
    TeamStateSnapshot,
)


class CurrentBBHStateSource:
    """Freeze one parent from the existing BBH system at its public state seam."""

    def __init__(self, system: Any) -> None:
        self.system = system

    def snapshot(self) -> TeamStateSnapshot:
        states, _, opportunities = self.system.current_states_and_opportunities()
        prompts = tuple(agent.current_prompt for agent in self.system.agents)
        if len(prompts) != 5:
            raise SearchContractError("current BBH requires exactly five members")
        outputs = tuple(tuple(ParsedOutput(row.team_answers[member],
                                           row.team_validity[member]) for row in states)
                        for member in range(len(prompts)))
        aggregated = tuple(ParsedOutput(row.vote_answer, not row.top_tie)
                           for row in states)
        scores = tuple(float(sum(row.team_correctness[member] for row in states))
                       for member in range(len(prompts)))
        residuals = tuple(row.question_hash for row in states if not row.vote_correct)
        return TeamStateSnapshot(
            self.system.team_prompt_state_hash(), prompts,
            member_outputs=outputs, aggregated_outputs=aggregated,
            member_scores=scores,
            team_scores={"vote_correct_count": float(sum(row.vote_correct for row in states))},
            residuals=residuals,
            diagnostics={"team_states": tuple(states),
                         "opportunities": opportunities},
        )


class PluralityResponsibilityAnalyzer:
    """Current raw overlapping legal D/N/C analysis, before any target routing."""

    identity = "plurality_raw_responsibility_v1"

    def __init__(self, system: Any) -> None:
        self.system = system

    def analyze(self, state: TeamStateSnapshot, history: HistoryState) -> Diagnosis:
        del history
        if self.system.team_prompt_state_hash() != state.team_state_id:
            raise SearchContractError("BBH parent state changed during diagnosis")
        from .binary_responsibility import BinaryPluralityResponsibilityAnalyzer
        from .schemas import BenchmarkCapabilities
        return BinaryPluralityResponsibilityAnalyzer(
            BenchmarkCapabilities(True, True, True, True, True),
        ).analyze(state, None, responsibility_state=self.system.responsibility_state)


class CurrentBBHEvidenceSource:
    """Role-bearing Optimize evidence from one immutable parent snapshot."""

    def __init__(self, system: Any, history: HistoryState) -> None:
        self.system = system
        self.history = history

    def for_member(self, state: TeamStateSnapshot, diagnosis: Diagnosis,
                   member_id: int) -> tuple[EvidenceItem, ...]:
        if self.system.fixed_probe is None:
            raise SearchContractError("BBH Optimize probe is not initialized")
        if state.team_state_id != self.system.team_prompt_state_hash():
            raise SearchContractError("BBH evidence parent changed")
        team_states = {row.question_hash: row for row in state.diagnostics["team_states"]}
        assigned = {
            row.question_hash: row
            for row in diagnosis.benchmark_signals["assigned"].get(member_id, ())
        }
        margins = diagnosis.benchmark_signals["margins"]
        latest = self.history.latest_member_transition(member_id)
        focus = set(latest.newly_broken_ids) if latest else set()
        anchor = set(latest.newly_fixed_ids) if latest else set()
        changed = focus | anchor
        rows: list[EvidenceItem] = []
        for index, example in enumerate(self.system.fixed_probe.examples):
            row = team_states[example.question_hash]
            opportunity = assigned.get(example.question_hash)
            target_answer = self.system.active_profiles[member_id][index]
            answers = row.team_answers
            validity = row.team_validity
            disagreement = len({answer for answer, valid in zip(answers, validity)
                                if valid and answer})
            residual_frequency = sum(not correct for correct in row.team_correctness)
            pivotal = False
            if row.vote_correct:
                without_answers = list(answers)
                without_validity = list(validity)
                without_answers[member_id] = ""
                without_validity[member_id] = False
                pivotal = not build_team_vote_state(
                    question_hash=row.question_hash,
                    gold_answer=row.gold_answer,
                    answers=without_answers,
                    valid_vector=without_validity,
                    normalize_answer=self.system.normalize_answer,
                    match_answer=self.system.match_answer,
                    tie_break=self.system.protocol.tie_policy,
                ).vote_correct
            if not row.vote_correct and opportunity is not None:
                lane = classify_opportunity_lane(opportunity, margins) or "coverage"
                roles = {"REPAIR", "TEAM_HARD", lane}
                legacy_tags = ("repair", lane, "team_hard")
                feedback = (
                    "Improve the general decision procedure for this legal residual "
                    f"while preserving unrelated competence; lane={lane}."
                )
            elif not row.vote_correct:
                lane = "team_hard"
                roles = {"TEAM_HARD", "vote_wrong"}
                legacy_tags = ("team_hard", "vote_wrong")
                feedback = (
                    "This is a current team failure. Preserve general competence "
                    "while testing team-level transfer."
                )
            else:
                lane = "preservation"
                roles = {"PRESERVATION", "target_correct" if row.team_correctness[member_id]
                         else "team_correct"}
                legacy_tags = ("preservation", "target_correct" if
                               row.team_correctness[member_id] else "team_correct")
                feedback = "Preserve current correct team behavior and the immutable output contract."
            if example.question_hash in focus:
                roles.add("TRANSITION_FOCUS")
            if example.question_hash in anchor:
                roles.add("TRANSITION_ANCHOR")
            rows.append(EvidenceItem(
                example.question_hash, "optimize", frozenset(roles),
                {"input_payload": example.question,
                 "gold": example.gold_answer,
                 "target_output": target_answer.answer if target_answer.valid else None,
                 "feedback": feedback,
                 "legacy_group": ("repair" if "REPAIR" in roles else
                                  "preservation" if "PRESERVATION" in roles
                                  else "team_hard"),
                 "legacy_tags": legacy_tags,
                 "lane": lane,
                 "team_disagreement": disagreement,
                 "residual_frequency": residual_frequency,
                 "team_margin": int(row.plurality_margin),
                 "mutation_sensitive": pivotal or example.question_hash in changed},
            ))
        return tuple(rows)
