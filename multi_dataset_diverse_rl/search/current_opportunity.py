"""Current target-first Gradient opportunity graph, without version dispatch."""
from .orchestrator import OpportunityBuilder
from .schemas import OptimizationOpportunity, SearchContractError

class CurrentOpportunityBuilder(OpportunityBuilder):
    def __init__(self, *, patterns, **kwargs):
        super().__init__(**kwargs)
        self.patterns = patterns

    def build(self, *, state, diagnosis, history, update_index):
        # Freeze all feasibility and ranking before optional target diagnostic.
        rows_by_member = {m:tuple(self.source.for_member(state, diagnosis, m)) for m in sorted(diagnosis.responsibility)}
        feasible = tuple(m for m,rows in rows_by_member.items()
            if self.feasibility.feasible(state, diagnosis, m, rows)
            and self.evidence.can_compose(rows))
        target = self.target.select(state, diagnosis, feasible, history)
        member = target.selected_member
        self.last_reason = target.reason
        if member is None: return None
        if member >= len(state.member_prompts): raise SearchContractError("target outside team")
        rows = rows_by_member[member]
        pattern = self.patterns.analyze(state, diagnosis, member, rows, history)
        view,audit=self.evidence.compose_opportunity(state,diagnosis,member,rows,pattern,
            ordinal=history.target_counts.get(member,0))
        signal = diagnosis.responsibility[member]
        return OptimizationOpportunity(f"{state.team_state_id}:{update_index}:{member}",
            state.team_state_id, member, state.member_prompts[member],
            {"selection_reason":target.reason,"raw_responsibility":signal.raw_value, "target_score":target.target_scores[member],
             "eligible_members":target.eligible_members,
             **{"target_scores":dict(target.target_scores),
                 "feasibility_by_member":{m:m in feasible for m in rows_by_member}}}, diagnosis, view, pattern_context=pattern,
            search_budget={"metric_calls":self.search_metric_budget},
            evaluation_plan={"max_promoted":2, "evidence_universe":rows,
                             'optimization_evidence_policy':self.evidence.policy,
                             "evidence_audit":audit, "v2_candidate_contract":True,
                             "current_parent_binding":True,
                             **{"allocation_failure_counts":{m:history.failure_counts.get(m, 0) for m in rows_by_member},
                                 "prior_opportunity_counts":{m:history.target_counts.get(m, 0) for m in rows_by_member},
                                 "prior_exposure_counts":{m:update_index for m in rows_by_member}}})
