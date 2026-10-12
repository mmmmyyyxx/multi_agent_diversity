"""A separate capability owns gate data; only aggregate feedback crosses out."""
from __future__ import annotations

from ..benchmarks.access import AdaptiveGateFeedback
from ..shadow_gate import ShadowGateMetrics, evaluate_configured_shadow_gate
from .binary_responsibility import binary_plurality_snapshot, observation_from_outputs
from .schemas import SearchContractError


def private_binary_gate(*, benchmark, load_examples, expected_count, solver, store):
    # Raw gate records/profiles remain in this closure, never on search state,
    # Pattern inputs, history or the public gate object.
    examples = None
    profiles = {}

    class Gate:
        operational_failure = False
        feedback = None
        winner_count = 0

        async def check_team(self, proposed, baseline):
            """One frozen final assembly, private joint outcomes, no reselection."""
            nonlocal examples
            if store.snapshot().team_state_id!=baseline.team_state_id or len(proposed)!=5:
                raise SearchContractError('FINAL_GATE_BASELINE_MISMATCH')
            if examples is None:
                examples=tuple(load_examples())
                if len(examples)!=expected_count:raise SearchContractError('GATE_MANIFEST_CARDINALITY_MISMATCH')
            def profile(prompt,member):
                lane_key=(member,prompt) if getattr(getattr(solver,'broker',None),'member_lane_policy',False) else prompt
                if lane_key not in profiles:
                    if hasattr(solver,'observe_member'):solver.observe_member(member)
                    batch=getattr(solver,'solve_batch',None)
                    profiles[lane_key]=tuple(batch(prompt,[e.item for e in examples],stage='adaptive_gate',split='shadow')
                        if callable(batch) else (solver.solve(prompt,e.item,stage='adaptive_gate',split='shadow') for e in examples))
                return profiles[lane_key]
            def table(prompts):
                predictions=tuple(profile(p,m) for m,p in enumerate(prompts))
                obs=tuple(observation_from_outputs(benchmark=benchmark,item=e.item,
                    member_outputs=tuple(predictions[m][j] for m in range(5)),gold=e.reference)
                    for j,e in enumerate(examples))
                from .system_prompt import team_identity
                states=binary_plurality_snapshot(state_id=team_identity(prompts),member_prompts=prompts,
                    observations=obs,capabilities=benchmark.capabilities).diagnostics['team_states']
                aggregates=tuple(store.aggregation.aggregate_sync(item=e.item,
                    member_outputs=tuple(predictions[m][j] for m in range(5)),benchmark=benchmark)
                    for j,e in enumerate(examples))
                if any(s.vote_correct!=(benchmark.score_member_output(a.parsed_output,e.reference)==1)
                    for s,a,e in zip(states,aggregates,examples,strict=True)):
                    raise SearchContractError('GATE_AGGREGATION_CONFORMANCE_MISMATCH')
                return states
            before,after=table(baseline.member_prompts),table(proposed)
            if not getattr(benchmark,'invalid_predictions_are_incorrect',False) and any(not all(s.team_validity) for s in (*before,*after)):
                raise SearchContractError('GATE_INVALID_RESPONSE')
            changed=[m for m in range(5) if proposed[m]!=baseline.member_prompts[m]]
            # Apply the existing winner gate to each changed member, with one
            # jointly measured Vote; no new threshold or preservation panel.
            targets=changed or [0]
            decisions=[evaluate_configured_shadow_gate(ShadowGateMetrics(
                sum(s.vote_correct for s in before),sum(s.vote_correct for s in after),
                sum(s.team_correctness[m] for s in before),sum(s.team_correctness[m] for s in after),expected_count),
                expected_row_count=expected_count) for m in targets]
            self.feedback=AdaptiveGateFeedback(float(sum(s.vote_correct for s in after)),all(d.passed for d in decisions))
            self.winner_count+=1
            return self.feedback.passed

        async def check(self, opportunity, candidate):
            nonlocal examples
            if candidate.full is None or not candidate.promoted or store.snapshot().team_state_id != opportunity.parent_state_id:
                raise SearchContractError("GATE_REQUIRES_FROZEN_FULL_WINNER")
            if (getattr(opportunity, 'evaluation_plan', {}).get('current_parent_binding') and
                    candidate.candidate.backend_details.get('parent_state_id') != opportunity.parent_state_id):
                raise SearchContractError('CANDIDATE_PARENT_MISMATCH')
            if examples is None:
                examples = tuple(load_examples())
                if len(examples) != expected_count:
                    raise SearchContractError("GATE_MANIFEST_CARDINALITY_MISMATCH")
            def profile(prompt, member):
                lane_key=(member,prompt) if getattr(getattr(solver,'broker',None),'member_lane_policy',False) else prompt
                if lane_key not in profiles:
                    if hasattr(solver, "observe_member"):
                        solver.observe_member(member)
                    batch=getattr(solver,'solve_batch',None)
                    profiles[lane_key] = (tuple(batch(prompt,[e.item for e in examples],stage='adaptive_gate',split='shadow'))
                        if callable(batch) else tuple(solver.solve(prompt, e.item, stage="adaptive_gate", split="shadow") for e in examples))
                return profiles[lane_key]
            prompts = store.snapshot().member_prompts
            parent = tuple(profile(p, i) for i, p in enumerate(prompts))
            target = opportunity.target_member
            proposed = profile(candidate.candidate.prompt, target)
            def table(candidate_profile):
                aggregates = tuple(store.aggregation.aggregate_sync(item=e.item,
                    member_outputs=tuple(candidate_profile[j] if i == target else parent[i][j] for i in range(5)), benchmark=benchmark)
                    for j, e in enumerate(examples))
                obs = tuple(observation_from_outputs(benchmark=benchmark, item=e.item,
                    member_outputs=tuple(candidate_profile[j] if i == target else parent[i][j] for i in range(5)),
                    gold=e.reference) for j, e in enumerate(examples))
                states = binary_plurality_snapshot(state_id=opportunity.parent_state_id, member_prompts=prompts,
                    observations=obs, capabilities=benchmark.capabilities).diagnostics["team_states"]
                if any(s.vote_correct != (benchmark.score_member_output(a.parsed_output, e.reference) == 1)
                       for s, a, e in zip(states, aggregates, examples, strict=True)):
                    raise SearchContractError("GATE_AGGREGATION_CONFORMANCE_MISMATCH")
                return states
            before, after = table(parent[target]), table(proposed)
            self.operational_failure = (not getattr(benchmark, "invalid_predictions_are_incorrect", False)
                and any(not all(s.team_validity) for s in (*before, *after)))
            if self.operational_failure:
                raise SearchContractError("GATE_INVALID_RESPONSE")
            metrics = ShadowGateMetrics(sum(s.vote_correct for s in before), sum(s.vote_correct for s in after),
                sum(s.team_correctness[target] for s in before), sum(s.team_correctness[target] for s in after), expected_count)
            decision = evaluate_configured_shadow_gate(metrics, expected_row_count=expected_count)
            self.feedback = AdaptiveGateFeedback(float(metrics.candidate_vote_correct), decision.passed)
            self.winner_count += 1
            return self.feedback.passed

    return Gate()
