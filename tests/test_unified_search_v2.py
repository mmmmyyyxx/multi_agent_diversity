"""Guarded structural proofs for V2, including the actual pinned GEPA seam."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import verify_frozen_gepa_engine_contract
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample, LocalOptimizationTask, LocalOptimizerBudget
from multi_dataset_diverse_rl.search.aggregation import PluralityAggregation
from multi_dataset_diverse_rl.search.context import SearchContextComposer
from multi_dataset_diverse_rl.search.current_bbh_runtime import CurrentBBHPromotion
from multi_dataset_diverse_rl.search.evaluation import CandidateEvaluationPipeline
from multi_dataset_diverse_rl.search.gepa_v2 import (EphemeralProposalCallback, GEPATeamExposureOptimizer,
    GEPATeamCandidateExposureEngine, V2GEPABridge, V2ReflectionAdapter)
from multi_dataset_diverse_rl.search.history import HistoryState, NullMemoryProvider, NullPatternAnalyzer
from multi_dataset_diverse_rl.search.memory import OpportunityOutcome, StructuredLongTermMemoryProviderV1
from multi_dataset_diverse_rl.search.orchestrator import StateAnalyzer, UnifiedSearchOrchestrator, UnifiedSearchContext
from multi_dataset_diverse_rl.search.patterns import PatternDiagnosticV1, PatternHypothesis
from multi_dataset_diverse_rl.search.policies import ResponsibilitySignal, TargetPolicyV1, GlobalStopPolicy
from multi_dataset_diverse_rl.search.runtime_v2 import V2OpportunityBuilder
from multi_dataset_diverse_rl.search.schemas import (BenchmarkCapabilities, Diagnosis, EvidenceItem,
    SearchContractError, SearchMethodConfig, TeamEvaluation, TeamStateSnapshot, EvaluatedCandidate, SearchCandidate)
from multi_dataset_diverse_rl.search.transition import CommonSafeTransitionPolicy, TeamStateCommitter
from multi_dataset_diverse_rl.search.variable_evidence import (PatternCapableVariableEvidencePolicyV1,
    VariableEvidenceFeasibilityV1, validation_capacity)
from multi_dataset_diverse_rl.team_search.schemas import TeamMiniBatchMetrics

A = "Compare semantic roles and explicit constraints carefully before resolving the referent."
B = "Check relational consistency systematically before selecting the most compatible interpretation."
PARENT = "Reason about possible interpretations using grammatical and contextual consistency."
C = "Resolve local ambiguities eagerly using salient relations without preserving prior distinctions."
LIMITS = dict(top_k_private=2, top_k_shared=2, max_context_chars=6000,
              private_storage_limit=4, shared_storage_limit=4)


def rows(n=5, repairs=None):
    return tuple(EvidenceItem(f"e{i}", "optimize", frozenset({"REPAIR", "direct_flip", "TEAM_HARD"}
        if repairs is None or i < repairs else {"PRESERVATION"}),
        dict(input_payload=f"Synthetic public item {i}.", gold="A", target_output="B",
             feedback="Test structural repair.", legacy_group="repair" if repairs is None or i < repairs else "preservation",
             legacy_tags=("repair", "direct_flip") if repairs is None or i < repairs else ("preservation",),
             lane="direct_flip", team_disagreement=2 if repairs is None or i < repairs else 0,
             mutation_sensitive=False, team_margin=1, residual_frequency=1)) for i in range(n))


class Store:
    def __init__(self): self.prompts = (PARENT,) * 5
    def snapshot(self):
        return TeamStateSnapshot(hashlib.sha256(repr(self.prompts).encode()).hexdigest(), self.prompts)
    def restore(self, snapshot): self.prompts = snapshot.member_prompts
    def replace_member(self, member, prompt):
        self.prompts = tuple(prompt if i == member else p for i,p in enumerate(self.prompts))
        return self.snapshot()


class Analyzer:
    def __init__(self, member=2): self.member = member
    def analyze(self, state, history):
        return Diagnosis({self.member:ResponsibilitySignal(self.member, 2, 0, 0, "direct_flip")})


class Source:
    def __init__(self, values): self.values = values
    def for_member(self, *args): return self.values


def builder(values=None, patterns=None):
    return V2OpportunityBuilder(source=Source(values if values is not None else rows()),
        feasibility=VariableEvidenceFeasibilityV1(), target=TargetPolicyV1(),
        evidence=PatternCapableVariableEvidencePolicyV1(), patterns=patterns)


def opportunity(values=None, patterns=None):
    state = Store().snapshot(); diagnosis = Analyzer().analyze(state, HistoryState())
    return builder(values, patterns).build(state=state, diagnosis=diagnosis, history=HistoryState(), update_index=0)


class Solver:
    solver_contract_id = "COMMON_SOLVER_CONTRACT_V1"
    output_contract_id = "synthetic"
    def __init__(self): self.calls = []; self.requests = []; self.structural=False; self.failure=False
    def evaluate(self, prompt, row):
        if self.failure: raise RuntimeError("fake provider timeout")
        self.calls.append((prompt, row.example_id)); self.requests.append((prompt, row.input_payload))
        if self.structural:
            correct = row.example_id in {
                PARENT:{"e1","e3","e4"}, A:{"e1","e2","e3","e4"},
                B:{"e0","e3","e4"}, C:{"e1","e2"}}[prompt]
        else:
            correct = prompt == A
        return LocalSolverObservation("A" if correct else "B", "synthetic reasoning", correct, True,
                                      input_tokens=1, output_tokens=1, provider_called=True)


class Reflection:
    def __init__(self, answer=B): self.answer=answer; self.calls=[]
    def __call__(self, prompt):
        self.calls.append(prompt)
        return "```"+self.answer+"```"


def scripted_gepa(**kw):
    """Fake search lifecycle; score vectors come from the injected Solver adapter."""
    cb=kw["callbacks"][0]; adapter=kw["adapter"]; parent=kw["seed_candidate"]; val=kw["valset"]
    root=adapter.evaluate(val,parent)
    candidates=[parent]; parents=[[None]]; scores=[sum(root.scores)/len(val)]
    for iteration,prompt in enumerate((C,) if getattr(adapter.evaluator,"catastrophe",False) else (A,B),1):
        by_id={r.example_id:r for r in kw["trainset"]}
        minibatch=[by_id[i] for i in ("e0","e1","e2")]
        cb.on_minibatch_sampled({"iteration":iteration,"minibatch_ids":[r.example_id for r in minibatch]})
        before=adapter.evaluate(minibatch,parent,True)
        cb.on_evaluation_end({"iteration":iteration,"candidate_idx":0,"scores":before.scores})
        cb.on_proposal_end({"iteration":iteration,"new_instructions":{"decision_procedure":prompt}})
        after=adapter.evaluate(minibatch,{"decision_procedure":prompt})
        cb.on_evaluation_end({"iteration":iteration,"candidate_idx":None,"scores":after.scores})
        if sum(after.scores)>sum(before.scores):
            full=adapter.evaluate(val,{"decision_procedure":prompt})
            candidates.append({"decision_procedure":prompt});parents.append([0]);scores.append(sum(full.scores)/len(val))
            cb.on_candidate_accepted({"iteration":iteration,"new_candidate_idx":len(candidates)-1,
                                     "new_score":sum(after.scores),"parent_ids":[0]})
        else:
            cb.on_candidate_rejected({"iteration":iteration,"old_score":sum(before.scores),
                                     "new_score":sum(after.scores),"reason":"strict_local_non_improvement"})
    payload=dict(candidates=candidates,parents=parents,val_aggregate_scores=scores)
    return SimpleNamespace(candidates=candidates,parents=parents,num_candidates=len(candidates),
        val_aggregate_scores=scores,val_subscores=[{i:s for i,s in enumerate(root.scores)}, {i:float(scores[1]) for i in range(len(val))}],
        per_val_instance_best_candidates={i:{1} for i in range(len(val))},
        discovery_eval_counts=[len(val),len(val)*2+6],to_dict=lambda:payload)


def make_engine(tmp_path, scripted=True):
    solver=Solver(); reflection=Reflection()
    optimizer=GEPATeamExposureOptimizer(evaluator=solver,reflection_lm=reflection,
        accounting_reader=lambda:{"successful_calls":len(reflection.calls),"input_tokens":2*len(reflection.calls),"output_tokens":len(reflection.calls)},run_root=tmp_path, optimize_fn=scripted_gepa if scripted else None)
    bridge=V2GEPABridge(optimizer=optimizer,history=HistoryState(),seed=81,
                       solver_contract_id="COMMON_SOLVER_CONTRACT_V1",output_contract_id="synthetic")
    return GEPATeamCandidateExposureEngine(bridge),solver,reflection


class TeamProvider:
    def __init__(self, catastrophe=False): self.probed=[];self.fulled=[];self.catastrophe=catastrophe
    @staticmethod
    def table(prompt):
        from multi_dataset_diverse_rl.peer_state import build_team_vote_state
        target={PARENT:{"e1","e3","e4"},A:{"e1","e2","e3","e4"},
                B:{"e0","e3","e4"},C:{"e1","e2"}}[prompt]
        states=[]
        for i in range(5):
            # e0/e3/e4: target is pivotal; e1: peers preserve the vote;
            # e2: target-only improvement cannot flip four wrong peers.
            peers=("A","A","B","B") if i in (0,3,4) else ("A",)*4 if i==1 else ("B",)*4
            answers=(*peers[:2],"A" if f"e{i}" in target else "B",*peers[2:])
            states.append(build_team_vote_state(question_hash=f"e{i}",gold_answer="A",answers=answers,valid_vector=(True,)*5))
        return states
    @classmethod
    def evaluation(cls,prompt,ids=None):
        table=[s for s in cls.table(prompt) if ids is None or s.question_hash in ids]
        return TeamEvaluation(float(sum(s.vote_correct for s in table)),None,
                              tuple(float(sum(s.team_correctness[i] for s in table)) for i in range(5)))
    async def active(self, opp): return self.evaluation(opp.parent_prompt)
    async def team_probe(self,opp,candidate):
        self.probed.append(candidate.prompt)
        ids={r.example_id for r in opp.evidence.team_probe_evidence}
        before=self.evaluation(opp.parent_prompt,ids);after=self.evaluation(candidate.prompt,ids)
        vote=int(after.aggregate_score-before.aggregate_score)
        target=int(after.member_scores[opp.target_member]-before.member_scores[opp.target_member])
        metrics=TeamMiniBatchMetrics(target_delta=target,vote_delta=vote,team_net_vote_delta=vote,
                                    responsibility_delta=int(candidate.prompt==B),broad_delta=target)
        return replace(after,aggregation_diagnostics={"bbh_probe":metrics,
                        **({"scientific_risk_code":"TEAM_PROBE_REJECTION"} if vote<=-2 else {})})
    async def full(self,opp,candidate):
        self.fulled.append(candidate.prompt)
        before=self.table(opp.parent_prompt);after=self.table(candidate.prompt)
        fixed=sum(not a.vote_correct and b.vote_correct for a,b in zip(before,after))
        broken=sum(a.vote_correct and not b.vote_correct for a,b in zip(before,after))
        evaluation=self.evaluation(candidate.prompt);parent=self.evaluation(opp.parent_prompt)
        risk=({"scientific_risk_code":"COMMON_SAFE_REJECTION"} if evaluation.aggregate_score < parent.aggregate_score
              or evaluation.member_scores[opp.target_member] < parent.member_scores[opp.target_member] else {})
        return replace(evaluation,aggregation_diagnostics={"team_newly_fixed_count":fixed,"team_newly_broken_count":broken,**risk})


class Gate:
    def __init__(self, passed=True):self.passed=passed;self.seen=[]
    async def check(self,opp,row):self.seen.append(row.candidate.prompt);return self.passed


def run_fixture(tmp_path, *, patterns=None, memory=None, catastrophe=False, gate_passed=True, max_opportunities=1, solver_failure=False, store=None, gate=None):
    engine,solver,reflection=make_engine(tmp_path)
    solver.structural=True;solver.catastrophe=catastrophe;solver.failure=solver_failure
    contexts=[]
    original = engine.bridge.optimizer._delegate_optimize
    def capture(**kwargs):
        contexts.append(kwargs["adapter"].optimization_context)
        return original(**kwargs)
    engine.bridge.optimizer._delegate_optimize = capture
    state=store if store is not None else Store(); history=HistoryState();provider=TeamProvider(catastrophe)
    gate=gate if gate is not None else Gate(gate_passed)
    mechanism={}
    if patterns is not None: mechanism["pattern_provider_binding"]="synthetic_fake_v1"
    if memory is not None: mechanism["memory"]=memory.limits
    method=SearchMethodConfig.v2(pattern_policy=patterns.identity if patterns is not None else versions.UNIFIED_NULL_PATTERN_VERSION,
        memory_policy=memory.identity if memory is not None else versions.UNIFIED_NULL_MEMORY_VERSION,mechanism_config=mechanism)
    values=rows()
    if patterns is None:
        values=tuple(replace(r,roles=frozenset({"REPAIR","direct_flip","TEAM_HARD"} if r.example_id=="e0" else
                            {"TEAM_HARD"} if r.example_id=="e2" else {"PRESERVATION"}),
                            signals={**r.signals,"mutation_sensitive":r.example_id in {"e1","e3","e4"},
                                     "legacy_group":"repair" if r.example_id=="e0" else "team_hard" if r.example_id=="e2" else "preservation"}) for r in values)
    built=builder(values,patterns=patterns)
    orchestrator=UnifiedSearchOrchestrator(method=method,benchmark=SimpleNamespace(capabilities=BenchmarkCapabilities(True,True,True)),
        aggregation=PluralityAggregation(),state=state,analyzer=StateAnalyzer(Analyzer()),opportunities=built,engine=engine,
        evaluation=CandidateEvaluationPipeline(provider,CurrentBBHPromotion()),transition=CommonSafeTransitionPolicy(),
        gate=gate,committer=TeamStateCommitter(state),history=history,memory=memory)
    result=asyncio.run(orchestrator.run(max_opportunities=max_opportunities))
    return SimpleNamespace(result=result, engine=engine,solver=solver,reflection=reflection,provider=provider,
                           state=state,history=history,gate=gate,orchestrator=orchestrator,contexts=contexts)


def test_local_rejected_team_good_candidate_can_commit(tmp_path):
    f=run_fixture(tmp_path)
    candidates=f.engine.bridge.raw_candidates
    rejected=[c for c in candidates.values() if c.backend_metadata["gepa_local_acceptance_status"]=="REJECTED"]
    assert len(rejected)==1 and rejected[0].prompt==B and rejected[0].backend_metadata["local_acceptance_delta"]==0
    assert f.provider.probed==[A,B] and B in f.provider.fulled
    assert f.gate.seen==[B] and f.state.prompts[2]==B
    assert len(f.result.transitions)==1 and f.result.trace[0].committed_candidate_id==rejected[0].candidate_id
    assert f.result.trace[0].proposal_exposed==2 and f.result.trace[0].local_search_survival_update==1
    assert f.result.trace[0].evidence_audit["search_validation_count"]==5


def test_local_accepted_team_bad_candidate_is_rejected(tmp_path):
    f=run_fixture(tmp_path,catastrophe=True)
    assert f.provider.probed==[C] and f.provider.fulled==[] and f.gate.seen==[]
    assert f.state.prompts==(PARENT,)*5 and not f.result.transitions
    assert any(c.backend_metadata["gepa_search_survived"] for c in f.engine.bridge.raw_candidates.values())


def test_gepa_internal_strict_acceptance_unchanged(tmp_path):
    verify_frozen_gepa_engine_contract()
    engine,solver,reflection=make_engine(tmp_path,False)
    opp=opportunity();ctx=UnifiedSearchContext(None,None,HistoryState(),{}, {})
    result=asyncio.run(engine.search(opp,ctx))
    assert result.candidates and all(not c.backend_details["gepa_search_survived"] for c in result.candidates)
    assert all(c.backend_details["local_acceptance_delta"]==0 for c in result.candidates)
    assert result.strict_accepted_count==0 and result.local_survival_update_count==0
    assert result.strict_rejected_exported_count==1
    assert len(solver.calls)==result.search_state["telemetry"]["local_metric_evaluations"]<=36
    assert not any("decision_procedure" in json.dumps(v) for v in result.search_state["callback_events"])
    assert len(result.candidates)==1 # repeated proposal deduplicated


def test_rejected_raw_prompt_never_persisted(tmp_path):
    f=run_fixture(tmp_path)
    raw=json.dumps(f.result.__dict__,default=lambda x:asdict(x))
    assert B not in raw
    for path in tmp_path.rglob("*"):
        if path.is_file(): assert B.encode() not in path.read_bytes()
    payload=f.engine.bridge.raw_candidates
    assert any(c.prompt==B for c in payload.values()) # in-memory only
    assert f.engine.bridge.optimizer._active_callback is None


@pytest.mark.parametrize("case",["unevaluated","contract_invalid","unchanged","duplicate"])
def test_candidate_export_eligibility_and_unique_prompts(case):
    examples=tuple(LocalEvidenceExample(f"q{i}",f"public case {i}","A") for i in range(3))
    task=LocalOptimizationTask("o",PARENT,examples,examples,"", "solver", "output",81,LocalOptimizerBudget(36))
    cb=EphemeralProposalCallback(parent_prompt=PARENT,examples=examples)
    prompts=[B,B] if case=="duplicate" else ["FINAL_ANSWER: A" if case=="contract_invalid" else PARENT if case=="unchanged" else B]
    for iteration,prompt in enumerate(prompts,1):
        cb.on_proposal_end({"iteration":iteration,"new_instructions":{"decision_procedure":prompt}})
        cb.on_minibatch_sampled({"iteration":iteration,"minibatch_ids":["q0","q1","q2"]})
        cb.on_evaluation_end({"iteration":iteration,"candidate_idx":0,"scores":[0,1,0]})
        if case!="unevaluated":
            cb.on_evaluation_end({"iteration":iteration,"candidate_idx":None,"scores":[0,1,0]})
            cb.observe_evaluation({"candidate_hash":hashlib.sha256(prompt.encode()).hexdigest(),"example_ids":["q0"],"provider_called":[False]})
    exported=cb.export(task,{})
    assert len(exported)==(1 if case=="duplicate" else 0)


@pytest.mark.parametrize("n",[3,5,7,9,12,30])
def test_variable_batch_sizes_and_capacity(n):
    o=opportunity(rows(n))
    assert len(o.evidence.search_validation_evidence)==min(n,12)
    assert len(o.evidence.team_probe_evidence)==min(n,12)
    assert len(o.evidence.mutation_evidence)==min(n,12)
    assert 2*len(o.evidence.search_validation_evidence)+4*3<=36
    assert validation_capacity()==12


def test_only_technical_minimum_backfill_and_no_exact_batch():
    o=opportunity(rows(20,repairs=1))
    assert len(o.evidence.mutation_evidence)==3
    assert len(o.evidence.search_validation_evidence)==1
    assert o.evaluation_plan["evidence_audit"]["generic_backfill_count"]==2
    assert opportunity(rows(2,repairs=1)) is None


@pytest.mark.parametrize("budget,expected",[(36,12),(24,6),(14,1),(12,0)])
def test_budget_capacity_never_overshoots(budget,expected):
    assert validation_capacity(budget)==expected
    if expected: assert 2*expected+4*3<=budget


class PatternProvider:
    def __init__(self, patterns):self.patterns=patterns;self.requests=[]
    def diagnose(self, request): self.requests.append(request);return self.patterns


def pattern_fixture():
    return PatternProvider((PatternHypothesis("P1","Missed relational constraint","Check relational constraints",("e0","e1","e2"),confidence=.8),
        PatternHypothesis("P2","Missed ordering constraint","Check ordering constraints",("e3",),confidence=.9),
        PatternHypothesis("P3","Missed negation scope","Check negation scope",("e4",),confidence=.9)))


def test_pattern_dnc_target_parity_and_single_focus():
    provider=pattern_fixture();analyzer=PatternDiagnosticV1(provider)
    off=opportunity();on=opportunity(patterns=analyzer)
    assert off.target_member==on.target_member and off.objective==on.objective
    assert off.diagnosis.responsibility==on.diagnosis.responsibility
    assert on.pattern_context["dominant_pattern_id"]=="P1"
    assert {r.example_id for r in on.evidence.mutation_evidence}=={"e0","e1","e2"}
    assert on.pattern_context["dominant_pattern_ratio"]==.6
    assert 0<on.pattern_context["normalized_entropy"]<1
    assert not hasattr(provider.requests[0],"diagnostics") and not hasattr(provider.requests[0],"shadow")
    assert all(r.source_split=="optimize" for r in provider.requests[0].evidence_rows)


def test_pattern_outside_universe_and_missing_binding_fail_closed():
    with pytest.raises(SearchContractError,match="PATTERN_PROVIDER_NOT_BOUND"):PatternDiagnosticV1()
    p=PatternProvider((PatternHypothesis("P","Mechanism","Principle",("heldout",)),))
    with pytest.raises(SearchContractError,match="outside"):opportunity(patterns=PatternDiagnosticV1(p))
    assert opportunity(tuple(replace(r,source_split="shadow") for r in rows())) is None


def test_pattern_backfill_technical_only():
    p=PatternProvider((PatternHypothesis("P1","Mechanism","Principle",("e0",)),
                       PatternHypothesis("P2","Other mechanism","Other principle",("e1","e2","e3","e4"))))
    o=opportunity(patterns=PatternDiagnosticV1(p))
    # Dominant P2 supplies its own four supports and does not fill to 12.
    assert len(o.evidence.mutation_evidence)==4
    p=PatternProvider((PatternHypothesis("P1","Mechanism","Principle",("e0",)),))
    o=opportunity(rows(20),PatternDiagnosticV1(p))
    assert len(o.evidence.mutation_evidence)==3 and len(o.evidence.search_validation_evidence)==1
    assert o.evaluation_plan["evidence_audit"]["generic_backfill_count"]==2


def test_pattern_fake_e2e_context_only_at_reflection(tmp_path):
    p=pattern_fixture();f=run_fixture(tmp_path,patterns=PatternDiagnosticV1(p))
    assert len(f.result.transitions)==1
    assert f.result.trace[0].target_member==2 and f.result.trace[0].evidence_audit["pattern_support_count"]==3
    task=f.engine.bridge.make_task(opportunity(patterns=PatternDiagnosticV1(p)),
        UnifiedSearchContext(None,None,HistoryState(),opportunity(patterns=PatternDiagnosticV1(p)).pattern_context,{}))
    assert "Missed relational constraint" in task.optimization_context
    assert all("Missed relational constraint" not in public for _,public in f.solver.requests)
    adapter=V2ReflectionAdapter(Solver(),parent_prompt=PARENT,all_examples=task.search_examples,
        optimization_context=task.optimization_context,output_contract_id="synthetic")
    batch=adapter.evaluate(list(task.search_examples),{"decision_procedure":PARENT},True)
    reflective=adapter.make_reflective_dataset({"decision_procedure":PARENT},batch,["decision_procedure"])
    assert "Missed relational constraint" in json.dumps(reflective)


def test_null_context_and_default_v2_are_zero_optional_effect(tmp_path):
    assert SearchContextComposer().compose("base",{}, {})=="base"
    f=run_fixture(tmp_path)
    assert f.result.trace[0].memory_audit=={} and not f.result.trace[0].evidence_audit["pattern_enabled"]
    assert SearchMethodConfig.v2().pattern_policy=="null_pattern_v1"
    assert SearchMethodConfig.v2().memory_policy=="null_memory_v1"
    assert SearchMethodConfig.from_mapping({"method":"unified_team_prompt_search_v2"})==SearchMethodConfig.v2()
    assert SearchMethodConfig().identity()!=SearchMethodConfig.v2().identity()


@pytest.mark.parametrize("pattern_enabled,memory_enabled",[(False,False),(True,False),(False,True),(True,True)])
def test_memory_and_pattern_independently_toggle(tmp_path,pattern_enabled,memory_enabled):
    p=PatternDiagnosticV1(pattern_fixture()) if pattern_enabled else None
    m=StructuredLongTermMemoryProviderV1(**LIMITS) if memory_enabled else None
    f=run_fixture(tmp_path,patterns=p,memory=m)
    assert len(f.result.transitions)==1
    if m is not None:
        assert len(m.private)==1 and m.private[0].owner_member==2
        assert m.private[0].pattern_id is not None if pattern_enabled else m.private[0].pattern_id is None


def test_memory_private_success_shared_risk_isolation_and_no_raw_leak(tmp_path):
    m=StructuredLongTermMemoryProviderV1(**LIMITS)
    f=run_fixture(tmp_path/"success",memory=m)
    assert len(m.private)==1 and m.private[0].kind=="SUCCESS"
    run_fixture(tmp_path/"risk",memory=m,catastrophe=True)
    assert m.shared and all(e.scope=="SHARED_RISK" for e in m.shared)
    o=opportunity();view=m.read_for_opportunity(o)
    assert len(view["private"])==1 and view["shared"]
    other=replace(o,target_member=1,diagnosis=Diagnosis({1:ResponsibilitySignal(1,2,0,0,"direct_flip")}))
    other_view=m.read_for_opportunity(other)
    assert other_view["private"]==[] and other_view["shared"]==view["shared"]
    assert m.read_for_opportunity(o)==view
    payload=json.dumps([asdict(e) for e in (*m.private,*m.shared)])
    assert PARENT not in payload and A not in payload and B not in payload
    assert "Synthetic public item" not in payload and "synthetic reasoning" not in payload
    context=SearchContextComposer().compose("base",{},view)
    assert "Preserve fixed-peer" in context
    assert all("Preserve fixed-peer" not in raw for _,raw in f.solver.requests)


@pytest.mark.parametrize("failure",["provider_timeout","budget_abort","persistence_failure","invalid_response","incomplete_evaluation"])
def test_operational_failure_writes_no_memory(failure):
    m=StructuredLongTermMemoryProviderV1(**LIMITS)
    delta=m.prepare_outcome(OpportunityOutcome(opportunity(),(),None,False,None,0,complete=False,operational_failure=True))
    m.validate_delta(delta);m.apply_outcome(delta)
    assert not m.private and not m.shared and m.write_count==0


def test_memory_shadow_rejection_and_precommit_failure(tmp_path):
    m=StructuredLongTermMemoryProviderV1(**LIMITS)
    f=run_fixture(tmp_path/"shadow",memory=m,gate_passed=False)
    assert not f.result.transitions and not m.private and any(e.risk_code=="SHADOW_REJECTION" for e in m.shared)
    class BrokenMemory(StructuredLongTermMemoryProviderV1):
        def prepare_outcome(self, outcome):raise ValueError("prepare failure")
    broken=BrokenMemory(**LIMITS)
    with pytest.raises(ValueError,match="prepare failure"):run_fixture(tmp_path/"broken",memory=broken)
    assert not broken.private and broken.write_count==0


def test_memory_limits_enter_method_identity_and_missing_limits_fail_closed():
    with pytest.raises(TypeError):StructuredLongTermMemoryProviderV1()
    a=SearchMethodConfig.v2(memory_policy=versions.UNIFIED_STRUCTURED_MEMORY_VERSION,mechanism_config={"memory":LIMITS})
    b=replace(a,mechanism_config={"memory":{**LIMITS,"top_k_shared":3}})
    assert a.identity()!=b.identity()!=SearchMethodConfig.v2().identity()


def test_v2_current_path_has_no_historical_quota_or_core_fork():
    root=Path(__file__).resolve().parents[1]/"multi_dataset_diverse_rl/search"
    source="\n".join((root/name).read_text(encoding="utf-8") for name in
        ("gepa_v2.py","runtime_v2.py","variable_evidence.py","patterns.py","memory.py"))
    for text in ("len(local_eval) == 12","len(repair) < 4","incomplete REPAIR evidence quota","GEPAEngine.run", "local_acceptance_delta <= 0"):
        assert text not in source


def test_second_opportunity_reads_private_and_shared_memory_in_search_only(tmp_path):
    m=StructuredLongTermMemoryProviderV1(**LIMITS)
    first=run_fixture(tmp_path/"first",memory=m)
    run_fixture(tmp_path/"risk",memory=m,catastrophe=True)
    second=run_fixture(tmp_path/"second",memory=m)
    assert "Preserve fixed-peer team competence" in second.contexts[0]
    assert "Avoid edits that regress" in second.contexts[0]
    assert "Preserve fixed-peer team competence" not in first.contexts[0]
    assert all("Preserve fixed-peer" not in problem for _,problem in second.solver.requests)


@pytest.mark.parametrize("n",[3,5,9,12])
def test_official_gepa_variable_capacity_no_extra_metric_calls(tmp_path,n):
    engine,solver,reflection=make_engine(tmp_path,False)
    opp=opportunity(rows(n));ctx=UnifiedSearchContext(None,None,HistoryState(),{}, {})
    result=asyncio.run(engine.search(opp,ctx))
    assert len(solver.calls)==result.solver_calls==result.search_state["telemetry"]["local_metric_evaluations"]<=36
    assert result.strict_accepted_count==0 and result.strict_rejected_exported_count==1


def test_invalid_proposal_never_calls_solver_with_official_gepa(tmp_path):
    engine,solver,reflection=make_engine(tmp_path,False)
    reflection.answer="FINAL_ANSWER: A"
    result=asyncio.run(engine.search(opportunity(),UnifiedSearchContext(None,None,HistoryState(),{}, {})))
    assert not result.candidates and all(prompt==PARENT for prompt,_ in solver.calls)


def test_accepted_metadata_wins_over_earlier_duplicate_rejection():
    examples=tuple(LocalEvidenceExample(f"q{i}",f"public case {i}","A") for i in range(3))
    task=LocalOptimizationTask("o",PARENT,examples,examples,"", "solver", "output",81,LocalOptimizerBudget(36))
    cb=EphemeralProposalCallback(parent_prompt=PARENT,examples=examples)
    for i in (1,2):
        cb.on_minibatch_sampled({"iteration":i,"minibatch_ids":["q0","q1","q2"]})
        cb.on_proposal_end({"iteration":i,"new_instructions":{"decision_procedure":B}})
        cb.on_evaluation_end({"iteration":i,"candidate_idx":0,"scores":[0,0,0]})
        cb.on_evaluation_end({"iteration":i,"candidate_idx":None,"scores":[1,1,1] if i==2 else [0,0,0]})
    cb.observe_evaluation({"candidate_hash":hashlib.sha256(B.encode()).hexdigest(),"example_ids":["q0"],"provider_called":[False]})
    cb.on_candidate_accepted({"iteration":2,"new_candidate_idx":1,"new_score":3,"parent_ids":[0]})
    c=cb.export(task,{"gepa_result":{"val_aggregate_scores":[0,1],"parents":[[None],[0]]},"local_gepa_frontier_indices":[1]})
    assert len(c)==1 and c[0].backend_metadata["gepa_search_survived"] and c[0].backend_metadata["proposal_iteration"]==2


def test_rejected_exposure_is_not_local_survival_update(tmp_path):
    engine,solver,reflection=make_engine(tmp_path,False)
    result=asyncio.run(engine.search(opportunity(),UnifiedSearchContext(None,None,HistoryState(),{}, {})))
    assert result.candidates and result.local_survival_update_count==0
    stop=GlobalStopPolicy()
    stop.observe_opportunity(parent_state_id="root",eligible_members=(2,3),selected_member=2,
                             local_update=result.local_survival_update_count>0,committed=False)
    assert not stop.any_local_update


def test_v1_method_identity_and_constants_retained():
    import ast, subprocess
    before=ast.parse(subprocess.check_output(["git","show","bad63e9577573a3bea7d004f3a6e333778c967ad:multi_dataset_diverse_rl/versions.py"]).decode())
    for node in before.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):
                    assert getattr(versions,target.id)==ast.literal_eval(node.value)
    payload=asdict(SearchMethodConfig());payload.pop("mechanism_config")
    assert SearchMethodConfig().identity()==hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def test_memory_non_null_manifest_requires_explicit_limits_and_pattern_binding():
    import yaml
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    root=Path(__file__).resolve().parents[1]
    draft=yaml.safe_load((root/"experiments/templates/unified_experiment_v2.yaml").read_text(encoding="utf-8"))
    assert validate_manifest_v2(root,draft)==[]
    draft["memory_identity"]="structured_agent_memory_v1"
    assert validate_manifest_v2(root,draft)
    draft["mechanism_config"]["memory"]=LIMITS
    assert validate_manifest_v2(root,draft)==[]
    draft["pattern_identity"]="pattern_diagnostic_v1"
    assert validate_manifest_v2(root,draft)
    draft["mechanism_config"]["pattern_provider_binding"]="fake_fixture_v1"
    assert validate_manifest_v2(root,draft)==[]


def test_pattern_entropy_extremes_and_risk_boundary():
    equal=PatternDiagnosticV1(PatternProvider(tuple(PatternHypothesis(f"P{i}","Mechanism","Principle",(f"e{i}",)) for i in range(5))))
    o=opportunity(patterns=equal)
    assert o.pattern_context["normalized_entropy"]==pytest.approx(1)
    assert o.pattern_context["dominant_pattern_ratio"]==.2
    one=opportunity(patterns=PatternDiagnosticV1(PatternProvider((PatternHypothesis("P","Mechanism","Principle",("e0","e1")),))))
    assert one.pattern_context["normalized_entropy"]==0 and one.pattern_context["dominant_pattern_ratio"]==1
    for ids in (("validation",),("test",),("shadow",)):
        with pytest.raises(SearchContractError,match="outside"):
            opportunity(patterns=PatternDiagnosticV1(PatternProvider((PatternHypothesis("P","Mechanism","Principle",("e0",),risk_ids=ids),))))


def test_memory_context_never_enters_aggregator_or_gold_evaluator(tmp_path):
    from multi_dataset_diverse_rl.search.aggregation import LLMAggregation,RuntimeModelIdentity,AggregationResponse
    from multi_dataset_diverse_rl.search.benchmark import BenchmarkInput
    from test_unified_search_architecture import FakeBenchmark
    from multi_dataset_diverse_rl.search.evaluation import TeamEvaluator
    m=StructuredLongTermMemoryProviderV1(**LIMITS);f=run_fixture(tmp_path,memory=m)
    view=m.read_for_opportunity(opportunity())
    assert view
    seen=[]
    async def provider(request):seen.append(request);return AggregationResponse("red")
    item=BenchmarkInput("synthetic","public problem","one answer")
    aggregation=LLMAggregation(runtime=RuntimeModelIdentity("fake",81,"greedy"),provider=provider)
    asyncio.run(TeamEvaluator(FakeBenchmark(),aggregation).evaluate_row(item=item,member_outputs=("red",)*5,gold="red"))
    assert len(seen)==1 and "private" not in seen[0].prompt and "Preserve fixed-peer" not in seen[0].prompt


def test_complete_common_safe_rejection_writes_shared_risk_only():
    m=StructuredLongTermMemoryProviderV1(**LIMITS);o=opportunity()
    candidate=SearchCandidate("c",B)
    row=EvaluatedCandidate(candidate,None,TeamEvaluation(0,None,(0,)*5),True,False,
                           {"scientific_risk_code":"COMMON_SAFE_REJECTION"})
    d=m.prepare_outcome(OpportunityOutcome(o,(row,),None,False,None,0))
    m.validate_delta(d);m.apply_outcome(d)
    assert not m.private and len(m.shared)==1 and m.shared[0].risk_code=="COMMON_SAFE_REJECTION"


def test_official_gepa_accepted_lineage_metadata_and_zero_export_cost(tmp_path):
    engine,solver,reflection=make_engine(tmp_path,False)
    reflection.answer=A
    result=asyncio.run(engine.search(opportunity(),UnifiedSearchContext(None,None,HistoryState(),{}, {})))
    assert result.strict_accepted_count==1 and result.local_survival_update_count==1
    assert len(result.candidates)==1
    c=result.candidates[0]
    assert c.backend_details["gepa_frontier_member"] and c.backend_details["gepa_search_survived"]
    assert c.backend_details["local_full_validation_delta"]==1 and c.backend_details["local_acceptance_delta"]==3
    assert result.solver_calls==len(solver.calls)<=36
    assert result.solver_tokens==2*len(solver.calls)
    assert result.search_meta_calls==len(reflection.calls) and result.search_meta_tokens==3*len(reflection.calls)
    assert A not in json.dumps(result.search_state)


def test_accepted_proposals_outside_frontier_still_exported():
    examples=tuple(LocalEvidenceExample(f"q{i}",f"public case {i}","A") for i in range(3))
    task=LocalOptimizationTask("o",PARENT,examples,examples,"", "solver", "output",81,LocalOptimizerBudget(36))
    cb=EphemeralProposalCallback(parent_prompt=PARENT,examples=examples)
    cb.on_minibatch_sampled({"iteration":1,"minibatch_ids":["q0","q1","q2"]})
    cb.on_proposal_end({"iteration":1,"new_instructions":{"decision_procedure":A}})
    cb.on_evaluation_end({"iteration":1,"candidate_idx":0,"scores":[0,0,0]})
    cb.on_evaluation_end({"iteration":1,"candidate_idx":None,"scores":[1,1,1]})
    cb.observe_evaluation({"candidate_hash":hashlib.sha256(A.encode()).hexdigest(),"example_ids":["q0"],"provider_called":[True]})
    cb.on_candidate_accepted({"iteration":1,"new_candidate_idx":1,"new_score":3,"parent_ids":[0]})
    c=cb.export(task,{"gepa_result":{"val_aggregate_scores":[0,1],"parents":[[None],[0]]},"local_gepa_frontier_indices":[0]})
    assert len(c)==1 and c[0].backend_metadata["gepa_search_survived"] and not c[0].backend_metadata["gepa_frontier_member"]


def test_memory_two_successive_opportunities_follow_committed_parent(tmp_path):
    m=StructuredLongTermMemoryProviderV1(**LIMITS)
    f=run_fixture(tmp_path,memory=m,max_opportunities=2)
    assert len(f.result.trace)==2 and len(f.result.transitions)==1
    assert f.result.trace[0].child_state_id==f.result.trace[1].parent_state_id
    assert f.result.trace[1].committed_candidate_id is None
    assert "Preserve fixed-peer team competence" not in f.contexts[0]
    assert "Preserve fixed-peer team competence" in f.contexts[1]
    assert len(m.private)==1 and m.private[0].owner_member==2
    assert any(e.risk_code=="COMMON_SAFE_REJECTION" for e in m.shared)


def test_v2_cli_default_is_hold_without_optional_mechanisms():
    from scripts.run_experiment import preflight
    manifest={"scientific":{"method":"unified_team_prompt_search_v2"},"runtime":{
        "seed":81,"provider_profile":"fake","solver_model":"solver","optimizer_model":"optimizer",
        "evaluator_model":"optimizer","run_identity_sha256":"fake","authorization_identity":"none",
        "cache_identity":"fake","ledger_identity":"fake"}}
    r=preflight(manifest)
    assert r["gate"]=="HOLD" and r["method_identity"]==SearchMethodConfig.v2().identity()
    assert r["provider_attempts"]==r["validation_calls"]==r["test_calls"]==0


def test_memory_third_opportunity_reads_shared_risk_and_saturates(tmp_path):
    m=StructuredLongTermMemoryProviderV1(**LIMITS)
    f=run_fixture(tmp_path,memory=m,max_opportunities=3)
    assert len(f.result.transitions)==1 and len(f.result.trace)==3
    assert "Preserve target, team and valid-output competence" in f.contexts[2]
    assert f.result.stop_reason=="SATURATION_REACHED"
    assert f.orchestrator.stop.no_commit_epochs==2


def test_provider_exception_preserves_team_and_existing_memory(tmp_path):
    m=StructuredLongTermMemoryProviderV1(**LIMITS)
    run_fixture(tmp_path/"prime",memory=m)
    before=(m.private,m.shared,m.revision,m.write_count)
    state=Store();initial=state.snapshot()
    with pytest.raises(RuntimeError,match="fake provider timeout"):
        run_fixture(tmp_path/"abort",memory=m,solver_failure=True,store=state)
    assert state.snapshot()==initial
    assert (m.private,m.shared,m.revision,m.write_count)==before


def test_invalid_solver_response_marks_memory_outcome_inert(tmp_path):
    engine,solver,reflection=make_engine(tmp_path,False)
    def invalid(prompt,row):
        return LocalSolverObservation(None,"invalid fake response",False,False,"invalid_response",provider_called=True)
    solver.evaluate=invalid
    result=asyncio.run(engine.search(opportunity(),UnifiedSearchContext(None,None,HistoryState(),{},{})))
    assert result.search_state["operational_failure"]
    m=StructuredLongTermMemoryProviderV1(**LIMITS)
    row=EvaluatedCandidate(SearchCandidate("c",B),None,None,False,False,{"scientific_risk_code":"TEAM_PROBE_REJECTION"})
    d=m.prepare_outcome(OpportunityOutcome(opportunity(),(row,),None,False,None,0,
                                         operational_failure=result.search_state["operational_failure"]))
    m.validate_delta(d);m.apply_outcome(d)
    assert not m.private and not m.shared and m.revision==0 and m.write_count==0


@pytest.mark.parametrize("valid",[True,False])
def test_shadow_invalid_response_is_an_operational_memory_flag(valid):
    from multi_dataset_diverse_rl.search.current_bbh_runtime import CurrentBBHShadowGate
    from multi_dataset_diverse_rl.team_search.candidate_evaluator import EvaluationCost
    cache=SimpleNamespace(key=lambda p,q:p+":"+q,cache={})
    for p in (PARENT,B):
        cache.cache[hashlib.sha256(p.encode()).hexdigest()+":e0"]=SimpleNamespace(valid=valid if p==B else True)
    evaluator=SimpleNamespace(system=SimpleNamespace(agents=[SimpleNamespace(current_prompt=PARENT)]*5,
        prompt_hash=lambda p:hashlib.sha256(p.encode()).hexdigest()),
        shadow_probe=SimpleNamespace(examples=[SimpleNamespace(question_hash="e0")],prompt_question_evaluator=cache),
        evaluate_shadow=lambda a,c:(SimpleNamespace(passed=False),EvaluationCost(0,0,0)))
    gate=CurrentBBHShadowGate(SimpleNamespace(build=lambda o:None),SimpleNamespace(raw_candidates={"c":SimpleNamespace(prompt=B)}),evaluator)
    row=EvaluatedCandidate(SearchCandidate("c",B),None,TeamEvaluation(0,None,(0,)*5),True,False)
    assert not asyncio.run(gate.check(opportunity(),row))
    assert gate.operational_failure is (not valid)


def test_previous_shadow_failure_does_not_suppress_ungated_complete_outcome(tmp_path):
    memory = StructuredLongTermMemoryProviderV1(**LIMITS)
    gate = Gate()
    gate.operational_failure = True
    fixture = run_fixture(tmp_path, memory=memory, catastrophe=True, gate=gate)
    assert not gate.seen and not fixture.result.transitions
    assert not memory.private
    assert any(entry.risk_code == 'TEAM_PROBE_REJECTION' for entry in memory.shared)
