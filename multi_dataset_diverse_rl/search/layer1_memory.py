"""Current evidence and bounded action memory; complete lineage stays private."""
from dataclasses import asdict, dataclass
import hashlib
import json

from .. import versions
from ..local_optimizers.schemas import LocalOptimizationTask, LocalEvidenceExample
from .layer1_responsibility import (Layer1Config, ResponsibilityConditionedOptimizer,
    ResponsibilityConditionedEngine, evaluation_panel)
from .action_memory import edit_action
from .semantic_contract import FocusedEvidencePolicyV2
from .schemas import SearchContractError
from ..evaluation.semantic_mutable_contract import candidate_failed_checks, mutation_shape, normalized_procedure
from ..persistence.durable_io import append_jsonl
from .layer1_responsibility import observation_record
from .schemas import SearchCandidate, SearchResult


@dataclass(frozen=True)
class MemoryLayer1Config(Layer1Config):
    identity_version: str = versions.LAYER1_FEEDBACK_SEARCH_VERSION
    optimizer_input_schema: str = versions.LAYER1_INPUT_SCHEMA_VERSION
    panel_policy: str = versions.LAYER1_ANCHOR_EVIDENCE_VERSION

    def __post_init__(self):
        expected=dict(identity_version=versions.LAYER1_FEEDBACK_SEARCH_VERSION,
            metric_limit=36,panel_size=6,max_generations=6,k_local_return=4,max_prompt_chars=3000,
            candidate_contract=versions.SEMANTIC_MUTABLE_CONTRACT_VERSION,official_gepa_fidelity=False,
            optimizer_input_schema=versions.LAYER1_INPUT_SCHEMA_VERSION,
            panel_policy=versions.LAYER1_ANCHOR_EVIDENCE_VERSION)
        if asdict(self)!=expected:raise SearchContractError('MEMORY_LAYER1_FROZEN_CONFIG_MISMATCH')


class PreservationAnchorEvidenceV3(FocusedEvidencePolicyV2):
    # Selection and WHO remain V2. The new backend panel adds one legal anchor.
    identity=versions.LAYER1_ANCHOR_EVIDENCE_VERSION


@dataclass(frozen=True)
class MemoryLocalTask(LocalOptimizationTask):
    anchor_example: LocalEvidenceExample | None = None
    responsibility_lane: str = 'general'


INSTRUCTION='''Edit the current generic mathematical decision procedure using the current
Optimize evidence and one primary repair objective. The evidence is primary; bounded experience
is auxiliary. Preservation anchors constrain collateral risk. Diagnose observed failures and
avoid repeating edits with measured losses. Exact forms, domain/unit checks, substitution,
verification and meaningful append-only changes are allowed. Do not copy supplied examples or
answers. Do not alter external output markers, line placement, response presentation or schemas.
Return one JSON object with decision_procedure (complete procedure, at most 3000 characters)
and change_summary (generic description relative to current parent, at most 240 characters).
Summary metadata is not part of the Solver procedure. Propose a changed procedure.'''


def bounded_input(task,parent,observations,memory):
    if len(json.dumps(memory,sort_keys=True,separators=(',',':'),ensure_ascii=True))>1200:
        raise SearchContractError('MEMORY_CONTEXT_LIMIT_EXCEEDED')
    # No example identifiers, candidate archive, generation or controller metadata.
    current=[{k:r[k] for k in ('problem','reference','prediction','correct','valid','invalid_reason','role_and_lane','evaluator_feedback')}
             for r in observations]
    return INSTRUCTION+'\n'+json.dumps(dict(schema=versions.LAYER1_INPUT_SCHEMA_VERSION,
        current_parent=parent,repair_objective=task.responsibility_lane+' failures',
        current_panel_observations=current,retrieved_memory=memory),
        sort_keys=True,separators=(',',':'),ensure_ascii=True)


def anchored_panel(task,size=6):
    panel=list(evaluation_panel(task,size))
    anchor=task.anchor_example
    if anchor is not None and all(r.example_id!=anchor.example_id for r in panel):
        if len(panel)==size:panel[-1]=anchor
        else:panel.append(anchor)
    return tuple(panel)


class MemoryConditionedEngine(ResponsibilityConditionedEngine):
    identity=versions.LAYER1_FEEDBACK_SEARCH_VERSION

    def make_task(self,opportunity,context):
        base=super().make_task(opportunity,context)
        universe=opportunity.evaluation_plan['evidence_universe']
        if any(r.source_split!='optimize' for r in universe):raise SearchContractError('LAYER1_HELDOUT_ACCESS')
        legal=[r for r in universe if {'PRESERVATION','target_correct'}<=r.roles]
        anchor=min(legal,key=lambda r:(not bool(r.signals.get('mutation_sensitive')),
            -r.signals.get('team_disagreement',0),r.signals.get('team_margin',0),r.example_id)) if legal else None
        local=None
        if anchor is not None:
            s=anchor.signals
            local=LocalEvidenceExample(anchor.example_id,s['input_payload'],s['gold'],s.get('target_output'),
                s.get('feedback'),tuple(s.get('legacy_tags',()))+tuple(sorted(anchor.roles))+('preservation_anchor',))
            # Match payload/tags for overlaps so LocalOptimizationTask validation remains exact.
            existing=next((r for r in (*base.search_examples,*base.local_validation_examples) if r.example_id==anchor.example_id),None)
            if existing is not None:local=existing
        lane=opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
        return MemoryLocalTask(**{**base.__dict__,'optimization_context':lane},anchor_example=local,responsibility_lane=lane)


class MemoryConditionedOptimizer(ResponsibilityConditionedOptimizer):
    def __init__(self, **kwargs):
        super().__init__(config=MemoryLayer1Config(),**kwargs)
        self.memory=None

    # search_task is the independently versioned V2 implementation below.
    async def search_task(self,task,opportunity_id):
        if (task.backend_state is not None or task.target_member not in range(5)
                or task.budget.max_metric_calls!=self.config.metric_limit
                or task.budget.max_returned_candidates!=self.config.k_local_return):
            raise SearchContractError('LAYER1_TASK_CONTRACT_MISMATCH')
        if (self.evaluator.solver_contract_id!=task.solver_contract_id or
                self.evaluator.output_contract_id!=task.output_contract_id):
            raise SearchContractError('LAYER1_SOLVER_PORT_MISMATCH')
        if self.memory is None:raise SearchContractError('ACTION_MEMORY_NOT_BOUND')
        panel=anchored_panel(task,self.config.panel_size)
        if not panel:raise SearchContractError('LAYER1_NO_LEGAL_PANEL')
        lineage=self.run_root/(task.task_id+'.lineage.jsonl')
        if lineage.exists():raise SearchContractError('LAYER1_FRESH_STATE_REQUIRED')
        all_examples=(*task.search_examples,*task.local_validation_examples,*((task.anchor_example,) if task.anchor_example else ()))
        if candidate_failed_checks(task.parent_prompt,parent_prompt='__root_contract_check__',examples=all_examples):
            raise SearchContractError('LAYER1_PARENT_CONTRACT_VIOLATION')
        before=dict(self.accounting_reader());metric=solver_calls=solver_tokens=0
        events=[];pool=[];seen={normalized_procedure(task.parent_prompt)}
        def evaluate(prompt):
            nonlocal metric,solver_calls,solver_tokens
            if metric+len(panel)>task.budget.max_metric_calls:
                raise SearchContractError('LAYER1_METRIC_LIMIT_PRE_SOLVER')
            metric+=len(panel)
            obs=[self.evaluator.evaluate(prompt,row) for row in panel]
            solver_calls+=sum(o.provider_called for o in obs)
            solver_tokens+=sum(o.input_tokens+o.output_tokens for o in obs)
            return [observation_record(row,o) for row,o in zip(panel,obs,strict=True)]
        root_records=evaluate(task.parent_prompt)
        root_bits=[int(r['correct']) for r in root_records];root_score=sum(root_bits)
        selected_parent=task.parent_prompt;best_score=root_score;best_bits=root_bits;current_records=root_records
        anchor_count=sum(r['correct'] and task.anchor_example is not None and row.example_id==task.anchor_example.example_id for row,r in zip(panel,root_records,strict=True))
        if task.anchor_example is not None and not anchor_count:raise SearchContractError('PRESERVATION_ANCHOR_REALIZATION_MISMATCH')
        measured=root_score>0
        input_stats=[]
        parent_hash=hashlib.sha256(task.parent_prompt.encode()).hexdigest()
        stop='LAYER1_GENERATION_BOUND_REACHED'
        for generation in range(1,self.config.max_generations+1):
            if metric+len(panel)>task.budget.max_metric_calls:
                stop='LAYER1_METRIC_BOUND_REACHED';break
            memory=self.memory.read_for_member(task.target_member,task.responsibility_lane)
            prompt=bounded_input(task,selected_parent,current_records,memory)
            generation_before=dict(self.accounting_reader())
            packet_hash=hashlib.sha256(prompt.encode()).hexdigest()
            raw=self.reflection_lm(prompt)
            generation_after=dict(self.accounting_reader())
            input_stats.append(dict(generation=generation,input_chars=len(prompt),memory_chars=len(json.dumps(memory,sort_keys=True,separators=(',',':'),ensure_ascii=True)),input_tokens=generation_after.get('input_tokens',0)-generation_before.get('input_tokens',0),memory_entries=sum(len(x) for x in memory.values())))
            try:
                obj=json.loads(raw)
                if not isinstance(obj,dict) or 'decision_procedure' not in obj or set(obj)-{'decision_procedure','change_summary'} or not isinstance(obj['decision_procedure'],str):
                    raise ValueError('envelope')
                proposed=obj['decision_procedure']
                summary=obj.get('change_summary')
                summary_status='WELL_FORMED_METADATA_DETERMINISTIC_ACTION_USED' if isinstance(summary,str) and 0<len(summary)<=240 else 'MISSING_OR_MALFORMED_DETERMINISTIC_FALLBACK'
            except (ValueError,TypeError):
                row=dict(generation=generation,status='CONTRACT_INVALID',failed_checks=['invalid_structure'],
                    evidence_packet_hash=packet_hash,source_proposal_call=generation)
                events.append(row);append_jsonl(lineage,row);continue
            h=hashlib.sha256(proposed.encode()).hexdigest()
            checks=candidate_failed_checks(proposed,parent_prompt=task.parent_prompt,
                examples=all_examples,max_chars=self.config.max_prompt_chars)
            norm=normalized_procedure(proposed)
            if norm in seen and not checks:checks=('duplicate_proposal',)
            seen.add(norm)
            current_parent_hash=hashlib.sha256(selected_parent.encode()).hexdigest()
            common=dict(candidate_id=f'layer1:{generation}:{h[:12]}',prompt_hash=h,parent_hash=current_parent_hash,
                root_parent_hash=parent_hash,target_member=task.target_member,generation=generation,
                source_proposal_call=generation,evidence_packet_hash=packet_hash,
                mutation_shape=mutation_shape(proposed,task.parent_prompt),change_summary_status=summary_status)
            if checks:
                row=dict(common,status='CONTRACT_INVALID',failed_checks=list(checks),solver_evaluated=False)
                events.append(row);append_jsonl(lineage,row);continue
            observed=evaluate(proposed);bits=[int(r['correct']) for r in observed]
            fixed=sum(not a and b for a,b in zip(root_bits,bits,strict=True))
            broken=sum(a and not b for a,b in zip(root_bits,bits,strict=True))
            score=sum(bits);delta=score-root_score;parent_delta=score-best_score;positive=parent_delta>0
            details=dict(common,local_examples_evaluated=[r.example_id for r in panel],
                local_correct_count=score,local_correct_delta=delta,local_newly_fixed=fixed,
                local_root_correct_count=root_score,local_parent_correct_count=best_score,
                local_parent_correct_delta=parent_delta,
                local_parent_newly_fixed=sum(not a and b for a,b in zip(best_bits,bits,strict=True)),
                local_parent_newly_broken=sum(a and not b for a,b in zip(best_bits,bits,strict=True)),
                local_newly_broken=broken,local_invalid_count=sum(not r['valid'] for r in observed),
                local_search_status='POSITIVE' if positive else 'NEGATIVE' if parent_delta<0 else 'NEUTRAL',
                locally_positive=positive,locally_rejected=not positive,local_acceptance_delta=parent_delta,
                local_effect_scope='fixed_optimize_panel',local_preservation_loss=broken if measured else None,
                preservation_locally_measurable=measured,panel_parent_correct_anchor_count=anchor_count,
                memory_action=edit_action(selected_parent,proposed),
                changed=True,contract_valid=True,solver_evaluated=True,duplicate=False,
                opportunity_id=opportunity_id,official_gepa_frontier=False,
                export_reason='eligible_all_admissible_scored_pool',exported=False)
            candidate=SearchCandidate(common['candidate_id'],proposed,float(score),{},details)
            pool.append(candidate)
            self.memory.observe_local_failure(member=task.target_member,lane=task.responsibility_lane,
                parent=selected_parent,prompt=proposed,details=details,opportunity_id=opportunity_id)
            row=dict(details,status='LOCALLY_EVALUATED');events.append(row);append_jsonl(lineage,row)
            if score>best_score:selected_parent=proposed;best_score=score;best_bits=bits;current_records=observed
        ranked=sorted(pool,key=lambda c:(-c.backend_details['local_newly_fixed'],
            c.backend_details['local_newly_broken'],-c.backend_details['local_correct_count'],
            c.backend_details['generation']))
        chosen=ranked[:self.config.k_local_return]
        candidates=tuple(SearchCandidate(c.candidate_id,c.prompt,c.search_score,c.lineage,
            {**c.backend_details,'exported':True,'export_reason':'bounded_rank_after_all_scored_pool',
             'LOCAL_REJECTED_EXPORTED':c.backend_details['locally_rejected']}) for c in chosen)
        exported={c.candidate_id for c in candidates}
        outcomes=[{**r,'exported':r.get('candidate_id') in exported} for r in events]
        after=dict(self.accounting_reader())
        meta_tokens=sum(after.get(k,0)-before.get(k,0) for k in ('input_tokens','output_tokens'))
        meta_calls=after.get('successful_calls',0)-before.get('successful_calls',0)
        telemetry=dict(proposal_count=len(events),physical_proposal_generations=meta_calls,
            unique_candidates=sum(r.get('solver_evaluated',False) for r in events),
            contract_rejects=sum(r['status']=='CONTRACT_INVALID' for r in events),
            local_solver_reached=len(pool),local_positive=sum(c.backend_details['locally_positive'] for c in pool),
            local_negative=sum(c.backend_details['local_parent_correct_delta']<0 for c in pool),
            local_neutral=sum(c.backend_details['local_parent_correct_delta']==0 for c in pool),
            team_candidate_count=len(candidates),strict_rejected_exported_count=sum(c.backend_details['locally_rejected'] for c in candidates),
            local_metric_evaluations=metric,metric_limit=task.budget.max_metric_calls,
            panel_size=len(panel),panel_repair_count=sum('repair' in r.tags or 'focus_repair_v2' in r.tags for r in panel),
            panel_anchor_available=task.anchor_example is not None,panel_parent_correct_anchor_count=anchor_count,
            root_correct_count=root_score,preservation_locally_measurable=measured,
            candidate_archive_in_provider_input=False,input_by_generation=input_stats,memory=self.memory.audit())
        state=dict(protocol_hash=self.config.identity(),backend=versions.LAYER1_FEEDBACK_SEARCH_VERSION,
            official_gepa_fidelity=False,telemetry=telemetry,proposal_diagnostics={'proposal_outcomes':outcomes},
            callback_events=outcomes,operational_failure=False,
            token_accounting={'solver_tokens':solver_tokens,'search_meta_tokens':meta_tokens})
        return SearchResult(candidates,stop,state,solver_calls,meta_calls,solver_tokens,meta_tokens,
            len(events),len(candidates),0,0,telemetry['strict_rejected_exported_count'])
