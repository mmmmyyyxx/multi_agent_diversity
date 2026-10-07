"""Current Gradient Layer1; no historical treatment inheritance."""
from dataclasses import asdict, dataclass, replace
import json
from .. import current_contract as versions
from ..local_optimizers.schemas import LocalEvidenceExample
from .bounded_layer1 import BoundedMemoryOptimizer, LocalTaskEngine, MemoryLocalTask, INSTRUCTION, build_optimizer_context, digest
from .schemas import SearchContractError

@dataclass(frozen=True)
class GradientPatternLayer1Config:
    identity_version: str = versions.LAYER1_FEEDBACK_SEARCH_VERSION
    metric_limit: int = 36
    panel_size: int = 6
    max_generations: int = 6
    k_local_return: int = 4
    max_prompt_chars: int = 3000
    candidate_contract: str = versions.SEMANTIC_MUTABLE_CONTRACT_VERSION
    official_gepa_fidelity: bool = False
    optimizer_input_schema: str = versions.GRADIENT_OPTIMIZER_INPUT_VERSION
    panel_policy: str = versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION

    def __post_init__(self):
        expected=dict(identity_version=versions.LAYER1_FEEDBACK_SEARCH_VERSION,metric_limit=36,panel_size=6,max_generations=6,k_local_return=4,max_prompt_chars=3000,candidate_contract=versions.SEMANTIC_MUTABLE_CONTRACT_VERSION,official_gepa_fidelity=False,optimizer_input_schema=versions.GRADIENT_OPTIMIZER_INPUT_VERSION,
            panel_policy=versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION)
        if asdict(self)!=expected:raise SearchContractError('PATTERN_GRADIENT_LAYER1_FROZEN_CONFIG_MISMATCH')

    def identity(self):return digest(asdict(self))

@dataclass(frozen=True)
class GradientPatternLocalTask(MemoryLocalTask):
    selected_pattern: str = ''
    failure_trajectories: str = '[]'

def gradient_pattern_input(task,parent,observations,memory):
    value=build_optimizer_context(task,parent,observations,memory)
    selected=json.loads(task.selected_pattern)
    if set(selected)!={'pattern_id','generalized_gradient','responsibility_value'}:
        raise SearchContractError('PATTERN_GRADIENT_OPTIMIZER_CONTEXT_INVALID')
    trajectories=json.loads(task.failure_trajectories)
    if not 1<=len(trajectories)<=3:raise SearchContractError('PATTERN_GRADIENT_REPRESENTATIVE_CAPACITY')
    value.update(schema=versions.GRADIENT_OPTIMIZER_INPUT_VERSION,selected_pattern=selected,
        repair_objective='Only the selected generalized corrective gradient',
        representative_failure_trajectories=trajectories)
    return INSTRUCTION+'\n'+json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True)

class GradientPatternMemoryOptimizer(BoundedMemoryOptimizer):
    config_factory=GradientPatternLayer1Config
    prompt_builder=staticmethod(gradient_pattern_input)

class GradientPatternMemoryEngine(LocalTaskEngine):
    async def search(self, opportunity, context):
        result = await super().search(opportunity, context)
        # Execution metadata only; no change to the provider-visible prompt.
        return replace(result, candidates=tuple(replace(c, backend_details={**c.backend_details,
            'parent_state_id':opportunity.parent_state_id}) for c in result.candidates))

    def make_task(self,opportunity,context):
        base=super().make_task(opportunity,context)
        universe=opportunity.evaluation_plan['evidence_universe']
        if any(r.source_split!='optimize' for r in universe):raise SearchContractError('LAYER1_HELDOUT_ACCESS')
        legal=[r for r in universe if r.signals.get('target_member_correct') is True]
        anchor=min(legal,key=lambda r:(not bool(r.signals.get('mutation_sensitive')),
            -r.signals.get('team_disagreement',0),r.signals.get('team_margin',0),r.example_id)) if legal else None
        local=None
        if anchor:
            s=anchor.signals
            local=next((r for r in (*base.search_examples,*base.local_validation_examples) if r.example_id==anchor.example_id),None)
            if local is None:
                local=LocalEvidenceExample(anchor.example_id,s['input_payload'],s['gold'],s.get('target_output'),
                    'Preserve correct reasoning; no additional repair objective.',('preservation','target_correct','preservation_anchor'))
        p=next(p for p in context.pattern_view['patterns'] if p['pattern_id']==context.pattern_view['focus_mechanism_id'])
        visible=self.pattern_projection(p)
        visible['responsibility_value']=p['responsibility']['raw_value']
        lane=opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
        task=GradientPatternLocalTask(**{**base.__dict__,'optimization_context':lane},anchor_example=local,
            responsibility_lane=lane,selected_pattern=json.dumps(visible,sort_keys=True,separators=(',',':')))

        repairs=[r for r in opportunity.evidence.mutation_evidence if 'REPAIR' in r.roles]
        if not 1<=len(repairs)<=3 or any(r.example_id not in p['support_ids'] for r in repairs):
            raise SearchContractError('PATTERN_GRADIENT_REPRESENTATIVE_CAPACITY')
        trajectories=[r.signals['failure_trajectory'] for r in repairs]
        return replace(task,failure_trajectories=json.dumps(trajectories,sort_keys=True,separators=(',',':')))

    @staticmethod
    def pattern_projection(pattern):
        return {k:pattern[k] for k in ('pattern_id','generalized_gradient')}

CurrentOptimizer=GradientPatternMemoryOptimizer
CurrentEngine=GradientPatternMemoryEngine
CurrentLayer1Config=GradientPatternLayer1Config
