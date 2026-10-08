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
        if self.optimizer_input_schema not in {versions.GRADIENT_OPTIMIZER_INPUT_VERSION,
                versions.GRADIENT_VISIBLE_OPTIMIZER_INPUT_VERSION}:
            raise SearchContractError('PATTERN_GRADIENT_LAYER1_FROZEN_CONFIG_MISMATCH')
        expected=dict(identity_version=versions.LAYER1_FEEDBACK_SEARCH_VERSION,metric_limit=36,panel_size=6,max_generations=6,k_local_return=4,max_prompt_chars=3000,candidate_contract=versions.SEMANTIC_MUTABLE_CONTRACT_VERSION,official_gepa_fidelity=False,optimizer_input_schema=self.optimizer_input_schema,
            panel_policy=versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION)
        if asdict(self)!=expected:raise SearchContractError('PATTERN_GRADIENT_LAYER1_FROZEN_CONFIG_MISMATCH')

    def identity(self):return digest(asdict(self))

@dataclass(frozen=True)
class GradientPatternLocalTask(MemoryLocalTask):
    selected_pattern: str = ''
    failure_trajectories: str = '[]'
    pattern_support: tuple[str,...] = ()
    private_feedback_texts: tuple[str,...] = ()


@dataclass(frozen=True)
class EvidenceLayer1Config:
    identity_version: str = 'INDEPENDENT_OPTIMIZE_VALIDATION_SEARCH_V1'
    metric_limit: int = 42
    panel_size: int = 6
    max_generations: int = 6
    k_local_return: int = 4
    max_prompt_chars: int = 3000
    candidate_contract: str = versions.SEMANTIC_MUTABLE_CONTRACT_VERSION
    official_gepa_fidelity: bool = False
    optimizer_input_schema: str = 'PATTERN_HYPOTHESIS_EDIT_EFFECT_INPUT_V6'
    panel_policy: str = 'DISJOINT_ROTATING_OPTIMIZE_EVIDENCE_V1'

    def __post_init__(self):
        defaults={name:field.default for name,field in self.__dataclass_fields__.items()}
        if asdict(self)!=defaults:raise SearchContractError('EVIDENCE_LAYER1_CONFIG_MISMATCH')

    def identity(self):return digest(asdict(self))

def gradient_pattern_input(task,parent,observations,memory):
    from .optimization_evidence import INPUT
    if observations and observations[0].get('evidence_revision')==INPUT:
        value=build_optimizer_context(task,parent,observations,memory)
        value.update(schema=INPUT,selected_pattern=json.loads(task.selected_pattern),
            repair_objective='Test the selected repair hypothesis with the smallest safe edit.')
        return ('''Use only the supplied mutation examples and bounded measured Memory.
The selected Pattern is a repair hypothesis, not a mandatory universal instruction.
Preserve existing successful behavior. Do not introduce dependencies on optimizer-only
evidence. Prefer a small conditional generic edit when supported; otherwise abstain.
Return {"decision":"NO_SAFE_EDIT"} or {"decision":"PROPOSE_EDIT",
"decision_procedure":"complete standalone procedure, at most 3000 characters",
"change_summary":"intended change, at most 240 characters"}. No other keys.
SearchValidation is measured after generation; prior measurement counts may guide
the next iteration but validation examples and answers are not supplied.
Respect missing/truncated written evidence; do not infer hidden reasoning.
''' + json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True))
    value=build_optimizer_context(task,parent,observations,memory)
    selected=json.loads(task.selected_pattern)
    if set(selected)!={'pattern_id','generalized_gradient','responsibility_value'}:
        raise SearchContractError('PATTERN_GRADIENT_OPTIMIZER_CONTEXT_INVALID')
    trajectories=json.loads(task.failure_trajectories)
    if not 1<=len(trajectories)<=3:raise SearchContractError('PATTERN_GRADIENT_REPRESENTATIVE_CAPACITY')
    if value['schema'] == versions.GRADIENT_VISIBLE_OPTIMIZER_INPUT_VERSION:
        from ..benchmarks.math_visible_trajectory import validate_adaptive_trajectory
        for trajectory in trajectories:
            visible = trajectory.get('solver_trajectory')
            if visible is None:
                raise SearchContractError('LAYER1_VISIBLE_TRAJECTORY_MISSING')
            validate_adaptive_trajectory(visible, example_id=visible['source']['example_id'],
                member_id=task.target_member, prompt=task.parent_prompt, problem=trajectory['problem'])
    value.update(selected_pattern=selected,
        repair_objective='Only the selected generalized corrective gradient',
        representative_failure_trajectories=trajectories)
    visible_instruction = ('\nUse the actual written Solver solution in solver_trajectory as observable evidence. '
        'Respect missing, invalid-boundary and truncation metadata; do not infer omitted or hidden steps. '
        'Do not copy problem-specific solution text into the procedure or Memory.'
        if value['schema'] == versions.GRADIENT_VISIBLE_OPTIMIZER_INPUT_VERSION else '')
    return INSTRUCTION+visible_instruction+'\n'+json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True)

class GradientPatternMemoryOptimizer(BoundedMemoryOptimizer):
    config_factory=GradientPatternLayer1Config
    prompt_builder=staticmethod(gradient_pattern_input)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if getattr(getattr(self.evaluator, 'broker', None), 'solver_trajectory_policy', None) is not None:
            self.config = GradientPatternLayer1Config(
                optimizer_input_schema=versions.GRADIENT_VISIBLE_OPTIMIZER_INPUT_VERSION)
        if getattr(getattr(self.evaluator,'broker',None),'contract',{}).get('optimization_evidence_policy'):
            from .optimization_evidence import frozen_policy
            frozen_policy(self.evaluator.broker.contract['optimization_evidence_policy'])
            self.config=EvidenceLayer1Config()

class GradientPatternMemoryEngine(LocalTaskEngine):
    def __init__(self,optimizer,seed):
        super().__init__(optimizer,seed)
        self.identity=optimizer.config.identity_version
    async def search(self, opportunity, context):
        result = await super().search(opportunity, context)
        # Execution metadata only; no change to the provider-visible prompt.
        return replace(result, candidates=tuple(replace(c, backend_details={**c.backend_details,
            'parent_state_id':opportunity.parent_state_id}) for c in result.candidates))

    def make_task(self,opportunity,context):
        base=super().make_task(opportunity,context)
        universe=opportunity.evaluation_plan['evidence_universe']
        if opportunity.evaluation_plan.get('optimization_evidence_policy'):
            if any(r.source_split!='optimize' for r in universe):raise SearchContractError('LAYER1_HELDOUT_ACCESS')
            p=next((p for p in context.pattern_view['patterns']
                if p['pattern_id']==context.pattern_view['focus_mechanism_id']),None)
            lane=opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
            repairs=[r for r in opportunity.evidence.mutation_evidence if 'REPAIR' in r.roles]
            return GradientPatternLocalTask(**{**base.__dict__,'optimization_context':lane},
                responsibility_lane=lane,selected_pattern=json.dumps(self.pattern_projection(p) if p
                    else dict(pattern_id=None,generalized_gradient=None)),
                pattern_support=tuple(p['support_ids']) if p else (),
                failure_trajectories=json.dumps([r.signals['failure_trajectory'] for r in repairs]),
                private_feedback_texts=tuple(str(r.signals.get(k,'')) for r in universe
                    for k in ('reference_solution',)) + tuple(r.signals.get('solver_trajectory',{}).get('visible_solution','') for r in universe))
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
