"""Current Gradient Layer1; no historical treatment inheritance."""
from dataclasses import asdict, dataclass, replace
import json
from .. import current_contract as versions
from ..local_optimizers.schemas import LocalEvidenceExample
from .bounded_layer1 import BoundedMemoryOptimizer, LocalTaskEngine, MemoryLocalTask, build_optimizer_context, digest
from .schemas import SearchContractError


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
    if not observations or observations[0].get('evidence_revision')!=INPUT:
        raise SearchContractError('CURRENT_LAYER1_EVIDENCE_REQUIRED')
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

class GradientPatternMemoryOptimizer(BoundedMemoryOptimizer):
    config_factory=EvidenceLayer1Config
    prompt_builder=staticmethod(gradient_pattern_input)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .optimization_evidence import frozen_policy
        broker=getattr(self.evaluator,'broker',None)
        frozen_policy(getattr(broker,'contract',{}).get('optimization_evidence_policy'))
        if getattr(broker,'solver_trajectory_policy',None) is None:
            raise SearchContractError('CURRENT_VISIBLE_TRAJECTORY_REQUIRED')

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
        from .optimization_evidence import frozen_policy
        frozen_policy(opportunity.evaluation_plan.get('optimization_evidence_policy'))
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

    @staticmethod
    def pattern_projection(pattern):
        return {k:pattern[k] for k in ('pattern_id','generalized_gradient')}

CurrentOptimizer=GradientPatternMemoryOptimizer
CurrentEngine=GradientPatternMemoryEngine
CurrentLayer1Config=EvidenceLayer1Config
