"""Current Gradient Layer1; no historical treatment inheritance."""
from dataclasses import asdict, dataclass, replace
import json
from .. import current_contract as versions
from ..local_optimizers.schemas import LocalEvidenceExample
from .bounded_layer1 import BoundedMemoryOptimizer, LocalTaskEngine, MemoryLocalTask, build_optimizer_context, digest
from .schemas import SearchContractError
from .readable_evidence import INSTRUCTION
from .optimizer_output_contract import JSON_INSTRUCTION


@dataclass(frozen=True)
class GradientPatternLocalTask(MemoryLocalTask):
    selected_pattern: str = ''
    failure_trajectories: str = '[]'
    pattern_support: tuple[str,...] = ()
    private_feedback_texts: tuple[str,...] = ()
    repair_evidence: str = '[]'
    task_types: tuple[str,...] = ()
    parent_state_id: str = ''


@dataclass(frozen=True)
class EvidenceLayer1Config:
    identity_version: str = versions.CURRENT_LAYER1_SEARCH_VERSION
    metric_limit: int = 42
    panel_size: int = 6
    max_generations: int = 6
    k_local_return: int = 4
    max_prompt_chars: int = 3000
    candidate_contract: str = versions.SEMANTIC_MUTABLE_CONTRACT_VERSION
    official_gepa_fidelity: bool = False
    optimizer_input_schema: str = versions.CURRENT_OPTIMIZER_INPUT_SCHEMA
    panel_policy: str = 'DISJOINT_QUOTA_FREE_REPAIR_EVIDENCE_V2'

    def __post_init__(self):
        defaults={name:field.default for name,field in self.__dataclass_fields__.items()}
        if asdict(self)!=defaults:raise SearchContractError('EVIDENCE_LAYER1_CONFIG_MISMATCH')

    def identity(self):return digest(asdict(self))

def gradient_pattern_input(task,parent,observations,memory,attempt_feedback=()):
    from .optimization_evidence import INPUT
    if not observations or observations[0].get('evidence_revision')!=INPUT:
        raise SearchContractError('CURRENT_LAYER1_EVIDENCE_REQUIRED')
    value=build_optimizer_context(task,parent,observations,memory)
    value.update(schema=INPUT,selected_pattern=json.loads(task.selected_pattern),
        repair_objective='Develop a testable conditional repair for useful team improvement.',
        preceding_attempts=list(attempt_feedback),
        edit_capacity=dict(rendered_characters=len(parent.render()),maximum_characters=3000,
            maximum_replacement_block_characters={k:3000-4-sum(len(getattr(parent,j))
                for j in ('role','strategy','answer') if j!=k) for k in ('role','strategy','answer')}))
    sources={r['example_id']:r for r in json.loads(task.repair_evidence)}
    if set(sources)!={r['example_id'] for r in observations}:
        raise SearchContractError('LAYER1_REPAIR_EVIDENCE_MEMBERSHIP_MISMATCH')
    for obs,visible in zip(observations,value['current_panel_observations'],strict=True):
        source=sources.get(obs['example_id'])
        if source:
            visible['worked_reference']=source['worked_reference']
            if source.get('diagnostic'):
                visible['root_repair_diagnostic']=source['diagnostic']
    return (INSTRUCTION + '''Use only the supplied mutation examples and bounded measured Memory.
The selected Pattern is a repair hypothesis, not a mandatory universal instruction.
Do not introduce dependencies on optimizer-only evidence. Use the observed failure,
diagnosis, reusable correction, expected effect and worked reference to judge the
selected hypothesis. Root diagnostics describe the opportunity's ORIGINAL prompt;
current written observations describe the current local parent. A diagnosis is not
a proven cause, and previously fixed cases need not still require that repair.
Memory opportunity_root_match compares earlier parent/child prompts and team to
the ORIGINAL opportunity root, not the later local parent. last_step_instruction
is only the final step in a possibly compound edit, not its isolated causal effect.
Read scoped Memory as earlier complete-edit observations. Relate its actual edit,
measured scope and parent applicability to this task; do not copy a past failure or
treat either success or rejection as a universal rule. Existing successful behavior
matters, but a small local loss is not a final team rejection. Full/Shadow decide
team admission. Prefer one specific, conditional repair over broad new checklists.
Appending the old block is not behavioral preservation: revise overlapping rules
concisely when warranted. Respect the supplied exact replacement-block capacity.
Use preceding attempts to avoid repeating unchanged abstentions or invalid edits;
consider a distinct grounded hypothesis when evidence supports one. NO_SAFE_EDIT
remains valid when no justified contract-legal edit can be proposed; do not invent
a repair merely to fill a generation slot.
Return {"decision":"NO_SAFE_EDIT"} or {"decision":"PROPOSE_EDIT",
"target_block":"strategy", "new_content":"complete replacement block",
"change_summary":"intended change, at most 240 characters"}. No other keys.
Choose the block from the evidence.
target_block must be ONE string: "role", "strategy", or "answer"; never an object,
array, or comma-separated list. decision and new_content are strings, and
change_summary is a nonempty string of at most 240 decoded characters.
The example shows shape only; choose the appropriate block independently.
All blocks, including presentation and reasoning preferences, are editable.
Change exactly one block. The full rendered system
prompt must remain at most 3000 characters. No fixed CoT or output format is required.
SearchValidation is measured after generation; prior measurement counts may guide
the next iteration but validation examples and answers are not supplied.
Respect missing/truncated written evidence; do not infer hidden reasoning.
''' + JSON_INSTRUCTION + json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True))

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
        lane=('general' if context.pattern_view.get('selection_reason')=='SEEDED_ZERO_RESPONSIBILITY_FALLBACK' else opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane)
        repairs=[r for r in opportunity.evidence.mutation_evidence if 'REPAIR' in r.roles]
        diagnostics={r['example_id']:r for r in context.pattern_view.get('gradient_diagnostics',())
            if r.get('disposition')=='ACTIONABLE'}
        repair_evidence=[]
        for row in opportunity.evidence.mutation_evidence:
            solution=row.signals.get('reference_solution')
            if not isinstance(solution,str) or not solution.strip():raise SearchContractError('OPTIMIZE_REFERENCE_SOLUTION_REQUIRED')
            diagnostic=diagnostics.get(row.example_id)
            projection=({k:diagnostic[k] for k in ('observed_failure','diagnosis','reusable_correction',
                'suggested_block','expected_effect')} if diagnostic and p and row.example_id in p['support_ids'] else None)
            if projection:projection['basis']='original_opportunity_prompt_observation_hypothesis'
            repair_evidence.append(dict(example_id=row.example_id,diagnostic=projection,
                worked_reference=dict(source='dataset_worked_solution',split='optimize',text=solution[:4096],
                    original_characters=len(solution),truncated=len(solution)>4096)))
        return GradientPatternLocalTask(**{**base.__dict__,'optimization_context':lane},
            responsibility_lane=lane,selected_pattern=json.dumps(self.pattern_projection(p) if p
                else dict(pattern_id=None,generalized_gradient=None)),
            pattern_support=tuple(p['support_ids']) if p else (),
            repair_evidence=json.dumps(repair_evidence),parent_state_id=opportunity.parent_state_id,
            task_types=tuple(sorted({r.signals['task_metadata']['type'] for r in opportunity.evidence.mutation_evidence
                if isinstance(r.signals.get('task_metadata',{}).get('type'),str)})),
            failure_trajectories=json.dumps([r.signals['failure_trajectory'] for r in repairs]),
            private_feedback_texts=tuple(str(r.signals.get(k,'')) for r in universe
                for k in ('reference_solution',)) + tuple(r.signals.get('solver_trajectory',{}).get('visible_solution','') for r in universe)
                + tuple(str(d['diagnostic'][k]) for d in repair_evidence if d['diagnostic']
                    for k in ('observed_failure','diagnosis')))

    @staticmethod
    def pattern_projection(pattern):
        return {k:pattern[k] for k in ('pattern_id','generalized_gradient')}

CurrentOptimizer=GradientPatternMemoryOptimizer
CurrentEngine=GradientPatternMemoryEngine
CurrentLayer1Config=EvidenceLayer1Config
