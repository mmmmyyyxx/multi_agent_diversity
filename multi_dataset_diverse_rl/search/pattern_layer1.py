"""Selected-pattern current evidence with the unchanged bounded Memory search."""
from dataclasses import asdict, dataclass
import json

from .. import versions
from ..local_optimizers.schemas import LocalEvidenceExample
from .layer1_memory import (MemoryLayer1Config, MemoryLocalTask, MemoryConditionedOptimizer,
    MemoryConditionedEngine, INSTRUCTION, bounded_input)
from .layer1_responsibility import ResponsibilityConditionedEngine
from .schemas import SearchContractError


@dataclass(frozen=True)
class PatternLayer1Config(MemoryLayer1Config):
    optimizer_input_schema: str = versions.PATTERN_OPTIMIZER_INPUT_VERSION
    panel_policy: str = versions.PATTERN_CONDITIONED_EVIDENCE_VERSION

    def __post_init__(self):
        expected={**asdict(MemoryLayer1Config()),'optimizer_input_schema':versions.PATTERN_OPTIMIZER_INPUT_VERSION,
            'panel_policy':versions.PATTERN_CONDITIONED_EVIDENCE_VERSION}
        if asdict(self)!=expected:raise SearchContractError('PATTERN_LAYER1_FROZEN_CONFIG_MISMATCH')


@dataclass(frozen=True)
class PatternLocalTask(MemoryLocalTask):
    selected_pattern: str = ''


def pattern_input(task,parent,observations,memory):
    # Reuse the unchanged visible current-panel and bounded-memory serialization.
    value=json.loads(bounded_input(task,parent,observations,memory)[len(INSTRUCTION)+1:])
    value.update(schema=versions.PATTERN_OPTIMIZER_INPUT_VERSION,
        selected_pattern=json.loads(task.selected_pattern),repair_objective='Only the selected semantic failure mechanism')
    return INSTRUCTION+'\n'+json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True)


class PatternMemoryOptimizer(MemoryConditionedOptimizer):
    config_factory=PatternLayer1Config
    prompt_builder=staticmethod(pattern_input)


class PatternMemoryEngine(MemoryConditionedEngine):
    def make_task(self,opportunity,context):
        base=ResponsibilityConditionedEngine.make_task(self,opportunity,context)
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
        visible={k:p[k] for k in ('pattern_id','failure_mechanism','corrective_principle')}
        visible['responsibility_value']=p['responsibility']['raw_value']
        lane=opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
        return PatternLocalTask(**{**base.__dict__,'optimization_context':lane},anchor_example=local,
            responsibility_lane=lane,selected_pattern=json.dumps(visible,sort_keys=True,separators=(',',':')))
