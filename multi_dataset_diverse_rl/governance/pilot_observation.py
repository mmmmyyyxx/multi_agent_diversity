"""Read-only Pilot journals. No provider calls, retrieval or selection read points."""
from copy import deepcopy
from dataclasses import fields, is_dataclass

from ..persistence.durable_io import append_jsonl


def plain(value):
    if is_dataclass(value):
        return {f.name:plain(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return deepcopy(value)


def memory_snapshot(memory):
    return {**memory.export(),'audit':plain(memory.audit())}


def attach_pilot_observer(composed, run_root):
    memory=composed.memory
    context={}
    def journal(stage,data):
        append_jsonl(run_root/'pilot_observation_private.jsonl',dict(stage=stage,
            context=plain(context),data=plain(data),memory_state=memory_snapshot(memory)))
    def opportunity(stage,data):
        context.clear();context.update(opportunity_id=data['opportunity_id'],target_member=data['target_member'])
        journal(stage,data)
    def generation(stage,data):
        context.update(slot=data['outcome']['slot'])
        journal(stage,data)
    composed.observation_observer=opportunity
    composed.engine.optimizer.execution_observer=generation
    journal('PRE_PROFILE_MEMORY_STORAGE',{})
