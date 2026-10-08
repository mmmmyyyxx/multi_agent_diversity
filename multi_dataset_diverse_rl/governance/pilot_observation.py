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
    names=('revision','sequence','clock','private','failures','shared','failure_events',
        'risk_counts','counts','peak','evictions','context_lengths','read_private_count',
        'read_shared_count','write_count','limits')
    result={k:plain(getattr(memory,k)) for k in names}
    if getattr(memory,'optimization_evidence_policy',None):
        result['competence']=plain(memory.competence)
        result['optimization_evidence_policy']=plain(memory.optimization_evidence_policy)
    for row,entry in zip(result['shared'],memory.shared,strict=True):
        row.update(memory_id=entry.memory_id,visible=entry.visible())
    result['audit']=plain(memory.audit())
    return result


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
        context.update(opportunity_id=data['opportunity_id'],target_member=data['target_member'],generation=data['generation'])
        journal(stage,data)
    composed.observation_observer=opportunity
    composed.engine.optimizer.observation_observer=generation
    memory.observation_observer=journal
    journal('INITIAL_EMPTY_MEMORY',{})
