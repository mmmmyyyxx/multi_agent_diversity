"""Complete fabricated default-run witness; private payloads are never published."""
import asyncio,json
from dataclasses import asdict,is_dataclass
from pathlib import Path
from tests.current.population_support import graph
from multi_dataset_diverse_rl.search.policies import GlobalStopPolicy

CASES=('default','reflection_recovery','group_exhaustion','uncertain','multi_opportunity','pool_evolution')

def normalize(value):
    if is_dataclass(value):return normalize(asdict(value))
    if isinstance(value,dict):return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [normalize(v) for v in value]
    if isinstance(value,(set,frozenset)):return sorted(normalize(v) for v in value)
    if isinstance(value,Path):return '<RUN_ROOT>'
    return value

def canonical_witness(case):
    # Independent Solver dispatch order is unspecified. Keep every full payload
    # and multiplicity within contiguous Solver blocks, plus all optimizer order.
    # Event/state ordering is independently retained in the same witness.
    case=json.loads(json.dumps(case));ordered=[];block=[]
    for request in case['requests']:
        if request['model']=='gpt-4o-mini':block.append(request)
        else:
            ordered.extend(sorted(block,key=lambda r:json.dumps(r,sort_keys=True)))
            block=[];ordered.append(request)
    ordered.extend(sorted(block,key=lambda r:json.dumps(r,sort_keys=True)))
    case['requests']=ordered
    return case

def capture(name,folder):
    kwargs={};horizon=1
    def recovered(packet,count):
        return 'bad JSON' if count==1 else json.dumps(dict(target_block='strategy',
            new_content=f'Apply a conditional repair rule {count}: check domains before transformations.',
            change_summary='Add one conditional domain repair.'))
    if name=='reflection_recovery':kwargs['reflection']=recovered
    elif name=='group_exhaustion':kwargs['group']=lambda packet:'bad JSON'
    elif name=='uncertain':kwargs['uncertain']=True
    elif name in {'multi_opportunity','pool_evolution'}:horizon=2
    elif name!='default':raise ValueError(name)
    run,broker,requests,packets=graph(folder,**kwargs)
    if horizon>1:run.stop=GlobalStopPolicy(run.method.global_stop.no_commit_patience)
    if name=='pool_evolution':run.history.failure_counts.update({m:99 for m in range(1,5)})
    events=[];run.execution_observer=lambda stage,data:events.append(dict(stage=stage,data=normalize(data)))
    result=asyncio.run(run.run(max_opportunities=horizon))
    return canonical_witness(normalize(dict(result=result,events=events,requests=requests,
        reflection_packets=packets,history=run.history.__dict__,pool=run.population.export(),
        memory=run.memory.export(),checkpoint=json.loads(run.checkpoint.path.read_bytes()),
        request_count=broker.successes,role_usage=broker.role_usage)))
