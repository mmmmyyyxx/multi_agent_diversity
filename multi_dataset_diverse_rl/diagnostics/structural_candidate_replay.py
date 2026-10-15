"""Offline reclassification of sealed historical draws; never evaluates candidates."""
import collections
import hashlib
import json
from pathlib import Path

from ..search.candidate_contract import parse_candidate_edit,POLICY
from ..search.system_prompt import SystemPrompt
from ..search.generation_failures import GeneratedOutputFailure

def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()

def compare_candidates(root,index_path):
    root=Path(root).resolve();index_path=Path(index_path)
    rows=json.loads(index_path.read_text(encoding='utf-8'));seen=set();out=[]
    index_before=hashlib.sha256(index_path.read_bytes()).hexdigest()
    for row in rows:
        if row['id'] in seen:raise ValueError('duplicate historical receipt')
        seen.add(row['id'])
        path=(root/row['receipt']).resolve()
        if not path.is_relative_to(root):raise ValueError('receipt outside workspace')
        raw=path.read_bytes();record=json.loads(raw)
        if record['integrity_sha256']!=row['id'] or _digest({k:v for k,v in record.items() if k!='integrity_sha256'})!=row['id']:
            raise ValueError('historical receipt integrity mismatch')
        rec=record['record'];response=rec['response'];text=response['text']
        if hashlib.sha256(text.encode()).hexdigest()!=row['response_sha256']:
            raise ValueError('historical candidate response mismatch')
        packet=json.loads(rec['request']['messages'][1]['content'])
        parent=SystemPrompt.from_dict(packet['current_system_prompt'])
        old='REJECTED' if row['logged_errors'] else 'PASSED'
        try:
            edit=parse_candidate_edit(text,parent=parent,finish_reason=response.get('finish_reason'))
            new='NO_OP_SKIP' if edit.no_op else 'ELIGIBLE_FOR_EVALUATION'
            chars=len(edit.prompt.render());candidate_hash=edit.prompt.prompt_hash
        except GeneratedOutputFailure:
            new='STRUCTURAL_OR_PARSE_REJECT';chars=candidate_hash=None
        out.append(dict(id=row['id'],source=row['source'],cell=row['cell'],arm=row['arm'],
            round=row['round'],draw=row['draw'],log=row['log'],receipt=row['receipt'],
            old_errors=row['logged_errors'],old=old,new=new,prior_manual_classification=row['classification'],
            rendered_characters=chars,new_candidate_sha256=candidate_hash,
            actual_evaluation=False,historical_outcome_modified=False))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashlib.sha256(raw).hexdigest():
            raise ValueError('historical receipt changed during read')
    if hashlib.sha256(index_path.read_bytes()).hexdigest()!=index_before:
        raise ValueError('historical index changed during read')
    def summarize(rs):
        transitions=collections.Counter(r['old']+' -> '+r['new'] for r in rs)
        by_reason=collections.defaultdict(collections.Counter)
        for r in rs:
            for reason in r['old_errors']:by_reason[reason][r['new']]+=1
        confirmed=[r for r in rs if r['prior_manual_classification']=='certain_false_rejection']
        return dict(draws=len(rs),new_decisions=dict(collections.Counter(r['new'] for r in rs)),
            transitions=dict(transitions),by_old_reason={k:dict(v) for k,v in by_reason.items()},
            confirmed_false_rejections=len(confirmed),confirmed_false_now_eligible=sum(r['new']=='ELIGIBLE_FOR_EVALUATION' for r in confirmed))
    return dict(identity='OFFLINE_STRUCTURAL_CANDIDATE_RECLASSIFICATION_V2',policy=POLICY,
        source_index_sha256=index_before,summary=summarize(out),
        by_arm={a:summarize([r for r in out if r['arm']==a]) for a in sorted(set(r['arm'] for r in out))},
        rows=out,real_api_calls=0,solver_or_validation_evaluations=0,
        interpretation='Independent draw reclassification; earlier new no-ops would stop old retry chains. Not a counterfactual experiment or historical acceptance.')
