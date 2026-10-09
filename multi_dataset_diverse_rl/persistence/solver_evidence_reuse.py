"""Read-only import of compatible initial Solver realization prefixes.

Old attempts, hashes, receipts and caches retain their original identities.
The new observation references these identities and charges only new transport.
No correctness outcome determines source selection or prefix admission.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json

from .durable_io import read_json
from .exact_output_cache import DurableExactOutputCache, digest
from .provider_receipts import digest as receipt_digest
from ..governance.token_accounting import serialized_request
from ..search.schemas import SearchContractError
from ..search.solver_execution import POLICY as EXECUTION, next_capacity
from ..benchmarks.math_prediction_validity import classify_prediction


POLICY=dict(identity='VERIFIED_INITIAL_REALIZATION_PREFIX_REUSE_V1',
    scope='INITIAL_OPTIMIZE_ONLY', source_selection='SOLE_FROZEN_COMPLETE_ATTEMPT',
    admission='EXACT_WIRE_BYTES_MEMBER_EXAMPLE_INTERFACE_AND_SEALED_RECEIPTS',
    prefix='STOP_BEFORE_FIRST_CAPACITY_DIVERGENCE',
    correctness_selection=False, retry_budget='FOUR_TOTAL_OLD_PLUS_NEW_SEMANTIC_DRAWS',
    old_cache_mutation=False, new_charges='NEW_TRANSPORT_ONLY',
    initial_scores='RECOMPUTE_FROM_RECONSTRUCTED_PROFILES')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wire_sha(request):
    return hashlib.sha256(serialized_request(request)).hexdigest()


def checked_path(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise SearchContractError('EVIDENCE_REUSE_PATH_ESCAPE')
    return path


def audit_source(root, *, prep_path, execution_directory, accounting_directory, require_complete=True):
    """Deterministic zero-provider inventory, producing a freezeable private manifest."""
    from ..search.provider_runtime import RequestBroker
    from ..benchmarks.math_domain_binding import execution_binding
    from ..benchmarks.math_v21_interface import interface_for_contract, solver_user_content
    prep=read_json(checked_path(root,prep_path))
    if digest({k:v for k,v in prep.items() if k!='startup_identity_sha256'})!=prep['startup_identity_sha256']:
        raise SearchContractError('EVIDENCE_REUSE_STARTUP_CORRUPTION')
    contract_path=prep['manifest']['execution_binding']['path']
    contract=read_json(checked_path(root,contract_path))
    if sha(checked_path(root,contract_path))!=prep['manifest']['execution_binding']['sha256']:
        raise SearchContractError('EVIDENCE_REUSE_BINDING_CORRUPTION')
    rr=checked_path(root,execution_directory)
    account=checked_path(root,accounting_directory)
    if read_json(rr/'lifecycle.json')['status']!='EXECUTION_ABORTED':
        raise SearchContractError('EVIDENCE_REUSE_SOURCE_NOT_CLOSED')
    inventory=read_json(rr/'raw_evidence_inventory.json')
    for row in inventory['files']:
        path=checked_path(rr,row['path'])
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise SearchContractError('EVIDENCE_REUSE_INVENTORY_CORRUPTION')
    events=[json.loads(line) for line in (account/'events.jsonl').read_bytes().splitlines()]
    pending={};charged=0;charges={};peak=0
    for index,row in enumerate(events):
        if (row.get('sequence')!=index or row.get('previous_sha256')!=(events[index-1]['event_sha256'] if index else None)
                or digest({k:v for k,v in row.items() if k!='event_sha256'})!=row['event_sha256']):
            raise SearchContractError('EVIDENCE_REUSE_LEDGER_CORRUPTION')
        if row['kind']=='AUTHORIZE':
            if index or row['task_sha256']!=contract['task_authorization_sha256']:
                raise SearchContractError('EVIDENCE_REUSE_AUTHORIZATION_CORRUPTION')
        elif row['kind']=='RESERVE':
            if row['reservation_id'] in pending:raise SearchContractError('EVIDENCE_REUSE_DUPLICATE_RESERVATION')
            pending[row['reservation_id']]=row
        elif row['kind']=='CHARGE':
            bound=pending.pop(row['reservation_id'])
            if row['charged']!=row['input_tokens']+row['output_tokens'] or not 0<=row['charged']<=bound['bound']['amount']:
                raise SearchContractError('EVIDENCE_REUSE_CHARGE_CORRUPTION')
            charged+=row['charged'];charges[row['reservation_id']]=(bound,row)
        else:raise SearchContractError('EVIDENCE_REUSE_LEDGER_EVENT_UNSUPPORTED')
        peak=max(peak,charged+sum(r['bound']['amount'] for r in pending.values()))
    end=read_json(rr/'accounting_end.json')
    if (pending or end['reserved_inflight'] or charged!=end['charged_total']
            or end['last_event_sha256']!=events[-1]['event_sha256']
            or peak>events[0]['policy']['authorized_total']):
        raise SearchContractError('EVIDENCE_REUSE_LEDGER_NOT_CLOSED')
    receipts={}
    for path in sorted((rr/'provider_response_receipts_private').glob('*.json')):
        sealed=read_json(path);payload={k:v for k,v in sealed.items() if k!='integrity_sha256'}
        bound,charge=charges[payload['reservation_id']]
        record=payload['record']
        if (receipt_digest(payload)!=sealed['integrity_sha256']
                or charge['response_receipt']['integrity_sha256']!=sealed['integrity_sha256']
                or payload['attempt_id']!=contract['execution_attempt_id']
                or payload['startup_identity_sha256']!=prep['startup_identity_sha256']
                or wire_sha(record['request'])!=bound['request_sha256']):
            raise SearchContractError('EVIDENCE_REUSE_RECEIPT_CORRUPTION')
        if 'response' in record:
            key=(record['request_sha256'],record['semantic_attempt_no'])
            if key in receipts:raise SearchContractError('EVIDENCE_REUSE_DUPLICATE_SEMANTIC_RESPONSE')
            receipts[key]=(path,sealed)
    binding=execution_binding(root,contract);benchmark=binding.benchmark()
    examples=binding.examples('optimize');team=read_json(root/contract['initial_team_path'])
    broker=RequestBroker(contract=contract,transport=lambda _: (_ for _ in ()).throw(AssertionError('ZERO_API')),
        arm='A4',seed=81)
    context=read_json(rr/'resolved_output_cache/scope.json')
    if (context['execution_attempt_id']!=contract['execution_attempt_id'] or context['binding_sha256']!=digest(contract)
            or context['source_sha']!=prep['manifest']['source_sha']
            or context['startup_identity_sha256']!=prep['startup_identity_sha256']):
        raise SearchContractError('EVIDENCE_REUSE_CACHE_SCOPE_CORRUPTION')
    cache=DurableExactOutputCache(rr/'resolved_output_cache',context)
    rows=[];features=Counter();draws=Counter();wirekeys=set();scores=Counter()
    for member in range(5):
        for example in examples:
            messages=[dict(role='system',content=interface_for_contract(contract)[0]),
                dict(role='user',content=solver_user_content(contract,team['members'][member]['prompt'],benchmark.format_input(example.item)))]
            request,key=broker._request_identity(role='solver',split='optimize',messages=messages,member_slot=member)
            wkey=(member,wire_sha(request))
            if wkey in wirekeys:raise SearchContractError('EVIDENCE_REUSE_AMBIGUOUS_EXAMPLE_WIRE')
            wirekeys.add(wkey);value=cache.get(key)
            if value is None:
                if require_complete:raise SearchContractError('EVIDENCE_REUSE_SOURCE_PROFILE_MISSING')
                continue
            refs=[];compatible=0;previous=None;diverged=False
            for ordinal,realization in enumerate(value['original_realizations'],1):
                path,sealed=receipts[(key,ordinal)];record=sealed['record'];response=record['response']
                if (record['role']!='solver' or record['split']!='optimize' or record['stage']!='initial'
                        or serialized_request(record['request'])!=serialized_request(request)
                        or response['text']!=realization['text'] or response['finish_reason']!=realization['finish_reason']
                        or response['input_tokens']!=realization['provider_reported_input_tokens']
                        or response['output_tokens']!=realization['provider_reported_output_tokens']):
                    raise SearchContractError('EVIDENCE_REUSE_REALIZATION_MISMATCH')
                cap=next_capacity(EXECUTION,previous)
                if cap!=request['max_tokens']:diverged=True
                if not diverged:compatible+=1
                previous=classify_prediction(realization['text'],realization['finish_reason'])
                refs.append(dict(receipt_path=path.relative_to(root).as_posix(),receipt_sha256=sha(path),
                    original_request_sha256=key,wire_sha256=wire_sha(request),
                    original_response_sha256=digest(response),semantic_attempt_no=ordinal))
                text=realization['text'] or '';lines=[line.strip() for line in text.splitlines() if line.strip()]
                features['physical_responses']+=1
                features['finish_'+str(realization['finish_reason'])]+=1
                features['marker_present']+=int('FINAL_ANSWER:' in text)
                features['visible_characters']+=len(text)
                features['output_reported_'+str(response['output_tokens'])]+=1
                substantive=[line for line in lines if len(line)>=20]
                repeated=(len(substantive)>4 and max(Counter(substantive).values(),default=0)>=3
                    and len(set(substantive))/len(substantive)<0.65)
                features['repeated_lines_indicator']+=int(repeated)
                if realization['finish_reason']=='length':
                    features['length_marker_present']+=int('FINAL_ANSWER:' in text)
                    features['length_repeated_lines_indicator']+=int(repeated)
                features['invalid_'+str(previous.invalid_reason)]+=int(not previous.prediction_valid)
            prediction=value['resolved_prediction'];draws[str(prediction['semantic_attempt_count'])]+=1
            from ..benchmarks.math_prediction_validity import prediction_from_persisted
            parsed=prediction_from_persisted(prediction).parsed()
            scores[member]+=int(benchmark.score_member_output(parsed,example.reference)==1)
            features['terminal_invalid']+=int(prediction['terminal_invalid'])
            rows.append(dict(member_id=member,example_id=example.item.input_id,wire_sha256=wire_sha(request),
                original_request_sha256=key,cache_path=(rr/'resolved_output_cache'/(key+'.json')).relative_to(root).as_posix(),
                cache_sha256=sha(rr/'resolved_output_cache'/(key+'.json')),compatible_prefix_draws=compatible,
                resolves_without_new_call=not diverged,realizations=refs))
    return dict(identity=POLICY['identity'],prep_path=prep_path,prep_sha256=sha(root/prep_path),
        contract_path=contract_path,contract_sha256=sha(root/contract_path),
        execution_directory=execution_directory,accounting_directory=accounting_directory,
        inventory_sha256=sha(rr/'raw_evidence_inventory.json'),ledger_sha256=sha(account/'events.jsonl'),
        source_attempt=contract['execution_attempt_id'],source_sha=prep['manifest']['source_sha'],
        startup_identity_sha256=prep['startup_identity_sha256'],historical_charged_tokens=charged,
        logical_profiles=len(rows),receipt_count=len(receipts),draw_distribution=dict(draws),features=dict(features),
        member_correct_counts=[scores[i] for i in range(5)],
        reuse_complete_profiles=sum(r['resolves_without_new_call'] for r in rows),
        reuse_physical_prefix_draws=sum(r['compatible_prefix_draws'] for r in rows),
        affected_profiles=sum(not r['resolves_without_new_call'] for r in rows),observations=rows)


class SolverEvidenceReuse:
    def __init__(self,root,contract):
        if contract.get('initial_evidence_reuse_policy')!=POLICY:
            raise SearchContractError('EVIDENCE_REUSE_POLICY_NOT_FROZEN')
        path=checked_path(root,contract['initial_evidence_reuse_manifest_path'])
        if sha(path)!=contract['initial_evidence_reuse_manifest_sha256']:
            raise SearchContractError('EVIDENCE_REUSE_MANIFEST_CORRUPTION')
        manifest=read_json(path)
        actual=audit_source(root,prep_path=manifest['prep_path'],execution_directory=manifest['execution_directory'],
            accounting_directory=manifest['accounting_directory'])
        if actual!=manifest or manifest['logical_profiles']!=300:
            raise SearchContractError('EVIDENCE_REUSE_SOURCE_CHANGED')
        self.root=root;self.manifest=manifest
        self.rows={(r['member_id'],r['wire_sha256']):r for r in manifest['observations']}

    def prefix(self,request,member):
        row=self.rows.get((member,wire_sha(request)))
        if row is None:raise SearchContractError('EVIDENCE_REUSE_INITIAL_REQUEST_INCOMPATIBLE')
        if sha(self.root/row['cache_path'])!=row['cache_sha256']:
            raise SearchContractError('EVIDENCE_REUSE_CACHE_CHANGED_AFTER_AUDIT')
        cache=read_json(self.root/row['cache_path'])['result']
        results=[]
        for realization,ref in zip(cache['original_realizations'][:row['compatible_prefix_draws']],row['realizations'],strict=False):
            if sha(self.root/ref['receipt_path'])!=ref['receipt_sha256']:
                raise SearchContractError('EVIDENCE_REUSE_RECEIPT_CHANGED_AFTER_AUDIT')
            results.append({**deepcopy(realization),'provider_called':False,'input_tokens':0,'output_tokens':0,
                'output_capacity_tokens':3600,'historical_input_tokens':realization['input_tokens'],
                'historical_output_tokens':realization['output_tokens'],
                'evidence_reuse_source':dict(source_attempt=self.manifest['source_attempt'],
                    source_sha=self.manifest['source_sha'],source_ledger_sha256=self.manifest['ledger_sha256'],
                    example_id=row['example_id'],member_id=member,**ref)})
        return tuple(results)
