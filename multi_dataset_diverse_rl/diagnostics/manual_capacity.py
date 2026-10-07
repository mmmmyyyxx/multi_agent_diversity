"""Frozen, Solver-only capacity diagnostic with isolated realizations and durable accounting.

This module borrows benchmark-owned request and prediction semantics. It never
constructs an orchestrator, optimizer, Pattern provider or Memory store.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import threading

from ..benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
from ..benchmarks.math_solver_decoding import generation_request_fields, frozen_solver_policy
from ..governance.artifacts import build_sha256_manifest
from ..governance.autonomous_math import create_transport
from ..governance.token_accounting import TokenLedger, POLICY_40M, OperationalAbort, reservation
from ..persistence.durable_io import append_jsonl, atomic_write_json, read_json
from ..persistence.exact_output_cache import DurableExactOutputCache
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..search.provider_runtime import RequestBroker, BenchmarkSolver
from ..search.schemas import SearchContractError

IDENTITY='MANUAL_PROMPT_CAPACITY_PROBE_V1'
PROMPTS=('P0_BASELINE','P1_GENERALIST','P2_SPECIALIST_A','P3_SPECIALIST_B','P4_SPECIALIST_C','P5_SPECIALIST_D')


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hash(path):
    return sha256(Path(path).read_text(encoding='utf8').encode()).hexdigest()


class SolverDiagnosticBroker(RequestBroker):
    """Retain frozen recovery/transport physics and enforce a narrower capability."""
    def __init__(self, *, diagnostic_identity, **kwargs):
        self.diagnostic_identity=dict(diagnostic_identity)
        super().__init__(**kwargs)
        if self.contract['models']['solver']!='qwen3-8b':
            raise SearchContractError('MANUAL_PROBE_SOLVER_NOT_FROZEN')

    def _request_identity(self, *, role, split, messages, member_slot=None):
        if role!='solver' or split!='optimize':
            raise SearchContractError('MANUAL_PROBE_ROLE_OR_SPLIT_FORBIDDEN')
        request,_=super()._request_identity(role=role,split=split,messages=messages,member_slot=member_slot)
        identity=dict(diagnostic=self.diagnostic_identity,role=role,split=split,request=request,
            cache_namespace=self.contract['cache_namespace'],member_slot=member_slot,
            solver_output_interface=self.contract['solver_output_interface'],
            prediction_validity_policy=self.prediction_policy,invalid_recovery_policy=self.recovery_policy)
        return request,digest(identity)

    def with_contract(self, contract):
        raise SearchContractError('MANUAL_PROBE_CONTRACT_REBIND_FORBIDDEN')

    def private_capability(self):
        raise SearchContractError('MANUAL_PROBE_CAPABILITY_EXPANSION_FORBIDDEN')


class DiagnosticTokenBudget:
    """A finite attempt charge ceiling layered over the unchanged cumulative journal."""
    def __init__(self, ledger, *, start_charge, ceiling):
        self.ledger,self.start,self.ceiling=ledger,start_charge,ceiling

    def reserve(self, request, **kwargs):
        spent=self.ledger.view()['charged_total']-self.start
        if spent<0 or spent+reservation(request)['amount']>self.ceiling:
            raise OperationalAbort('MANUAL_PROBE_TOKEN_CEILING_PRE_TRANSPORT')
        return self.ledger.reserve(request,**kwargs)

    def reconcile(self, *args, **kwargs):
        return self.ledger.reconcile(*args,**kwargs)


def validate_manifest(manifest, parent):
    if (manifest.get('identity')!=IDENTITY or manifest.get('logical_evaluations')!=720
            or manifest.get('replicates')!=[1,2] or manifest.get('Optimize_count')!=60
            or manifest.get('manual_team')!=list(PROMPTS[1:])
            or manifest.get('allowed_roles')!=['solver'] or manifest.get('allowed_splits')!=['optimize']
            or manifest.get('scientific_method_changed') is not False
            or manifest.get('prompt_edit_after_results') is not False
            or manifest.get('scope_expansion') is not False):
        raise SearchContractError('MANUAL_PROBE_FROZEN_PROTOCOL_MISMATCH')
    if (manifest.get('evaluation_order') != 'replicate_then_prompt_then_frozen_optimize_order'
            or manifest.get('replacement_member_slots') != dict(zip(PROMPTS,(0,0,1,2,3,4)))
            or manifest.get('attempt_id') == parent['execution_attempt_id']
            or manifest.get('cache_namespace') == parent['cache_namespace']
            or not manifest.get('attempt_id') or not manifest.get('cache_namespace')):
        raise SearchContractError('MANUAL_PROBE_REALIZATION_IDENTITY_MISMATCH')
    design=manifest.get('prompts',[])
    if [p.get('prompt_id') for p in design]!=list(PROMPTS):
        raise SearchContractError('MANUAL_PROBE_PROMPT_SET_MISMATCH')
    for p in design:
        text=p.get('prompt_text')
        if (not isinstance(text,str) or not text.strip() or len(text)>1200
                or sha256(text.encode()).hexdigest()!=p.get('prompt_sha256')):
            raise SearchContractError('MANUAL_PROBE_PROMPT_HASH_OR_LENGTH_MISMATCH')
    if design[0]['prompt_text']!='Solve the problem.':
        raise SearchContractError('MANUAL_PROBE_BASELINE_MISMATCH')
    expected_fields=generation_request_fields(parent,'solver')
    if (manifest.get('model')!=parent['models']['solver'] or manifest.get('model')!='qwen3-8b'
            or manifest.get('solver_request_fields')!=expected_fields
            or manifest.get('solver_output_interface')!=parent['solver_output_interface']
            or manifest.get('prediction_validity_policy')!=parent['prediction_validity_policy']
            or manifest.get('invalid_recovery_policy')!=parent['invalid_recovery_policy']
            or manifest.get('transport_decoding')!=parent['decoding']):
        raise SearchContractError('MANUAL_PROBE_SOLVER_SEMANTICS_CHANGED')
    if (manifest.get('successful_provider_call_ceiling')!=720*4
            or manifest.get('transport_attempt_ceiling')!=720*4*(parent['decoding']['transport_retries']+1)
            or type(manifest.get('token_charge_ceiling')) is not int
            or not 0<manifest['token_charge_ceiling']<=40_000_000):
        raise SearchContractError('MANUAL_PROBE_FINITE_BOUNDS_MISMATCH')


def validate_prep(root, prep):
    root,prep=Path(root).resolve(),Path(prep).resolve()
    if not prep.is_relative_to(root/'runs'):
        raise SearchContractError('MANUAL_PROBE_PREP_OUTSIDE_IGNORED_RUNS')
    payload=read_json(prep/'prep.json');manifest=read_json(prep/'manifest_private.json')
    auth=read_json(prep/'authorization.json')
    parent_path=root/payload['parent_binding_path']
    parent=read_json(parent_path)
    if sha256(parent_path.read_bytes()).hexdigest()!=payload['parent_binding_sha256']:
        raise SearchContractError('MANUAL_PROBE_PARENT_BINDING_HASH_MISMATCH')
    validate_manifest(manifest,parent)
    if digest(manifest)!=payload['manifest_sha256'] or auth.get('scope')!=payload['scope']:
        raise SearchContractError('MANUAL_PROBE_MANIFEST_OR_SCOPE_MISMATCH')
    if (manifest['attempt_id']!=payload['scope']['attempt_id']
            or manifest['token_charge_ceiling']!=payload['scope']['token_charge_ceiling']
            or payload['scope']['allowed_roles']!=['solver'] or payload['scope']['allowed_splits']!=['optimize']
            or payload['scope']['logical_evaluations']!=720
            or payload['scope']['user_task_sha256']!=payload['user_task_sha256']):
        raise SearchContractError('MANUAL_PROBE_AUTHORIZATION_BINDING_MISMATCH')
    bound = dict(source_sha=payload['source_sha'], manifest_sha256=payload['manifest_sha256'],
        parent_binding_sha256=payload['parent_binding_sha256'], examples_sha256=payload['examples_sha256'],
        model=manifest['model'], cache_namespace=manifest['cache_namespace'],
        successful_provider_call_ceiling=manifest['successful_provider_call_ceiling'],
        transport_attempt_ceiling=manifest['transport_attempt_ceiling'],
        prompt_hashes={p['prompt_id']:p['prompt_sha256'] for p in manifest['prompts']})
    if any(payload['scope'].get(k)!=v for k,v in bound.items()):
        raise SearchContractError('MANUAL_PROBE_EXACT_SCOPE_BINDING_MISMATCH')
    if auth.get('source_sha')!=payload['source_sha'] or auth.get('manifest_sha256')!=payload['manifest_sha256']:
        raise SearchContractError('MANUAL_PROBE_AUTHORIZATION_SOURCE_MISMATCH')
    task_path=root/payload['user_task_path']
    if sha256(task_path.read_bytes()).hexdigest()!=payload['user_task_sha256']:
        raise SearchContractError('MANUAL_PROBE_USER_AUTHORITY_HASH_MISMATCH')
    if subprocess.check_output(['git','cat-file','-t',payload['source_sha']],cwd=root,text=True).strip()!='commit':
        raise SearchContractError('MANUAL_PROBE_SOURCE_COMMIT_MISSING')
    for path,expected in payload['source_file_hashes'].items():
        if source_hash(root/path)!=expected:
            raise SearchContractError('MANUAL_PROBE_SOURCE_WORKTREE_MISMATCH')
        committed=subprocess.check_output(['git','show',payload['source_sha']+':'+path],cwd=root)
        normalized=committed.decode('utf8').replace('\r\n','\n')
        if sha256(normalized.encode()).hexdigest()!=expected:
            raise SearchContractError('MANUAL_PROBE_SOURCE_COMMIT_MISMATCH')
    if sha256((prep/'examples_private.json').read_bytes()).hexdigest()!=payload['examples_sha256']:
        raise SearchContractError('MANUAL_PROBE_OPTIMIZE_DATA_HASH_MISMATCH')
    return payload,manifest,parent,auth


def preflight(root, prep):
    payload,manifest,parent,_=validate_prep(root,prep)
    if MATHGradientPatternBinding(Path(root),parent).blockers():
        raise SearchContractError('MANUAL_PROBE_PARENT_DEPENDENCIES_INVALID')
    return dict(gate='MANUAL_PROBE_READY_NOT_AUTHORIZED',attempt_id=manifest['attempt_id'],
        manifest_sha256=payload['manifest_sha256'],provider_calls=0,
        logical_evaluations=720,successful_provider_call_ceiling=manifest['successful_provider_call_ceiling'],
        transport_attempt_ceiling=manifest['transport_attempt_ceiling'],token_charge_ceiling=manifest['token_charge_ceiling'])


def execute(root, prep, run_root, *, transport_override=None):
    """Consume one exact frozen scope. Failure never permits a partial resume."""
    root,prep,run_root=Path(root).resolve(),Path(prep).resolve(),Path(run_root).resolve()
    payload,manifest,parent,auth=validate_prep(root,prep)
    if (auth.get('explicit_user_authorized') is not True or auth.get('consumed')
            or auth.get('closed') or auth.get('source_sha')!=payload['source_sha']
            or auth.get('manifest_sha256')!=payload['manifest_sha256']):
        raise SearchContractError('MANUAL_PROBE_NOT_EXACTLY_AUTHORIZED')
    if (run_root.exists() or not run_root.is_relative_to(root/'runs')
            or run_root!=root/payload['scope']['run_root']):
        raise SearchContractError('MANUAL_PROBE_FRESH_RUN_REQUIRED')
    if MATHGradientPatternBinding(root,parent).blockers():
        raise SearchContractError('MANUAL_PROBE_PARENT_DEPENDENCIES_INVALID')
    binding=MATHGradientPatternBinding(root,parent)
    examples=binding.examples('optimize');benchmark=binding.benchmark()
    expected=read_json(prep/'examples_private.json')
    actual=[dict(input_id=e.item.input_id,problem=benchmark.format_input(e.item),reference=e.reference) for e in examples]
    if actual!=expected or len(actual)!=60 or len({e['input_id'] for e in actual})!=60:
        raise SearchContractError('MANUAL_PROBE_OPTIMIZE_MEMBERSHIP_MISMATCH')
    marker=prep/'authorization_consumed.json'
    with marker.open('x',encoding='utf8') as stream:
        import os
        json.dump(dict(scope=payload['scope'],manifest_sha256=payload['manifest_sha256']),stream,sort_keys=True)
        stream.flush();os.fsync(stream.fileno())
    run_root.mkdir()
    atomic_write_json(run_root/'consumed_authorization.json',{**auth,'consumed':True})
    atomic_write_json(run_root/'lifecycle.json',dict(status='RUNNING',attempt_id=manifest['attempt_id']))
    usage=dict(attempts=0,successes=0,failures=0,input_tokens=0,output_tokens=0,
               solver=0,reflection=0,pattern=0,pattern_gradient=0,pattern_cluster=0,validation=0,test=0)
    lock=threading.RLock();resolved=[];client=None
    try:
        if transport_override is None:
            transport,client=create_transport(parent)
        else:
            transport=transport_override
        with TokenLedger(root/parent['token_ledger_directory'],task_sha256=parent['task_authorization_sha256'],
                         policy=POLICY_40M,best_effort_snapshots=True) as ledger:
            start=ledger.view();atomic_write_json(run_root/'accounting_start.json',start)
            if manifest['token_charge_ceiling']>ledger.remaining:
                raise OperationalAbort('MANUAL_PROBE_TOKEN_BOUND_EXCEEDS_REMAINING')
            capped=DiagnosticTokenBudget(ledger,start_charge=start['charged_total'],ceiling=manifest['token_charge_ceiling'])
            receipts=ProviderResponseReceipts(run_root/'provider_response_receipts',
                attempt_id=manifest['attempt_id'],startup_identity_sha256=payload['manifest_sha256'])
            for replicate in manifest['replicates']:
                for prompt in manifest['prompts']:
                    pid=prompt['prompt_id'];lane=max(0,PROMPTS.index(pid)-1)
                    namespace=manifest['cache_namespace']+'/'+pid+'/r'+str(replicate)
                    c=deepcopy(parent);c.update(execution_attempt_id=manifest['attempt_id'],cache_namespace=namespace,
                        execution_phase='manual_prompt_capacity_probe')
                    c['provider_bounds']=dict(solver_calls=manifest['successful_provider_call_ceiling'],
                        reflection_calls=0,pattern_calls=0,pattern_gradient_calls=0,pattern_cluster_calls=0,
                        successful_provider_calls=manifest['successful_provider_call_ceiling'],
                        transport_attempts=manifest['transport_attempt_ceiling'])
                    context=dict(execution_attempt_id=manifest['attempt_id'],cache_namespace=namespace,
                        startup_identity_sha256=payload['manifest_sha256'],source_sha=payload['source_sha'],
                        authorization_sha256=digest(auth),binding_sha256=digest(c),
                        generation_policy_sha256=digest(frozen_solver_policy(c)),
                        recovery_policy_sha256=digest(manifest['invalid_recovery_policy']))
                    cache=DurableExactOutputCache(run_root/'cache'/pid/('r'+str(replicate)),context)
                    broker=SolverDiagnosticBroker(contract=c,transport=transport,arm='MANUAL_PROBE',seed=81,
                        diagnostic_identity=dict(identity=IDENTITY,manifest_sha256=payload['manifest_sha256'],
                            prompt_id=pid,replicate=replicate),usage=usage,lock=lock,token_ledger=capped,
                        durable_cache=cache,response_receipts=receipts,
                        ledger_writer=lambda row:append_jsonl(run_root/'ledger.jsonl',row),
                        raw_writer=lambda row:append_jsonl(run_root/'provider_trace_private.jsonl',row))
                    solver=BenchmarkSolver(benchmark,broker);solver.observe_member(lane)
                    outputs=[]
                    for example in examples:
                        prediction=solver.solve(prompt['prompt_text'],example.item,stage=pid+'_r'+str(replicate),split='optimize')
                        value=asdict(prediction);outputs.append(value)
                        append_jsonl(run_root/'logical_results_private.jsonl',dict(prompt_id=pid,replicate=replicate,
                            input_id=example.item.input_id,prediction=value))
                    row=dict(prompt_id=pid,replicate=replicate,member_slot=lane,outputs=outputs)
                    resolved.append(row);append_jsonl(run_root/'profiles_private.jsonl',row)
                    print(json.dumps(dict(status='MANUAL_PROFILE_COMPLETE',prompt_id=pid,replicate=replicate,
                        logical_completed=len(resolved)*60,successful_provider_calls=usage['successes'],
                        charged_so_far=ledger.view()['charged_total']-start['charged_total'],remaining=ledger.remaining)),flush=True)
            if len(resolved)!=12 or usage['reflection'] or usage['pattern'] or usage['validation'] or usage['test']:
                raise OperationalAbort('MANUAL_PROBE_SCOPE_COMPLETION_MISMATCH')
            atomic_write_json(run_root/'profiles_complete_private.json',resolved)
            atomic_write_json(run_root/'provider_usage.json',usage)
            atomic_write_json(run_root/'accounting_end.json',ledger.view())
            if ledger.view()['reserved_inflight']:
                raise OperationalAbort('MANUAL_PROBE_RESERVATION_NOT_CLOSED')
        atomic_write_json(run_root/'lifecycle.json',dict(status='EXECUTION_COMPLETE',attempt_id=manifest['attempt_id'],
            logical_evaluations=720,scientific_method_changed=False))
        atomic_write_json(run_root/'raw_evidence_inventory.json',build_sha256_manifest(run_root))
        return dict(status='EXECUTION_COMPLETE',attempt_id=manifest['attempt_id'],logical_evaluations=720,**usage)
    except BaseException as exc:
        atomic_write_json(run_root/'lifecycle.json',dict(status='EXECUTION_ABORTED',attempt_id=manifest['attempt_id'],
            error_category=type(exc).__name__,error_code=str(exc) if isinstance(exc,(OperationalAbort,SearchContractError)) else None,
            completed_profiles=len(resolved),scientific_result='NOT_EVALUABLE_OPERATIONAL_FAILURE'))
        atomic_write_json(run_root/'provider_usage.json',usage)
        atomic_write_json(run_root/'raw_evidence_inventory.json',build_sha256_manifest(run_root))
        raise
    finally:
        if client is not None:
            client.close()
