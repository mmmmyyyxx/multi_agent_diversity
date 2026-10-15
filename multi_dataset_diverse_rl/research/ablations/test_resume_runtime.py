"""Test-only paid Solver prefix and retry of the unchanged local mathematical scorer."""
from copy import deepcopy
from dataclasses import asdict
import hashlib,time
from ...governance.token_accounting import serialized_request
from ...persistence.durable_io import append_jsonl
from ...benchmarks.data_freeze import digest
from ...search.schemas import SearchContractError
from .runtime import AblationBroker


def draw_key(cell,stage,request):
    return (cell,stage,hashlib.sha256(serialized_request(request)).hexdigest())


def evaluator_timeout(error):
    if not isinstance(error,SearchContractError):return False
    if str(error)=='MATH_EVALUATOR_TIMEOUT':return True
    return str(error)=='REFERENCE_UNSCORABLE' and evaluator_timeout(error.__cause__)


class TestResumeBroker(AblationBroker):
    def __init__(self,*,paid_prefix,terminal_observations,**kwargs):
        super().__init__(**kwargs)
        if self.phase!='test':raise SearchContractError('TEST_RESUME_TEST_ONLY')
        self.paid_prefix=deepcopy(paid_prefix)
        self.terminal_observations=deepcopy(terminal_observations)
        self.used_prefix=set();self.local_retry_count=0

    def complete(self,*,role,split,stage,messages,capacity=None):
        if role!='solver' or split!='test':raise SearchContractError('TEST_RESUME_SOLVER_ONLY')
        request,key=self._request_identity(role=role,split=split,messages=messages,capacity=capacity)
        slot=draw_key(self.cell_id,stage,request)
        if slot in self.paid_prefix:
            entry=self.paid_prefix[slot]
            with self.lock:
                self.dispatch_guard(self.cell_id)
                if slot in self.used_prefix:raise SearchContractError('TEST_RESUME_PREFIX_DRAW_REUSED')
                if entry['request']!=request:raise SearchContractError('TEST_RESUME_PREFIX_REQUEST_CHANGED')
                self.used_prefix.add(slot);self.usage['replayed_calls']+=1
                append_jsonl(self.receipts.directory/'solver_paid_prefix.jsonl',dict(cell_id=self.cell_id,
                    stage=stage,request_sha256=key,provider_request_sha256=slot[2],
                    parent_reservation_id=entry['reservation_id'],parent_receipt_sha256=entry['receipt_sha256']))
            from ...governance.token_accounting import reservation
            return self._validated_response(deepcopy(entry['response']),request,role,key,reservation(request))
        # A different request must never replace an existing parent slot.
        problem=request['messages'][1]['content']
        if any(c==self.cell_id and s==stage and e['request']['messages'][1]['content']==problem
                for (c,s,_),e in self.paid_prefix.items()):
            raise SearchContractError('TEST_RESUME_PREFIX_REQUEST_CHANGED')
        return super().complete(role=role,split=split,stage=stage,messages=messages,capacity=capacity)

    def score_resolved(self,resolved,row):
        return self.retry_local(lambda:super(TestResumeBroker,self).score_resolved(resolved,row),
            dict(operation='resolved_answer_scoring',example_id=row['stable_example_id'],
                resolved_prediction_sha256=digest(asdict(resolved))))

    def retry_local(self,operation,context):
        retry=0
        while True:
            self.dispatch_guard(self.cell_id)
            try:return operation()
            except SearchContractError as exc:
                if not evaluator_timeout(exc):raise
                schedule=self.scope['local_evaluator_recovery']['backoff_seconds']
                delay=schedule[min(retry,len(schedule)-1)];retry+=1
                with self.lock:
                    self.local_retry_count+=1
                    append_jsonl(self.receipts.directory/'local_evaluator_recovery.jsonl',dict(
                        context,cell_id=self.cell_id,retry_index=retry,
                        error='MATH_EVALUATOR_TIMEOUT',backoff_seconds=delay,
                        model_call_added=False))
                time.sleep(delay)

    def prediction(self,*args,**kwargs):
        value=super().prediction(*args,**kwargs)
        key=(self.cell_id,value['request_sha256'])
        if key in self.terminal_observations and digest(value)!=self.terminal_observations[key]:
            raise SearchContractError('TEST_RESUME_PARENT_TERMINAL_SCORE_CHANGED')
        return value
