"""Private, sealed resolved outputs; reuse is confined to one frozen attempt."""
import hashlib
import json
from pathlib import Path
from .durable_io import atomic_write_json
from ..search.schemas import SearchContractError

FIELDS={'execution_attempt_id','cache_namespace','startup_identity_sha256','source_sha',
        'authorization_sha256','binding_sha256','generation_policy_sha256','recovery_policy_sha256'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class DurableExactOutputCache:
    def __init__(self, directory, context):
        self.directory=Path(directory)
        if set(context)!=FIELDS or any(not isinstance(v,str) or not v for v in context.values()):
            raise SearchContractError('DURABLE_CACHE_CONTEXT_INVALID')
        self.context=dict(context)
        self.directory.mkdir(parents=True,exist_ok=True)
        scope=self.directory/'scope.json'
        if scope.exists():
            if json.loads(scope.read_bytes()) != self.context:
                raise SearchContractError('DURABLE_CACHE_SCIENTIFIC_ATTEMPT_MISMATCH')
        else:
            if any(self.directory.iterdir()):
                raise SearchContractError('DURABLE_CACHE_SCOPE_MISSING')
            atomic_write_json(scope,self.context)

    def get(self,key):
        if len(key)!=64 or any(ch not in '0123456789abcdef' for ch in key):
            raise SearchContractError('DURABLE_CACHE_KEY_INVALID')
        p=self.directory/(key+'.json')
        if not p.exists():return None
        try:
            row=json.loads(p.read_bytes());seal=row.pop('integrity_seal')
            if (digest(row)!=seal or row['context']!=self.context or row['request_sha256']!=key
                    or digest(row['result'])!=row['response_sha256']):
                raise ValueError('seal mismatch')
            from ..benchmarks.math_structured_answer import prediction_from_persisted
            prediction=prediction_from_persisted(row['result']['resolved_prediction'])
            result=row['result']
            if (result['request_sha256']!=key or result['text']!=prediction.text
                    or result['finish_reason']!=prediction.finish_reason
                    or len(result['original_realizations'])!=prediction.semantic_attempt_count
                    or any(r['text']!=p.text or r['finish_reason']!=p.finish_reason
                        for r,p in zip(result['original_realizations'],prediction.original_predictions,strict=True))
                    or any(result[k]!=sum(r[k] for r in result['original_realizations'])
                        for k in ('input_tokens','output_tokens'))):
                raise ValueError('resolved evidence mismatch')
            return row['result']
        except (ValueError,KeyError,TypeError) as exc:
            raise SearchContractError('DURABLE_CACHE_CORRUPTION') from exc

    def put(self,key,result):
        from ..benchmarks.math_structured_answer import prediction_from_persisted
        prediction_from_persisted(result['resolved_prediction'])
        row=dict(context=self.context,request_sha256=key,result=result,response_sha256=digest(result))
        row['integrity_seal']=digest(row)
        p=self.directory/(key+'.json')
        if p.exists():
            if self.get(key)!=result:raise SearchContractError('DURABLE_CACHE_ENTRY_IMMUTABLE')
            return
        atomic_write_json(p,row)
