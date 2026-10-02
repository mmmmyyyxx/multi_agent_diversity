"""Reproducible, bounded whole-source reference probes without model runtime."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from ..benchmarks.math_domain_prep import CONTEXT,source_references
from ..benchmarks.math_domain_worker import initialize,evaluate,ambiguous_parentheses


def category(text,types):
    if ambiguous_parentheses(text): return 'ambiguous_two_component_parentheses'
    if any('Matrix' in t for t in types): return 'matrix_like'
    if 'Tuple' in types: return 'ordered_tuple_vector_like'
    if any(t in ['Interval','Union'] for t in types): return 'interval'
    if 'FiniteSet' in types: return 'finite_set' if '{' in text else 'multiple_values'
    if any(t in ['Equality','And','LessThan','StrictLessThan','GreaterThan','StrictGreaterThan'] for t in types): return 'equation_relation'
    if '%' in text: return 'percentage'
    if any(t in ['Integer','Rational','Float','Half','One','Zero','NegativeOne','Pi'] for t in types): return 'scalar_number'
    if types: return 'symbolic_expression'
    return 'textual_mixed' if any(c.isalpha() for c in text) else 'unclassified'


def probe(text,*,native):
    if native:
        from ..benchmarks.math_worker import evaluate as evaluate_native
        from math_verify import parse,LatexExtractionConfig
        from ..benchmarks.math import payload_supported
        result=evaluate_native([text])
        parsed=parse(text if '$' in text or '\\boxed' in text else '$'+text+'$',
            extraction_config=(LatexExtractionConfig(),),fallback_mode='no_fallback',extraction_mode='first_match',parsing_timeout=3)
        types=[type(x).__name__ for x in parsed]
    else:
        result=evaluate([text]);types=result['parsed_types'][0]
    data=dict(parseable=bool(types),parsed_types=types,self_equivalent=result['equivalence'][0][0],
        scorable=result['valid'][0] and result['equivalence'][0][0],
        representation_category=category(text,types),exception=False,timeout=False)
    if native: data['old_payload_guard_supported']=payload_supported(text)
    return data


def worker(native):
    import socket
    def blocked(*args,**kwargs): raise RuntimeError('EVALUATOR_PREPARATION_NETWORK_FORBIDDEN')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=blocked
    initialize()
    for line in sys.stdin:
        try:
            result=probe(json.loads(line)['reference'] or '',native=native)
        except Exception as exc:
            result=dict(parseable=False,parsed_types=[],self_equivalent=False,scorable=False,
                representation_category='unclassified',exception=True,timeout=False,exception_category=type(exc).__name__)
        print(json.dumps(result),flush=True)


def run_census(canonical_root,*,expected_manifest_sha256,destination,native=False,workers=8):
    if workers!=8: raise ValueError('PREPARATION_WORKER_POLICY_MISMATCH')
    rows=list(source_references(canonical_root,expected_manifest_sha256=expected_manifest_sha256,context=CONTEXT))
    if len(rows)!=12500: raise ValueError('FULL_REFERENCE_DOMAIN_AUDIT_REQUIRED')
    destination.mkdir(parents=True,exist_ok=False)
    lock=threading.Lock();progress=0
    def lane(items):
        nonlocal progress
        process=None;outputs=[]
        for row in items:
            if process is None:
                args=[sys.executable,'-m',__name__,'--worker']+(['--native'] if native else [])
                process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                    text=True,encoding='utf-8',bufsize=1,env=dict(os.environ))
            process.stdin.write(json.dumps({'reference':row['reference']})+'\n');process.stdin.flush()
            timer=threading.Timer(8,process.kill);timer.start()
            try:
                line=process.stdout.readline()
                if not line:
                    result=dict(parseable=False,parsed_types=[],self_equivalent=False,scorable=False,
                        representation_category='unclassified',exception=False,timeout=True)
                    process.wait();process=None
                else: result=json.loads(line)
            finally: timer.cancel()
            result.update({k:v for k,v in row.items() if k!='reference'})
            result.update(extractable=bool(str(row['reference'] or '').strip()),reference_sha256=hashlib.sha256((row['reference'] or '').encode()).hexdigest())
            outputs.append(result)
            with lock:
                progress+=1
                if progress%500==0: print(json.dumps(dict(mode='native' if native else 'v2',progress=progress,total=12500)),flush=True)
        if process is not None:
            process.stdin.close();process.wait(timeout=10)
        return outputs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs=[row for part in pool.map(lane,[rows[i::workers] for i in range(workers)]) for row in part]
    outputs.sort(key=lambda r:(r['source_split'],r['source_index']))
    (destination/'census.jsonl').write_bytes(b''.join((json.dumps(r,sort_keys=True)+'\n').encode() for r in outputs))
    summary={source:{key:sum(bool(r.get(key)) for r in outputs if r['source_split']==source)
        for key in ['extractable','parseable','self_equivalent','scorable','exception','timeout','old_payload_guard_supported']} for source in ['train','test']}
    summary['parsed_types']=dict(Counter(t for r in outputs for t in r['parsed_types']))
    summary['context']=CONTEXT;summary['model_calls']=0
    (destination/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
    return summary


if __name__=='__main__':
    if '--worker' not in sys.argv: raise SystemExit('WORKER_ONLY')
    worker('--native' in sys.argv)
