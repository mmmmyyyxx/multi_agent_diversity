"""Test-only exact power witness before the unchanged native comparison."""
import faulthandler,json,sys
from . import math_domain_worker as native
from .math_power_certificate import POLICY,certified_far_unequal


def evaluate(expressions):
    parsed=[native.parse_payload(text) for text in expressions]
    valid=[v[1] for v in parsed];certificates=[]
    def compare(i,j):
        if not valid[i] or not valid[j]:return False
        def pair(g,p):
            if not native.compatible(g,p):return False
            if certified_far_unequal(g,p):
                certificates.append([i,j]);return False
            return native.verify_payload([g],[p])
        return any(pair(g,p) for g in parsed[i][0] for p in parsed[j][0])
    matrix=[[compare(i,j) for j in range(len(parsed))] for i in range(len(parsed))]
    return dict(valid=valid,equivalence=matrix,parsed_types=[[type(x).__name__ for x in v[0]] for v in parsed],
        certificate_policy=POLICY['identity'],certified_pair_rejections=certificates)


def main():
    import socket
    def blocked(*args,**kwargs):raise RuntimeError('MATH_WORKER_NETWORK_FORBIDDEN')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=blocked
    faulthandler.enable();faulthandler.dump_traceback_later(10)
    try:
        native.initialize();request=json.load(sys.stdin)
        if request['policy']!=POLICY:raise ValueError('MATH_POWER_CERTIFICATE_POLICY_CHANGED')
        json.dump(evaluate(request['expressions']),sys.stdout);return 0
    except Exception:
        json.dump(dict(error='MATH_EVALUATOR_FAILURE'),sys.stdout);return 1
    finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':raise SystemExit(main())
