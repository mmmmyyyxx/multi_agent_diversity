"""Same pinned parser/verifier, preceded only by exact far-inequality certificates."""
import json,logging,sys
from . import math_domain_worker as native
from .math_interval_certificate import POLICY,certified_far_unequal


def evaluate(expressions):
    parsed=[native.parse_payload(text) for text in expressions]
    valid=[x[1] for x in parsed];certificates=[]
    def compare(i,j):
        if not valid[i] or not valid[j]:return False
        def pair(g,p):
            if not native.compatible(g,p):return False
            if certified_far_unequal(g,p):
                certificates.append([i,j]);return False
            # Call the exact original predicate for every noncertified pair.
            return native.verify_payload([g],[p])
        return any(pair(g,p) for g in parsed[i][0] for p in parsed[j][0])
    matrix=[[compare(i,j) for j in range(len(parsed))] for i in range(len(parsed))]
    return dict(valid=valid,equivalence=matrix,parsed_types=[[type(x).__name__ for x in row[0]] for row in parsed],
        certificate_policy=POLICY['identity'],certified_pair_rejections=certificates)


def main():
    import socket
    def blocked(*args,**kwargs):raise RuntimeError('MATH_WORKER_NETWORK_FORBIDDEN')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=socket.getaddrinfo=blocked
    try:
        native.initialize();request=json.load(sys.stdin)
        if request['policy']!=POLICY:raise ValueError('MATH_CERTIFICATE_POLICY_CHANGED')
        json.dump(evaluate(request['expressions']),sys.stdout);return 0
    except Exception:
        json.dump(dict(error='MATH_EVALUATOR_FAILURE'),sys.stdout);return 1


if __name__=='__main__':raise SystemExit(main())
