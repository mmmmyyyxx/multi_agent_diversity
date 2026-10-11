"""Frozen engineering-only progression, independent of Canary task accuracy."""
from ..search.schemas import SearchContractError
from ..persistence.durable_io import read_json

IDENTITY='INTEGRITY_COMPLETE_CANARY_TO_FROZEN_FORMAL_V1'


def require_policy(value):
    if (not isinstance(value,dict) or set(value)!={'identity','canary_prep','canary_run',
            'canary_binding_path','canary_binding_sha256','canary_attempt_id'}
            or value['identity']!=IDENTITY or
            any(not isinstance(value[k],str) or not value[k] for k in value)):
        raise SearchContractError('FORMAL_CANARY_PREREQUISITE_NOT_FROZEN')


def verify(root,contract,payload):
    from .unified_execution import validate_prep
    from .postfreeze_validation import closed_search,local_path
    value=contract.get('canary_prerequisite');require_policy(value)
    canary=validate_prep(root,local_path(root,value['canary_prep']))
    ref=canary['manifest']['execution_binding']
    if (ref['path']!=value['canary_binding_path'] or ref['sha256']!=value['canary_binding_sha256']
            or canary['scope']['attempt_id']!=value['canary_attempt_id']
            or canary['scope']['phase']!='canary_search_only'
            or canary['manifest']['source_sha']!=payload['manifest']['source_sha']):
        raise SearchContractError('CANARY_FORMAL_SOURCE_OR_SCOPE_MISMATCH')
    receipt=closed_search(root,value['canary_run'],canary)
    run=local_path(root,value['canary_run']);summary=read_json(run/'execution_summary.json')
    if (receipt['scientific_complete'] is not True or receipt['stop_reason'] not in {
            'CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE','NO_REPAIR_SIGNAL'}
            or summary['ledger']['solver']<=0 or summary['accounting']['reserved_inflight']!=0
            or not (run/'INITIAL_SOLVER_PROFILE.review_pass.json').is_file()
            or (summary['result']['trace'] and not (run/'FIRST_COMPLETE_OPPORTUNITY.review_pass.json').is_file())):
        raise SearchContractError('CANARY_ENGINEERING_PROGRESSION_HOLD')
    return dict(identity=IDENTITY,canary_closure=receipt,efficacy_gate=False)
