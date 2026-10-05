"""Ordered deployment hashes and unchanged-team Validation skip contract."""
import hashlib
from ..search.schemas import SearchContractError

def team_change_receipt(initial,final):
    if len(initial)!=5 or len(final)!=5 or any(not isinstance(p,str) for p in (*initial,*final)):
        raise SearchContractError('DEPLOYED_TEAM_CHANGE_IDENTITY_INVALID')
    a=[hashlib.sha256(p.encode()).hexdigest() for p in initial]
    b=[hashlib.sha256(p.encode()).hexdigest() for p in final]
    changed=[i for i in range(5) if a[i]!=b[i]]
    return dict(ordered_initial_prompt_hashes=a,ordered_final_prompt_hashes=b,
        changed_members=changed,changed_member_count=len(changed),team_changed=bool(changed),
        validation_status='REQUIRES_POST_SEARCH_VALIDATION' if changed else 'SKIPPED_NO_TEAM_CHANGE',
        pilot_signal='PENDING_VALIDATION' if changed else 'NOT_ESTIMATED',
        search_outcome='INTERVENTION' if changed else 'NO_INTERVENTION')
