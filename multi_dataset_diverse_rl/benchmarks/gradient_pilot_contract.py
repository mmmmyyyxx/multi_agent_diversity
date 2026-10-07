"""Current-only Pilot scope and conservative bounds.

V2.1's historical Pilot bound used the fact that every commit strictly
increased integer team Vote and therefore allowed at most N commits. V2.2
admits safe target-only progress as well, so that proof is no longer valid.
Until a new target-or-team progress bound is frozen, current Pilot derivation
fails closed before any provider call. Historical V2.1 receipts remain
unchanged and replayable through their historical binding.
"""
from copy import deepcopy

from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.schemas import SearchContractError


OBSERVATION_POLICY = 'PILOT_READ_ONLY_STATE_JOURNAL_V1'


def pilot_provider_bounds(parent):
    CURRENT_POLICY_BUNDLE.validate_contract(parent)
    n = parent['low_cost_protocol']['counts']['pilot_optimize']
    shadow = parent['low_cost_protocol']['counts']['pilot_shadow']
    if n != 60 or shadow != 40 or parent['stop_policy'] != 'team_epoch_no_commit_v1':
        raise SearchContractError('PILOT_BOUND_CONTRACT_MISMATCH')
    # V2.1 bounded accepted state changes by strict integer Vote gain. V2.2 can
    # also commit Vote-neutral target gains, so reusing that proof would make
    # the operational ceiling scientifically unsound.
    raise SearchContractError('TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN')


def derive_current_pilot_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authorization_path, authorization_sha256):
    CURRENT_POLICY_BUNDLE.validate_contract(parent)
    if parent.get('execution_phase') != 'canary' or parent.get('execution_arm') != 'A4':
        raise SearchContractError('CURRENT_PILOT_PARENT_INVALID')
    c = deepcopy(parent)
    c.update(binding_path=binding_path, execution_attempt_id=attempt,
        canary_attempt_id=None, cache_namespace=attempt, execution_phase='pilot',
        pilot_parent_binding_path=parent_path, pilot_parent_binding_sha256=parent_sha256,
        pilot_execution_authorization_path=authorization_path,
        pilot_execution_authorization_sha256=authorization_sha256,
        pilot_observation_policy=OBSERVATION_POLICY,
        continuation_authorization_sha256=authorization_sha256,
        provider_bounds=pilot_provider_bounds(parent),
        search_only_scope=dict(validation_authorized=False,test_authorized=False,
            raw_diagnostic_authorized=False,llm_judge_authorized=False,push_authorized=False))
    c['initial_competence_binding'].update(count=60,
        support_identity=c['low_cost_protocol']['membership_hashes']['pilot_optimize'])
    return c
