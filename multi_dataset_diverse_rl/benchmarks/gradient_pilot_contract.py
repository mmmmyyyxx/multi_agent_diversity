"""V2.2 Pilot is closed until a new finite operational bound is frozen.

The V2.1 integer-Vote proof lives in legacy/gradient_pilot_contract_v21.py.
Scientific no-commit epochs remain unchanged; they do not prove a resource bound.
"""
from ..search.schemas import SearchContractError

OBSERVATION_POLICY = 'PILOT_READ_ONLY_STATE_JOURNAL_V1'
PILOT_BOUND_BLOCKER = 'TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN'


def pilot_provider_bounds(parent):
    raise SearchContractError(PILOT_BOUND_BLOCKER)


def derive_current_pilot_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authorization_path, authorization_sha256):
    raise SearchContractError(PILOT_BOUND_BLOCKER)
