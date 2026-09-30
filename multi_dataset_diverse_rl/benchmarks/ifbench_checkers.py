"""The pinned checker code is frozen; resource/package identities are not yet frozen."""
from ..search.schemas import SearchContractError


def load_pinned_checkers():
    # Never import upstream automatic-download modules or silently enable an
    # unverified local resource installation. A later dependency freeze must
    # bind package/model/resource hashes before changing this runtime gate.
    raise SearchContractError("IFBENCH_LOCAL_EVALUATOR_DEPENDENCIES_NOT_FROZEN")
