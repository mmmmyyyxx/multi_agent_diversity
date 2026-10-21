"""Fresh scoped client; timeout retains a private stack instead of a wrong score."""
from functools import lru_cache
import json,subprocess,sys
from .math_domain_v2 import SETTINGS
from .math_power_certificate import POLICY
from ..search.schemas import SearchContractError


@lru_cache(maxsize=2048)
def power_matrix(expressions):
    raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')
