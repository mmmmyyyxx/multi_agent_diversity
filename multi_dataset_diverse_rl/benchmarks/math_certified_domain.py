"""Explicit Test-only certified-negative worker; default benchmark is unchanged."""
from functools import lru_cache
import json,subprocess,sys
from .math_domain_v2 import SETTINGS
from .math_interval_certificate import POLICY
from ..search.schemas import SearchContractError


@lru_cache(maxsize=2048)
def certified_matrix(expressions):
    raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')
