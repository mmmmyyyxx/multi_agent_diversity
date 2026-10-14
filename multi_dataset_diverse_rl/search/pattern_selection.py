"""Model-free scoring and selection; semantic partitioning has no ranking authority."""
from dataclasses import asdict
import random
from typing import Protocol
from .. import current_contract as identities
from .parent_population import digest
from .pattern_primitives import PatternResponsibilitySignal, sample_labels
from .weighted_responsibility import LABELS
from .schemas import SearchContractError


class PatternScorer(Protocol):
    identity: str
    def score(self, patterns, universe): ...


class PatternSelector(Protocol):
    identity: str
    def select(self, patterns, *, seed, state_id, ordinal): ...


class WeightedSupportScorer:
    identity=identities.PATTERN_RESPONSIBILITY_VERSION
    def score(self, patterns, universe):
        result=[]
        for p in patterns:
            counts=[sum(label in sample_labels(universe[x]) for x in p['support_ids']) for label in LABELS]
            uncovered=sum(not sample_labels(universe[x]) for x in p['support_ids'])
            signal=PatternResponsibilitySignal(p['pattern_id'],*counts[:3],len(p['support_ids']),
                r3_count=counts[3],r4_count=counts[4],uncovered_count=uncovered)
            result.append({**p,'responsibility':{**asdict(signal),'raw_value':signal.raw_value}})
        return result


class WeightedResponsibilitySelector:
    identity=identities.RAW_RESPONSIBILITY_VALUE_VERSION
    def select(self, patterns, *, seed, state_id, ordinal):
        positive=[p for p in patterns if p['responsibility']['raw_value']>0]
        if positive:
            return min(positive,key=lambda p:(-p['responsibility']['raw_value'],p['pattern_id'])), 'POSITIVE_RESPONSIBILITY'
        return random.Random(f'{seed}:{state_id}:{ordinal}:pattern-fallback-v2').choice(
            sorted(patterns,key=lambda p:p['pattern_id'])), 'SEEDED_ZERO_RESPONSIBILITY_FALLBACK'


class LargestUniqueSupportSelector:
    """Independent study focus: no team labels, Vote or responsibility fallback."""
    identity='LARGEST_UNIQUE_SUPPORT_SEED_ROUND_V1'
    def select(self, patterns, *, seed, state_id='', ordinal):
        if not patterns:raise SearchContractError('PATTERN_SELECTION_REQUIRES_PARTITION')
        maximum=max(len(set(p['support_ids'])) for p in patterns)
        ties=sorted(p['pattern_id'] for p in patterns if len(set(p['support_ids']))==maximum)
        chosen=random.Random(int(digest([seed,ordinal,'largest_pattern_tie_v1']),16)).choice(ties)
        return next(p for p in patterns if p['pattern_id']==chosen),ties
