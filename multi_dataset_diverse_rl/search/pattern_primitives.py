"""Shared wrong-universe, labels, raw F and current abstraction checks."""
from dataclasses import dataclass
import re
from .. import current_contract as versions
from ..evaluation.semantic_mutable_contract import semantic_violation_reasons
from ..local_optimizers.example_text import contains_supplied_example_text
from ..local_optimizers.schemas import LocalEvidenceExample
from .responsibility_value import responsibility_value
from .schemas import SearchContractError
from .abstraction_content import current_specific_content_leaked

@dataclass(frozen=True)
class PatternResponsibilitySignal:
    pattern_id: str
    direct_count: int
    near_margin_count: int
    coverage_count: int
    support_count: int
    identity: str = versions.PATTERN_RESPONSIBILITY_VERSION

    @property
    def raw_value(self):
        return responsibility_value(self.direct_count, self.near_margin_count, self.coverage_count)

def wrong_universe(rows):
    rows = tuple(rows)
    if any(r.source_split != 'optimize' for r in rows):
        raise SearchContractError('PATTERN_HELDOUT_ACCESS')
    if len({r.example_id for r in rows}) != len(rows):
        raise SearchContractError('PATTERN_UNIVERSE_DUPLICATE')
    for r in rows:
        s = r.signals
        if (s.get('correctness_signal_identity') != versions.TARGET_CORRECTNESS_SIGNAL_VERSION
                or type(s.get('target_member_correct')) is not bool
                or type(s.get('target_member_valid')) is not bool
                or s['target_member_correct'] and not s['target_member_valid']):
            raise SearchContractError('PATTERN_CORRECTNESS_SIGNAL_REQUIRED')
    return tuple(r for r in rows if not r.signals['target_member_correct'])

def sample_labels(row):
    labels = row.signals.get('responsibility_labels')
    if (not isinstance(labels, (tuple, list)) or len(set(labels)) != len(labels)
            or not set(labels) <= {'direct_flip','near_margin','coverage'}):
        raise SearchContractError('PATTERN_RESPONSIBILITY_LABELS_REQUIRED')
    return tuple(labels)

def guard_abstraction(text, rows, *, abstraction_guard_version=None):
    if abstraction_guard_version != versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION:
        raise SearchContractError('PATTERN_ABSTRACTION_GUARD_NOT_BOUND')
    if (not isinstance(text,str) or not text.strip() or len(text)>600
            or re.search(r'\d',text) or semantic_violation_reasons(text)):
        raise SearchContractError('PATTERN_DISCOVERY_INVALID_ABSTRACTION')
    examples=tuple(LocalEvidenceExample(r.example_id,r.signals['input_payload'],r.signals['gold']) for r in rows)
    if contains_supplied_example_text(text,examples):
        raise SearchContractError('PATTERN_DISCOVERY_EXAMPLE_LEAKAGE')
    if abstraction_guard_version == versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION:
        if current_specific_content_leaked(text, rows):
            raise SearchContractError('PATTERN_DISCOVERY_EXAMPLE_LEAKAGE')
        return

def single_failure_example(row):
    return dict(example_id=row.example_id,problem=row.signals['input_payload'],reference=row.signals['gold'],prediction=row.signals.get('target_output'),valid=row.signals['target_member_valid'],responsibility_labels=sample_labels(row),team_margin=row.signals.get('team_margin'),team_disagreement=row.signals.get('team_disagreement'))
