"""Versioned controller completion of known omissions, without semantic inference."""
from collections import Counter
from copy import deepcopy
import hashlib
import json

from ..current_contract import GRADIENT_PARTITION_COMPLETION_VERSION
from .schemas import SearchContractError

IDENTITY = GRADIENT_PARTITION_COMPLETION_VERSION
POLICY = dict(identity=IDENTITY, admissible_defect='KNOWN_ALIAS_OMISSION_OR_INVALID_PATTERN_CONTENT',
    destination='unassigned_ids', ordering='existing_then_missing_in_input_order',
    preserve_raw_response=True, preserve_explicit_supports=True,
    semantic_assignment=False, semantic_regeneration=False, extra_provider_calls=0,
    invalid_pattern='known_support_to_unassigned_preserve_valid_patterns')


def partition_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':')).encode()).hexdigest()


def complete_known_alias_partition(raw, aliases, *, validate_generalized):
    """Validate every existing field first; append only absent supplied aliases.

    The mandatory validator enforces the unchanged generalized-gradient guard
    using controller-only provenance. No labels or scores are consulted here.
    """
    aliases = tuple(aliases)
    if (not aliases or len(set(aliases)) != len(aliases)
            or any(not isinstance(x, str) or not x for x in aliases)):
        raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INPUT_INVALID')
    audit = dict(policy=IDENTITY, raw_complete=False, completion_applied=False,
        missing_aliases=[], missing_count=0, duplicate_count=0, unknown_count=0,
        raw_partition_sha256=partition_hash(raw), normalized_partition_sha256=None,
        status='FAIL', defect=None,discarded_pattern_count=0,discarded_support_count=0)

    def reject(category, defect):
        audit['defect'] = defect
        exc = SearchContractError(category)
        exc.partition_completion_audit = deepcopy(audit)
        raise exc

    if (not isinstance(raw, dict) or set(raw) != {'patterns', 'unassigned_ids'}
            or not isinstance(raw['patterns'], list)
            or not isinstance(raw['unassigned_ids'], list)):
        reject('PATTERN_GRADIENT_CLUSTER_INVALID', 'MALFORMED_SCHEMA')
    all_ids = list(raw['unassigned_ids'])
    for pattern in raw['patterns']:
        if (not isinstance(pattern, dict)
                or 'support_ids' not in pattern
                or not isinstance(pattern['support_ids'], list)
                or not pattern['support_ids']):
            reject('PATTERN_GRADIENT_CLUSTER_INVALID', 'MALFORMED_PATTERN')
        all_ids.extend(pattern['support_ids'])
    if any(not isinstance(x, str) for x in all_ids):
        reject('PATTERN_GRADIENT_CLUSTER_INVALID_MEMBERSHIP', 'INVALID_ALIAS_TYPE')
    counts = Counter(all_ids)
    audit['duplicate_count'] = sum(v > 1 for v in counts.values())
    audit['unknown_count'] = len(set(all_ids) - set(aliases))
    audit['missing_aliases'] = [x for x in aliases if x not in counts]
    audit['missing_count'] = len(audit['missing_aliases'])
    if audit['unknown_count'] or audit['duplicate_count']:
        reject('PATTERN_GRADIENT_CLUSTER_INVALID_MEMBERSHIP', 'CONTRADICTORY_MEMBERSHIP')
    normalized = deepcopy(raw)
    normalized['patterns']=[]
    for pattern in raw['patterns']:
        try:
            if set(pattern)!={'generalized_gradient','support_ids'}:
                raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INVALID')
            validate_generalized(pattern['generalized_gradient'])
        except SearchContractError as exc:
            if str(exc)=='PATTERN_DISCOVERY_EXAMPLE_LEAKAGE':raise
            normalized['unassigned_ids'].extend(pattern['support_ids'])
            audit['discarded_pattern_count']+=1
            audit['discarded_support_count']+=len(pattern['support_ids'])
        else:normalized['patterns'].append(deepcopy(pattern))
    normalized['unassigned_ids'].extend(audit['missing_aliases'])
    audit.update(raw_complete=not audit['missing_count'],
        completion_applied=bool(audit['missing_count']), status='PASS',
        defect='INVALID_PATTERN_CONTENT' if audit['discarded_pattern_count'] else 'KNOWN_ALIAS_OMISSION_ONLY' if audit['missing_count'] else 'NONE',
        normalized_partition_sha256=partition_hash(normalized),
        normalized_membership_count=len(aliases), input_count=len(aliases))
    return normalized, audit


def completion_statistics(audits):
    return dict(cluster_calls=len(audits),
        raw_complete_partitions=sum(a['raw_complete'] for a in audits),
        raw_missing_only_partitions=sum(a['defect'] == 'KNOWN_ALIAS_OMISSION_ONLY' for a in audits),
        partition_completion_events=sum(a['completion_applied'] for a in audits),
        total_missing_aliases_completed=sum(a['missing_count'] for a in audits if a['completion_applied']),
        maximum_missing_aliases_per_call=max((a['missing_count'] for a in audits if a['completion_applied']), default=0),
        duplicate_partition_failures=sum(a['duplicate_count'] > 0 for a in audits),
        unknown_alias_failures=sum(a['unknown_count'] > 0 for a in audits))
