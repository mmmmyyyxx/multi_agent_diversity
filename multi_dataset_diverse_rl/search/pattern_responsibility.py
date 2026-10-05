"""Target-first set-level failure discovery; algorithmic WHAT, never WHO."""
from dataclasses import asdict, dataclass, replace
import json
import math
import re

from .. import versions
from ..evaluation.semantic_mutable_contract import semantic_violation_reasons
from ..local_optimizers.gepa_adapter import contains_supplied_example_text
from ..local_optimizers.schemas import LocalEvidenceExample
from .responsibility_value import responsibility_value
from .schemas import EvidenceView, SearchContractError
from .semantic_contract import mechanism_identity
from .variable_evidence import PatternCapableVariableEvidencePolicyV1
from .abstraction_content import specific_content_leaked


PROMPT = '''Discover generalizable recurring mathematical failure mechanisms in the ENTIRE
supplied set of incorrect selected-member Optimize examples, across all responsibility lanes.
Return one JSON object: {"patterns":[{"failure_mechanism":"generic mechanism",
"update_direction":"generic corrective reasoning principle","support_ids":["supplied id"]}],
"unassigned_ids":["remaining supplied id"]}. Supports must be nonempty and disjoint. All supplied
IDs must occur exactly once, as support or unassigned. A singleton mechanism is allowed when
justified; leave unsupported abstractions unassigned. Do not rank patterns or members, assign
importance, choose a focus or budget, rewrite a complete procedure, specify a response interface,
copy example facts, named entities, numeric constants or answers. No confidence is required.
Discovery describes structure only; the controller computes responsibility and selects a focus.'''

POLICY = dict(discovery=versions.PATTERN_AWARE_DISCOVERY_VERSION,
    responsibility=versions.PATTERN_RESPONSIBILITY_VERSION,
    raw_value=versions.RAW_RESPONSIBILITY_VALUE_VERSION,
    schema=versions.PATTERN_DISCOVERY_SCHEMA_VERSION,
    correctness=versions.TARGET_CORRECTNESS_SIGNAL_VERSION,
    universe='all_selected_member_wrong_optimize', calls_per_opportunity=1,
    selection='raw_responsibility_then_canonical_identity', input_limit_tokens=991808,
    context_limit_tokens=1000000, input_estimate='serialized_utf8_bytes_plus_1024',
    context_source='https://help.aliyun.com/zh/model-studio/qwen3-7-flash')

ABSTRACTION_GUARD_VERSIONS = (None, versions.PATTERN_ABSTRACTION_GUARD_VERSION,
    versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)


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


def discovery_payload(rows):
    return dict(schema=versions.PATTERN_DISCOVERY_SCHEMA_VERSION, examples=[dict(
        example_id=r.example_id, problem=r.signals['input_payload'], reference=r.signals['gold'],
        prediction=r.signals.get('target_output'), valid=r.signals['target_member_valid'],
        responsibility_labels=sample_labels(r), team_margin=r.signals.get('team_margin'),
        team_disagreement=r.signals.get('team_disagreement')) for r in rows])


def guard_abstraction(text, rows, *, abstraction_guard_version=None):
    if abstraction_guard_version not in ABSTRACTION_GUARD_VERSIONS:
        raise SearchContractError('PATTERN_ABSTRACTION_GUARD_NOT_BOUND')
    if (not isinstance(text,str) or not text.strip() or len(text)>600
            or re.search(r'\d',text) or semantic_violation_reasons(text)):
        raise SearchContractError('PATTERN_DISCOVERY_INVALID_ABSTRACTION')
    examples=tuple(LocalEvidenceExample(r.example_id,r.signals['input_payload'],r.signals['gold']) for r in rows)
    if contains_supplied_example_text(text,examples):
        raise SearchContractError('PATTERN_DISCOVERY_EXAMPLE_LEAKAGE')
    if abstraction_guard_version == versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION:
        if specific_content_leaked(text, rows):
            raise SearchContractError('PATTERN_DISCOVERY_EXAMPLE_LEAKAGE')
        return
    # Reject copied multi-character answers and example-specific proper names.
    normalized=' '.join(text.casefold().split())
    for r in rows:
        gold=str(r.signals['gold']).strip().casefold()
        if gold and (normalized==gold or len(gold)>=3 and gold in normalized):
            raise SearchContractError('PATTERN_DISCOVERY_EXAMPLE_LEAKAGE')
        names=re.findall(r'(?<!\w)[A-Z][a-z]{2,}(?!\w)',str(r.signals['input_payload']))
        generic={'Find','Compute','Determine','Let','Suppose','Given','What','How','The','For','When','If','Math'}
        if abstraction_guard_version==versions.PATTERN_ABSTRACTION_GUARD_VERSION:
            # A mathematical imperative remains generic when capitalized at
            # the beginning of an example. Do not classify it as a proper name.
            generic.add('Simplify')
        if any(name not in generic and re.search(r'\b'+re.escape(name.casefold())+r'\b',normalized) for name in names):
            raise SearchContractError('PATTERN_DISCOVERY_EXAMPLE_LEAKAGE')


def score_partition(value, rows, *, abstraction_guard_version=None):
    universe={r.example_id:r for r in rows}
    if (not isinstance(value,dict) or set(value)!={'patterns','unassigned_ids'}
            or not isinstance(value['patterns'],list) or not isinstance(value['unassigned_ids'],list)):
        raise SearchContractError('PATTERN_DISCOVERY_INVALID')
    used=set(); grouped={}
    for p in value['patterns']:
        if (not isinstance(p,dict) or not {'failure_mechanism','update_direction','support_ids'}<=set(p)
                or set(p)-{'failure_mechanism','update_direction','support_ids','confidence','importance','priority','ranking'}
                or not isinstance(p['support_ids'],list) or not p['support_ids']
                or any(not isinstance(x,str) for x in p['support_ids'])):
            raise SearchContractError('PATTERN_DISCOVERY_INVALID')
        support=set(p['support_ids'])
        if len(support)!=len(p['support_ids']) or not support<=set(universe) or used & support:
            raise SearchContractError('PATTERN_DISCOVERY_INVALID_MEMBERSHIP')
        used.update(support)
        for key in ('failure_mechanism','update_direction'):guard_abstraction(p[key],rows,abstraction_guard_version=abstraction_guard_version)
        mid=mechanism_identity(p['failure_mechanism'],p['update_direction'])
        if mid not in grouped:
            grouped[mid]=dict(pattern_id=mid,failure_mechanism=p['failure_mechanism'].strip(),
                corrective_principle=p['update_direction'].strip(),support_ids=(),counterexample_ids=(),risk_ids=())
        grouped[mid]['support_ids']=tuple(sorted(set(grouped[mid]['support_ids'])|support))
    unassigned=value['unassigned_ids']
    if (any(not isinstance(x,str) for x in unassigned) or len(set(unassigned))!=len(unassigned)
            or used & set(unassigned) or used | set(unassigned)!=set(universe)):
        raise SearchContractError('PATTERN_DISCOVERY_INVALID_MEMBERSHIP')
    active=[]
    for p in grouped.values():
        counts={k:sum(k in sample_labels(universe[x]) for x in p['support_ids']) for k in ('direct_flip','near_margin','coverage')}
        signal=PatternResponsibilitySignal(p['pattern_id'],counts['direct_flip'],counts['near_margin'],counts['coverage'],len(p['support_ids']))
        active.append({**p,'responsibility':{**asdict(signal),'raw_value':signal.raw_value}})
    if not active:raise SearchContractError('PATTERN_DISCOVERY_NOT_ACTIONABLE')
    focus=min(active,key=lambda p:(-p['responsibility']['raw_value'],p['pattern_id']))
    if focus['responsibility']['raw_value']==0:
        raise SearchContractError('PATTERN_RESPONSIBILITY_COVERAGE_FAILURE')
    assigned=len(used);total=len(rows);n=len(focus['support_ids'])
    weights=[len(p['support_ids'])/assigned for p in active]
    entropy=-sum(w*math.log(w) for w in weights)/math.log(len(weights)) if len(weights)>1 else 0.0
    return dict(patterns=tuple(sorted(active,key=lambda p:p['pattern_id'])),
        dominant_pattern_id=focus['pattern_id'],focus_mechanism_id=focus['pattern_id'],
        selected_pattern_responsibility=focus['responsibility']['raw_value'],
        selection_tiebreak_used=sum(p['responsibility']['raw_value']==focus['responsibility']['raw_value'] for p in active)>1,
        total_residual_count=total,assigned_residual_count=assigned,support_count=n,
        unassigned_residual_ids=tuple(sorted(unassigned)),DPR_all=n/total,ConditionalDPR=n/assigned,
        Coverage=assigned/total,dominant_pattern_ratio=n/total,normalized_entropy=entropy,mixed_pattern_count=len(active),
        selection_authority=versions.RAW_RESPONSIBILITY_VALUE_VERSION,
        target_selected_before_pattern=True,pattern_memory_access_before_selection=False,
        wrong_universe_size=total,valid_wrong_count=sum(r.signals['target_member_valid'] for r in rows),
        invalid_wrong_count=sum(not r.signals['target_member_valid'] for r in rows),
        wrong_by_lane={k:sum(k in sample_labels(r) for r in rows) for k in ('direct_flip','near_margin','coverage')},
        zero_responsibility_wrong_count=sum(not sample_labels(r) for r in rows))


class ResponsibilityPatternDiscoveryV3:
    identity=versions.PATTERN_AWARE_DISCOVERY_VERSION

    def __init__(self,provider,*,abstraction_guard_version=None):
        if provider is None:raise SearchContractError('PATTERN_PROVIDER_NOT_BOUND')
        if abstraction_guard_version not in ABSTRACTION_GUARD_VERSIONS:
            raise SearchContractError('PATTERN_ABSTRACTION_GUARD_NOT_BOUND')
        self.provider=provider
        self.abstraction_guard_version=abstraction_guard_version
        self.attempted_opportunities=set()

    def analyze(self,state,diagnosis,target_member,evidence_rows,history):
        key=(state.team_state_id,target_member,history.target_counts.get(target_member,0))
        if key in self.attempted_opportunities:
            raise SearchContractError('PATTERN_ONE_SUCCESS_PER_OPPORTUNITY')
        rows=wrong_universe(evidence_rows)
        if not rows:raise SearchContractError('PATTERN_DISCOVERY_NOT_ACTIONABLE')
        self.attempted_opportunities.add(key)
        return score_partition(self.provider.diagnose(discovery_payload(rows)),rows,abstraction_guard_version=self.abstraction_guard_version)


class SetLevelPatternProvider:
    def __init__(self,broker,prompt= PROMPT):
        self.broker=broker;self.prompt=prompt;self.calls=0;self.input_audit=[]

    def diagnose(self,payload):
        from ..governance.token_accounting import serialized_request
        messages=[dict(role='system',content=self.prompt),dict(role='user',content=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True))]
        request,_=self.broker._request_identity(role='pattern',split='optimize',messages=messages)
        estimate=len(serialized_request(request))+1024
        audit=dict(wrong_universe_size=len(payload['examples']),estimated_input_tokens=estimate,
            provider_input_limit=POLICY['input_limit_tokens'],provider_context_limit=POLICY['context_limit_tokens'])
        self.input_audit.append(audit)
        if estimate>POLICY['input_limit_tokens'] or estimate+1810>POLICY['context_limit_tokens']:
            raise SearchContractError('STOP_PATTERN_CONTEXT_LIMIT_POLICY_REQUIRED')
        self.calls+=1
        result=self.broker.complete(role='pattern',split='optimize',stage='set_level_pattern_discovery',messages=messages)
        try:return json.loads(result['text'])
        except (ValueError,TypeError):raise SearchContractError('PATTERN_DISCOVERY_INVALID') from None


class PatternConditionedEvidenceV4(PatternCapableVariableEvidencePolicyV1):
    identity=versions.PATTERN_CONDITIONED_EVIDENCE_VERSION

    def compose(self,state,diagnosis,member_id,rows,pattern_context=None):
        if any(r.source_split!='optimize' for r in rows):raise SearchContractError('PATTERN_HELDOUT_ACCESS')
        context=pattern_context or {}
        focus=next((p for p in context.get('patterns',()) if p['pattern_id']==context.get('focus_mechanism_id')),None)
        if focus is None:raise SearchContractError('PATTERN_DISCOVERY_NOT_ACTIONABLE')
        support=set(focus['support_ids']); byid={r.example_id:r for r in rows}
        if not support or not support<=set(byid):raise SearchContractError('PATTERN_SUPPORT_MAPPING_INVALID')
        ordered=sorted((byid[x] for x in support),key=lambda r:(-responsibility_value(*(int(k in sample_labels(r)) for k in ('direct_flip','near_margin','coverage'))),r.example_id))
        repairs=[replace(r,roles=frozenset({'REPAIR','FOCUS_REPAIR'}|set(sample_labels(r))|({'TRANSITION_FOCUS','TRANSITION_ANCHOR'}&r.roles)),
            signals={**r.signals,'legacy_tags':('repair','focus_repair_v2'),
                'feedback':'Repair only the selected semantic mechanism.'}) for r in ordered]
        safety=[r for r in rows if r.example_id not in support and
            (r.signals.get('target_member_correct') is True or {'TRANSITION_FOCUS','TRANSITION_ANCHOR'} & r.roles)]
        safety=sorted(safety,key=lambda r:(not r.signals.get('target_member_correct'),
            not bool(r.signals.get('mutation_sensitive')),-r.signals.get('team_disagreement',0),r.signals.get('team_margin',0),r.example_id))
        safety=[replace(r,roles=(r.roles-{'REPAIR','TEAM_HARD','direct_flip','near_margin','coverage','pure_coverage'})|{'SAFETY_BOUNDARY'},
            signals={**r.signals,'legacy_tags':('safety_boundary_v2',),'feedback':'Safety only: preserve competence; no second repair objective.'}) for r in safety]
        # The preservation anchor and recent transition safeguards have distinct
        # slots. Extra correct examples cannot displace transition context.
        chosen=repairs[:3]
        preservation=next((r for r in safety if r.signals.get('target_member_correct')),None)
        if preservation is not None:chosen.append(preservation)
        for role in ('TRANSITION_FOCUS','TRANSITION_ANCHOR'):
            boundary=next((r for r in safety if role in r.roles),None)
            if boundary is not None and boundary.example_id not in {r.example_id for r in chosen}:chosen.append(boundary)
        if len(chosen)<self.minimum:
            chosen+=[r for r in repairs[3:]+safety if r.example_id not in {x.example_id for x in chosen}][:self.minimum-len(chosen)]
        if len(chosen)<self.minimum:
            raise SearchContractError('FOCUSED_BACKEND_MINIMUM_WITHOUT_LEGAL_BOUNDARIES')
        chosen=tuple(chosen[:6])
        view=EvidenceView(tuple(replace(r,roles=r.roles|{'MUTATION'}) for r in chosen),
            tuple(replace(r,roles=r.roles|{'SEARCH_VALIDATION'}) for r in chosen),
            tuple(replace(r,roles=r.roles|{'TEAM_PROBE'}) for r in chosen),'optimize_full','shadow_adaptive')
        return view,dict(pattern_enabled=True,focus_pattern_id=focus['pattern_id'],
            representative_pattern_examples=sum('REPAIR' in r.roles for r in chosen),
            preservation_anchor_count=sum(r.signals.get('target_member_correct') is True for r in chosen),
            transition_evidence_count=sum(bool({'TRANSITION_FOCUS','TRANSITION_ANCHOR'}&r.roles) for r in chosen),
            nonfocus_repair_count=0,generic_backfill_count=0,mutation_count=len(chosen),
            search_validation_count=len(chosen),team_probe_count=len(chosen),pattern_selection_authority=versions.RAW_RESPONSIBILITY_VALUE_VERSION)
