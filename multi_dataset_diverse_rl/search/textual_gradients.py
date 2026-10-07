"""Fresh per-failure corrective signals, then a gradients-only semantic partition.

Signals are hypotheses from observable evidence, never latent-cause witnesses.
The LLM groups corrections; the unchanged shared responsibility primitive ranks.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import re
import unicodedata

from .. import current_contract as versions
from .pattern_primitives import (guard_abstraction, wrong_universe, sample_labels,
    single_failure_example, PatternResponsibilitySignal)
from .selected_evidence import compose_selected_evidence
from .schemas import SearchContractError


GRADIENT_PROMPT = '''Return exactly {"gradient":"one corrective reasoning instruction"}.
HARD LIMIT: the gradient string must contain at most 400 characters, including spaces.
Prefer one or two short sentences totaling at most 250-300 characters.
Infer one hypothetical, actionable prompt-level correction from this single observable
failure and the current member procedure. State one reasoning behavior to strengthen,
change, add or avoid, directly usable as a reasoning instruction on other problems.
Use abstract roles and operations. Do not copy problem-specific numeric constants,
answers, quantities, entities, formula fragments, or other example-specific details.
Generic mathematical constants or structural quantities are allowed only when
necessary to state a reusable reasoning rule.
Do not explain this particular example or claim proof of its hidden causal error.
If the root cause is uncertain, state a conservative generalizable corrective behavior.
Before output, ensure the string is short, abstract and actionable; omit example details.
Do not write a full replacement procedure or candidate prompt, or control the external
response interface, presentation or markers. Do not infer other examples, members,
memory or held-out data. Output only the JSON object with the single gradient key.'''

CLUSTER_PROMPT = '''Compare the entire gradient set before forming clusters.
Group gradients when they express the same reusable corrective reasoning behavior,
even when their underlying mathematical questions differ. Merge paraphrastic or
semantically equivalent corrections. Do not form separate groups merely because of
mathematical topics or surface forms. Avoid unnecessary singleton fragmentation;
use a singleton only when its corrective behavior is distinct. Do not invent a shared
correction to force a group and do not impose any fixed number or minimum cluster size.
Each cluster summarizes its members into one generalized textual gradient, at most
600 characters. Return only {"patterns":[{"generalized_gradient":"correction",
"support_ids":["supplied id"]}],"unassigned_ids":[]}. Supports are nonempty,
disjoint and contain supplied IDs only. Every ID occurs exactly once in support or
unassigned. Leave unsupported abstractions unassigned. Do not rank clusters, report
importance/confidence, choose a focus or rewrite a complete procedure. Do not include
answers, constants, entities, example fragments or response-interface instructions.
The controller computes responsibility and selects which correction receives budget.'''

GRADIENT_POLICY = dict(identity=versions.PER_EXAMPLE_GRADIENT_VERSION,
    prompt_identity=versions.GRADIENT_PROMPT_VERSION,
    schema='PER_EXAMPLE_TEXTUAL_GRADIENT_SCHEMA_V1', max_characters=versions.GRADIENT_MAX_CHARACTERS,
    universe='all_selected_member_wrong_optimize', logical_calls_per_wrong=1,
    successful_generations_per_wrong=1, output_cache=False, semantic_regeneration=False,
    provider_role='pattern_gradient', generation_contract_role='pattern',
    generation_policy=versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION,
    abstraction_guard=versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)
POLICY = dict(discovery=versions.GRADIENT_PATTERN_DISCOVERY_VERSION,
    responsibility=versions.PATTERN_RESPONSIBILITY_VERSION, raw_value=versions.RAW_RESPONSIBILITY_VALUE_VERSION,
    schema=versions.GRADIENT_PATTERN_SCHEMA_VERSION, correctness=versions.TARGET_CORRECTNESS_SIGNAL_VERSION,
    universe='all_selected_member_wrong_optimize', gradient_policy=GRADIENT_POLICY,
    cluster_calls_per_opportunity=1, provider_role='pattern_cluster', output_cache=False,
    generation_contract_role='pattern', generation_policy=versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION,
    identity=versions.GRADIENT_PATTERN_IDENTITY_VERSION,
    selection='raw_responsibility_then_canonical_identity',
    generalized_gradient_max_characters=versions.GENERALIZED_GRADIENT_MAX_CHARACTERS,
    accounting='wrong_universe_ceiling_plus_one_cluster_per_opportunity_v1',
    input_limit_tokens=991808, context_limit_tokens=1000000,
    input_estimate='serialized_utf8_bytes_plus_1024')


def gradient_identity(text):
    normalized=' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    if not normalized:raise SearchContractError('EMPTY_GENERALIZED_GRADIENT')
    return hashlib.sha256(json.dumps([versions.GRADIENT_PATTERN_IDENTITY_VERSION,normalized],
        ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def validate_gradient(text, rows, *, generalized=False):
    limit=versions.GENERALIZED_GRADIENT_MAX_CHARACTERS if generalized else versions.GRADIENT_MAX_CHARACTERS
    if not isinstance(text,str) or not text.strip() or len(text)>limit:
        raise SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID' if not generalized else 'PATTERN_GRADIENT_CLUSTER_INVALID')
    # Complete-role/replacement declarations are incompatible with a local signal.
    # This bounded check complements the shared content/interface guard; it is not
    # a claim to classify every possible semantic full-procedure paraphrase.
    if re.search(r'\b(?:(?:you are|act as) (?:an? )?(?:expert |math(?:ematical)? )?(?:solver|assistant)|complete (?:replacement |solver )?(?:prompt|procedure)|'
            r'(?:replace|rewrite) (?:the |your )?(?:entire|complete|full) (?:prompt|procedure)|'
            r'(?:solve|answer) (?:every|all) (?:math(?:ematical)? )?(?:problem|question)s?)\b',text,re.I):
        raise SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID' if not generalized else 'PATTERN_GRADIENT_CLUSTER_INVALID')
    try:guard_abstraction(text,rows,abstraction_guard_version=versions.PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)
    except SearchContractError:
        raise SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID' if not generalized else 'PATTERN_GRADIENT_CLUSTER_INVALID') from None
    return text.strip()


class _FreshSearchProvider:
    role=None
    stage=None
    def __init__(self,broker,prompt):
        self.broker=broker;self.prompt=prompt;self.calls=0;self.input_audit=[]

    def _call(self,payload):
        from ..governance.token_accounting import serialized_request
        messages=[dict(role='system',content=self.prompt),dict(role='user',content=json.dumps(payload,
            sort_keys=True,separators=(',',':'),ensure_ascii=True))]
        request,_=self.broker._request_identity(role=self.role,split='optimize',messages=messages)
        estimate=len(serialized_request(request))+1024
        self.input_audit.append(dict(role=self.role,estimated_input_tokens=estimate,
            provider_input_limit=POLICY['input_limit_tokens'],provider_context_limit=POLICY['context_limit_tokens']))
        if estimate>POLICY['input_limit_tokens'] or estimate+1810>POLICY['context_limit_tokens']:
            raise SearchContractError('STOP_PATTERN_CONTEXT_LIMIT_POLICY_REQUIRED')
        self.calls+=1
        result=self.broker.complete(role=self.role,split='optimize',stage=self.stage,messages=messages)
        try:return json.loads(result['text'])
        except (ValueError,TypeError):
            raise SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID' if self.role=='pattern_gradient' else 'PATTERN_GRADIENT_CLUSTER_INVALID') from None


class PerExampleGradientProvider(_FreshSearchProvider):
    role='pattern_gradient';stage='per_example_textual_gradient'
    def __init__(self,broker,prompt=GRADIENT_PROMPT,*,numeric_guard_writer=None):
        super().__init__(broker,prompt);self.numeric_guard_writer=numeric_guard_writer
    def extract(self,payload):
        expected={'example_id','problem','reference','prediction','valid','responsibility_labels','team_margin','team_disagreement'}
        if (not isinstance(payload,dict) or set(payload)!={'schema','current_member_procedure','example'}
                or payload['schema']!=GRADIENT_POLICY['schema']
                or not isinstance(payload['current_member_procedure'],str)
                or not payload['current_member_procedure'].strip()
                or not isinstance(payload['example'],dict) or set(payload['example'])!=expected):
            raise SearchContractError('PATTERN_GRADIENT_INPUT_INVALID')
        return self._call(payload)


class GradientClusterProvider(_FreshSearchProvider):
    role='pattern_cluster';stage='set_level_gradient_clustering'
    support_id_transport=versions.PATTERN_SUPPORT_ID_ALIAS_VERSION
    def __init__(self,broker,prompt=CLUSTER_PROMPT,*,gradient_provider=None,
            partition_completion_policy=None, partition_writer=None):
        super().__init__(broker,prompt)
        self.gradient_provider=gradient_provider
        if partition_completion_policy not in (None, versions.GRADIENT_PARTITION_COMPLETION_VERSION):
            raise SearchContractError('PATTERN_PARTITION_COMPLETION_POLICY_MISMATCH')
        self.partition_completion_policy=partition_completion_policy
        self.partition_writer=partition_writer
        self.partition_audit=[]
    def cluster(self,payload,*,evidence_rows=None):
        if not isinstance(payload,dict) or set(payload)!={'gradients'}:
            raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INPUT_INVALID')
        validate_records(payload['gradients'])
        # Exact aliases carry no source metadata, lane, score or problem topic.
        wire=deepcopy(payload);mapping={}
        for i,row in enumerate(wire['gradients'],1):
            alias=f'e{i}';mapping[alias]=row['example_id'];row['example_id']=alias
        if self.partition_completion_policy is not None:
            rows=tuple(evidence_rows or ())
            if (wrong_universe(rows)!=rows or len(rows)!=len(mapping)
                    or {r.example_id for r in rows}!=set(mapping.values())):
                raise SearchContractError('PATTERN_PARTITION_COMPLETION_PROVENANCE_REQUIRED')
        value=self._call(wire)
        if self.partition_completion_policy is not None:
            from .partition_completion import complete_known_alias_partition
            raw=deepcopy(value)
            try:
                value,audit=complete_known_alias_partition(raw,tuple(mapping),
                    validate_generalized=lambda text:validate_gradient(text,rows,generalized=True))
            except SearchContractError as exc:
                audit=getattr(exc,'partition_completion_audit',None)
                if audit is not None:self._record_partition(raw,None,audit)
                raise
            self._record_partition(raw,value,audit)
        if not isinstance(value,dict) or set(value)!={'patterns','unassigned_ids'} or not isinstance(value['patterns'],list):
            raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INVALID')
        def decode(ids):
            if not isinstance(ids,list) or any(not isinstance(x,str) or x not in mapping for x in ids):
                raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INVALID_MEMBERSHIP')
            return [mapping[x] for x in ids]
        for p in value['patterns']:
            if not isinstance(p,dict) or 'support_ids' not in p:raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INVALID')
            p['support_ids']=decode(p['support_ids'])
        value['unassigned_ids']=decode(value['unassigned_ids'])
        return value

    def _record_partition(self,raw,normalized,audit):
        # Persist before decode/scoring. A failed durable write remains a hard
        # operational failure; it never authorizes an in-attempt second draw.
        if self.partition_writer is not None:
            self.partition_writer(dict(cluster_index=self.calls,raw_partition=deepcopy(raw),
                normalized_controller_partition=deepcopy(normalized),audit=deepcopy(audit)))
        self.partition_audit.append(deepcopy(audit))


class GradientExtractor:
    def __init__(self,provider):
        if provider is None:raise SearchContractError('PATTERN_GRADIENT_PROVIDER_NOT_BOUND')
        self.provider=provider;self.numeric_guard_audit=[]
    def extract(self,procedure,rows):
        if not isinstance(procedure,str) or not procedure.strip():raise SearchContractError('PATTERN_MEMBER_PROCEDURE_REQUIRED')
        rows=tuple(rows)
        if wrong_universe(rows)!=rows:raise SearchContractError('PATTERN_GRADIENT_WRONG_UNIVERSE_REQUIRED')
        if not rows or len({r.example_id for r in rows})!=len(rows):
            raise SearchContractError('PATTERN_GRADIENT_WRONG_UNIVERSE_REQUIRED')
        gradients=[]
        for row in rows:
            example=single_failure_example(row)
            value=self.provider.extract(dict(schema=GRADIENT_POLICY['schema'],current_member_procedure=procedure,example=example))
            if not isinstance(value,dict) or set(value)!={'gradient'}:
                raise SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID')
            if isinstance(value['gradient'],str):
                from .numeric_provenance import numeric_guard_result
                audit=dict(example_id=row.example_id,gradient_sha256=hashlib.sha256(
                    value['gradient'].encode()).hexdigest(),numeric_guard_result=numeric_guard_result(value['gradient'],(row,)))
                self.numeric_guard_audit.append(audit)
                writer=getattr(self.provider,'numeric_guard_writer',None)
                if writer is not None:writer(audit)
            gradients.append(dict(example_id=row.example_id,gradient=validate_gradient(value['gradient'],(row,))))
        return tuple(gradients)


def validate_records(gradients):
    if (not isinstance(gradients,(list,tuple)) or not gradients
            or any(not isinstance(g,dict) or set(g)!={'example_id','gradient'}
                or not isinstance(g['example_id'],str) or not g['example_id']
                or not isinstance(g['gradient'],str) or not g['gradient'].strip()
                or len(g['gradient'])>versions.GRADIENT_MAX_CHARACTERS for g in gradients)
            or len({g['example_id'] for g in gradients})!=len(gradients)):
        raise SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID')


def score_gradient_partition(value,rows,gradients):
    rows=tuple(rows)
    if not rows or wrong_universe(rows)!=rows or len({r.example_id for r in rows})!=len(rows):
        raise SearchContractError('PATTERN_GRADIENT_WRONG_UNIVERSE_REQUIRED')
    validate_records(gradients)
    universe={r.example_id:r for r in rows}
    if len(gradients)!=len(rows) or {g['example_id'] for g in gradients}!=set(universe):
        raise SearchContractError('PATTERN_GRADIENT_EXTRACTION_INVALID')
    for g in gradients:validate_gradient(g['gradient'],(universe[g['example_id']],))
    if (not isinstance(value,dict) or set(value)!={'patterns','unassigned_ids'}
            or not isinstance(value['patterns'],list) or not isinstance(value['unassigned_ids'],list)):
        raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INVALID')
    used=set();grouped={}
    for p in value['patterns']:
        if not isinstance(p,dict) or set(p)!={'generalized_gradient','support_ids'}:
            raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INVALID')
        ids=p['support_ids']
        if (not isinstance(ids,list) or not ids or any(not isinstance(x,str) for x in ids)
                or len(set(ids))!=len(ids) or not set(ids)<=set(universe) or used & set(ids)):
            raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INVALID_MEMBERSHIP')
        used.update(ids)
        text=validate_gradient(p['generalized_gradient'],rows,generalized=True);mid=gradient_identity(text)
        if mid not in grouped:grouped[mid]=dict(pattern_id=mid,generalized_gradient=text,support_ids=())
        grouped[mid]['support_ids']=tuple(sorted(set(grouped[mid]['support_ids'])|set(ids)))
    remaining=value['unassigned_ids']
    if (any(not isinstance(x,str) for x in remaining) or len(set(remaining))!=len(remaining)
            or used & set(remaining) or used|set(remaining)!=set(universe)):
        raise SearchContractError('PATTERN_GRADIENT_CLUSTER_INVALID_MEMBERSHIP')
    patterns=[]
    for p in grouped.values():
        counts=[sum(label in sample_labels(universe[x]) for x in p['support_ids']) for label in ('direct_flip','near_margin','coverage')]
        signal=PatternResponsibilitySignal(p['pattern_id'],*counts,len(p['support_ids']))
        patterns.append({**p,'responsibility':{**asdict(signal),'raw_value':signal.raw_value}})
    if not patterns:raise SearchContractError('PATTERN_DISCOVERY_NOT_ACTIONABLE')
    focus=min(patterns,key=lambda p:(-p['responsibility']['raw_value'],p['pattern_id']))
    if not focus['responsibility']['raw_value']:raise SearchContractError('PATTERN_RESPONSIBILITY_COVERAGE_FAILURE')
    total=len(rows);assigned=len(used);size=len(focus['support_ids']);weights=[len(p['support_ids'])/assigned for p in patterns]
    return dict(patterns=tuple(sorted(patterns,key=lambda p:p['pattern_id'])),
        per_example_gradients=tuple(gradients),focus_mechanism_id=focus['pattern_id'],dominant_pattern_id=focus['pattern_id'],
        selected_pattern_responsibility=focus['responsibility']['raw_value'],
        selection_tiebreak_used=sum(p['responsibility']['raw_value']==focus['responsibility']['raw_value'] for p in patterns)>1,
        total_residual_count=total,assigned_residual_count=assigned,support_count=size,
        unassigned_residual_ids=tuple(sorted(remaining)),DPR_all=size/total,ConditionalDPR=size/assigned,
        Coverage=assigned/total,dominant_pattern_ratio=size/total,
        normalized_entropy=-sum(w*math.log(w) for w in weights)/math.log(len(weights)) if len(weights)>1 else 0.0,
        mixed_pattern_count=len(patterns),wrong_universe_size=total,
        valid_wrong_count=sum(r.signals['target_member_valid'] for r in rows),
        invalid_wrong_count=sum(not r.signals['target_member_valid'] for r in rows),
        wrong_by_lane={k:sum(k in sample_labels(r) for r in rows) for k in ('direct_flip','near_margin','coverage')},
        zero_responsibility_wrong_count=sum(not sample_labels(r) for r in rows),
        target_selected_before_pattern=True,pattern_memory_access_before_selection=False,
        selection_authority=versions.RAW_RESPONSIBILITY_VALUE_VERSION,
        gradient_logical_calls=total,gradient_physical_calls=total,gradient_cache_hits=0,
        cluster_logical_calls=1,cluster_physical_calls=1,cluster_cache_hits=0,total_pattern_meta_calls=total+1)


class GradientPatternDiscovery:
    identity=versions.GRADIENT_PATTERN_DISCOVERY_VERSION
    def __init__(self,extractor,cluster_provider):
        if extractor is None or cluster_provider is None:raise SearchContractError('PATTERN_GRADIENT_PROVIDER_NOT_BOUND')
        self.extractor=extractor;self.cluster_provider=cluster_provider;self.attempted_opportunities=set()
    def analyze(self,state,diagnosis,target_member,evidence_rows,history):
        key=(state.team_state_id,target_member,history.target_counts.get(target_member,0))
        if key in self.attempted_opportunities:raise SearchContractError('PATTERN_ONE_SUCCESS_PER_OPPORTUNITY')
        rows=wrong_universe(evidence_rows)
        if not rows:raise SearchContractError('PATTERN_DISCOVERY_NOT_ACTIONABLE')
        self.attempted_opportunities.add(key)
        gradients=self.extractor.extract(state.member_prompts[target_member],rows)
        payload={'gradients':[dict(g) for g in gradients]}
        completion=getattr(self.cluster_provider,'partition_completion_policy',None)
        value=(self.cluster_provider.cluster(payload,evidence_rows=rows) if completion is not None
            else self.cluster_provider.cluster(payload))
        context=score_gradient_partition(value,rows,gradients)
        if completion is not None:
            context['partition_completion_audit']=deepcopy(self.cluster_provider.partition_audit[-1])
        return context


class GradientPatternConditionedEvidence:
    identity=versions.GRADIENT_CONDITIONED_EVIDENCE_VERSION
    def __init__(self, *, metric_budget=36, reflection_minibatch_size=3):
        self.metric_budget=metric_budget;self.minimum=reflection_minibatch_size
    def compose(self,state,diagnosis,member_id,rows,pattern_context=None):
        context=pattern_context or {};view,audit=compose_selected_evidence(rows,context,self.minimum)
        byid={g['example_id']:g['gradient'] for g in context.get('per_example_gradients',())}
        def enrich(row):
            if 'REPAIR' not in row.roles:return row
            if row.example_id not in byid:raise SearchContractError('PATTERN_GRADIENT_TRAJECTORY_MISSING')
            trajectory=dict(problem=row.signals['input_payload'],prediction=row.signals.get('target_output'),
                reference=row.signals['gold'],per_example_gradient=byid[row.example_id],
                responsibility_labels=sample_labels(row))
            return replace(row,signals={**row.signals,'per_example_gradient':byid[row.example_id],
                'failure_trajectory':trajectory,'feedback':'Repair only the selected generalized corrective gradient.'})
        view=replace(view,mutation_evidence=tuple(map(enrich,view.mutation_evidence)),
            search_validation_evidence=tuple(map(enrich,view.search_validation_evidence)),
            team_probe_evidence=tuple(map(enrich,view.team_probe_evidence)))
        if sum('REPAIR' in r.roles for r in view.mutation_evidence)>3:
            raise SearchContractError('PATTERN_GRADIENT_REPRESENTATIVE_CAPACITY')
        return view,{**audit,'failure_trajectory_count':sum('REPAIR' in r.roles for r in view.mutation_evidence),
            'per_example_gradients_retained':True,'trajectory_is_full_reasoning_trace':False}
