"""Root-snapshot group evidence and bounded auxiliary hypotheses, Optimize only."""
from copy import deepcopy
from dataclasses import asdict
import json
from .schemas import SearchContractError
from .parent_population import digest
from .generation_failures import GeneratedOutputFailure
from .diagnostic_source_checks import group_diagnostic_source_rejected
from .optimizer_output_contract import JSON_INSTRUCTION

KINDS=('wrong_peer_contrast','negative_experience','correct_peer_contrast','positive_experience')
POLICY=dict(identity='ROOT_TEAM_GROUP_DIAGNOSIS_POLICY_V1',
    input_schema='ROOT_TEAM_GROUP_DIAGNOSIS_INPUT_V1',output_schema='ROOT_TEAM_GROUP_EXPERIENCE_V1',
    instruction_identity='ROOT_TEAM_GROUP_DIAGNOSER_INSTRUCTION_V1',
    maximum_support_examples=3,maximum_draws=2,maximum_field_characters=300,
    maximum_entry_characters=900,maximum_output_characters=6000,
    maximum_input_bytes=100000,configured_context_token_ceiling=131072,
    input_reservation='serialized_utf8_bytes_plus_4096_conservative_upper_bound',
    exhaustion='GENERATION_FAILED_CONTINUE_WITH_EXPLICIT_EMPTY_EXPERIENCE',
    no_insight='NO_SUPPORTED_INSIGHT',source='verified_root_team_optimize_snapshot',
    sampling='after_frozen_parent_draws_no_rng',solver_calls=0)

INSTRUCTION='''Diagnose only the selected Pattern in the supplied Root Team snapshot.
The five complete Prompts and ordinary Solver responses are observations, not hidden
reasoning or evidence of causality. Classify peers separately for each example;
a member may be correct on one example and wrong on another. Compare wrong peers
without declaring their whole strategy wrong, and correct peers without declaring
their whole Prompt superior. A final-only response supplies no observed solution steps.
Provide NEW Pattern-relevant insights: wrong_peer_contrast, negative_experience,
correct_peer_contrast, positive_experience. Negative experience is a tentative risk,
not a forbidden edit. Separate observation, hypothesis and modification_implication.
Do not select a member/Pattern, change responsibility, evaluate candidates or write
a replacement Prompt. Never copy example-specific questions, answers or solutions.
Return exactly {"status":"SUPPORTED_INSIGHT or NO_SUPPORTED_INSIGHT",
"wrong_peer_contrast":null,"negative_experience":null,"correct_peer_contrast":null,
"positive_experience":null}. Each nonnull value has exactly observation, hypothesis,
modification_implication (nonempty strings, each <=300 characters, total <=900), and
support:[{"example_id":"an input ID","member_ids":[0,1]}]. Cite only supplied examples
and members whose observations support this comparison. Contrast requires target
and at least one relevant other peer; positive peers must actually be correct,
negative/wrong peers must actually be wrong on that example. Unsupported fields are
null. If nothing adds a supported insight beyond the Pattern, use NO_SUPPORTED_INSIGHT
and four nulls. Entire compact JSON output must be <=6000 characters.
'''

def readable_observation(trajectory, *, correct):
    """Lossless ordinary response plus decision-relevant status, never a preview."""
    return dict(observed_response=trajectory['observed_response'],
        solution_status=trajectory['solution_status'],response_truncated=trajectory['response_truncated'],
        prediction_valid=trajectory['prediction_valid'],invalid_reason=trajectory['invalid_reason'],correct=correct)

class RootTeamEvidence:
    def __init__(self,store):self.store=store
    def freeze(self,opportunity,pattern):
        from ..benchmarks.math_response_evidence import adaptive_trajectory
        from .binary_responsibility import BinaryPluralityResponsibilityAnalyzer
        state=self.store.snapshot();t=opportunity.target_member;rows=opportunity.evidence.repair_examples
        if state.team_state_id!=opportunity.parent_state_id or state.member_prompts[t]!=opportunity.parent_prompt:
            raise SearchContractError('GROUP_ROOT_SNAPSHOT_MISMATCH')
        if not 1<=len(rows)<=3 or len({r.example_id for r in rows})!=len(rows):
            raise SearchContractError('GROUP_SUPPORT_INVALID')
        diagnosis=BinaryPluralityResponsibilityAnalyzer(self.store.benchmark.capabilities).analyze(state,None)
        indexed={e.item.input_id:(j,e) for j,e in enumerate(self.store.examples)}
        canonical=[]
        for r in rows:
            if r.source_split!='optimize' or r.example_id not in pattern['support_ids'] or r.example_id not in indexed:
                raise SearchContractError('GROUP_OPTIMIZE_SUPPORT_REQUIRED')
            j,e=indexed[r.example_id];s=r.signals;team=state.diagnostics['team_states'][j]
            if (s['input_payload']!=self.store.benchmark.format_input(e.item) or s['gold']!=e.reference
                    or s.get('reference_solution')!=e.reference_solution or not e.reference_solution
                    or tuple(s['responsibility_labels'])!=diagnosis.benchmark_signals['responsibility_labels'][t][r.example_id]):
                raise SearchContractError('GROUP_SUPPORT_SOURCE_MISMATCH')
            members=[]
            for m in range(5):
                tr=adaptive_trajectory(state.diagnostics['raw_profiles'][m][j],member_id=m,
                    prompt=state.member_prompts[m],example_id=r.example_id,problem=s['input_payload'])
                parsed=state.member_outputs[m][j]
                if tr['prediction_valid']!=parsed.valid or parsed.valid!=team.team_validity[m]:
                    raise SearchContractError('GROUP_PROFILE_VALIDITY_MISMATCH')
                members.append(dict(member_id=m,trajectory=tr,answer=parsed.answer if parsed.valid else None,
                    correct=team.team_correctness[m],valid=parsed.valid,equivalence_class=team.team_answers[m] if parsed.valid else None))
            if (digest(members[t]['trajectory'])!=digest(s['solver_trajectory'])
                    or members[t]['correct']!=s['target_member_correct'] or members[t]['valid']!=s['target_member_valid']):
                raise SearchContractError('GROUP_TARGET_OBSERVATION_MISMATCH')
            canonical.append(dict(example_id=r.example_id,problem=s['input_payload'],reference_final_answer=e.reference,
                reference_solution=e.reference_solution,members=members,
                responsibility_label=list(s['responsibility_labels']),
                team_vote=asdict(team),team_impact=asdict(state.diagnostics['opportunities'][r.example_id][t])))
        body=dict(schema=POLICY['input_schema'],root_team_state_id=state.team_state_id,
            root_prompts=[p.to_dict() for p in state.member_prompts],target_member=t,
            aggregation=self.store.aggregation.identity,pattern=deepcopy(pattern),examples=canonical)
        return {**body,'evidence_sha256':digest(body)}

def verify_canonical(value):
    if (not isinstance(value,dict) or value.get('schema')!=POLICY['input_schema']
            or value.get('evidence_sha256')!=digest({k:v for k,v in value.items() if k!='evidence_sha256'})
            or type(value.get('target_member')) is not int or value['target_member'] not in range(5)
            or len(value.get('root_prompts',()))!=5 or not 1<=len(value.get('examples',()))<=3):
        raise SearchContractError('GROUP_CANONICAL_INTEGRITY_REQUIRED')
    from .system_prompt import SystemPrompt,team_identity
    from ..benchmarks.math_response_evidence import validate_adaptive_trajectory
    from ..peer_state import build_team_vote_state,build_peer_vote_context
    from ..responsibility import compute_member_aware_repair_opportunity
    prompts=tuple(SystemPrompt.from_dict(p) for p in value['root_prompts'])
    if team_identity(prompts)!=value['root_team_state_id']:
        raise SearchContractError('GROUP_ROOT_PROMPT_IDENTITY_MISMATCH')
    seen=set()
    for e in value['examples']:
        if (e['example_id'] in seen or e['example_id'] not in value['pattern']['support_ids']
                or [m['member_id'] for m in e['members']]!=list(range(5))):
            raise SearchContractError('GROUP_CANONICAL_MEMBERSHIP_MISMATCH')
        seen.add(e['example_id'])
        vote=e['team_vote']
        reconstructed=build_team_vote_state(question_hash=e['example_id'],gold_answer=vote['gold_answer'],
            answers=vote['team_answers'],valid_vector=vote['team_validity'])
        from .weighted_responsibility import classify_error
        label, _ = classify_error(reconstructed, value['target_member'])
        if label is None or e['responsibility_label'] != [label]:
            raise SearchContractError('GROUP_RESPONSIBILITY_SOURCE_MISMATCH')
        impact=compute_member_aware_repair_opportunity(team_state=reconstructed,
            peer_context=build_peer_vote_context(reconstructed,value['target_member']))
        if digest(asdict(reconstructed))!=digest(vote) or digest(asdict(impact))!=digest(e['team_impact']):
            raise SearchContractError('GROUP_COUNTERFACTUAL_SOURCE_MISMATCH')
        for m in e['members']:
            i=m['member_id']
            validate_adaptive_trajectory(m['trajectory'],example_id=e['example_id'],member_id=i,
                prompt=prompts[i],problem=e['problem'])
            if (type(m['correct']) is not bool or type(m['valid']) is not bool
                    or m['valid']!=m['trajectory']['prediction_valid'] or m['valid']!=reconstructed.team_validity[i]
                    or m['correct']!=reconstructed.team_correctness[i]):
                raise SearchContractError('GROUP_CANONICAL_CORRECTNESS_MISMATCH')
    return value

def group_view(canonical):
    verify_canonical(canonical)
    examples=[]
    for e in canonical['examples']:
        members=[dict(member_id=m['member_id'],answer=m['answer'],equivalence_class=m['equivalence_class'],
            **readable_observation(m['trajectory'],correct=m['correct'])) for m in e['members']]
        vote=e['team_vote']
        vote_class=vote['vote_answer']
        final_answer=next((m['answer'] for m in e['members'] if m['valid'] and m['equivalence_class']==vote_class),None)
        examples.append(dict(example_id=e['example_id'],problem=e['problem'],reference_final_answer=e['reference_final_answer'],
            reference_solution=e['reference_solution'],members=members,
            correct_members=[m['member_id'] for m in e['members'] if m['correct']],
            wrong_members=[m['member_id'] for m in e['members'] if not m['correct']],
            invalid_members=[m['member_id'] for m in e['members'] if not m['valid']],
            team_vote=dict(answer=final_answer,answer_class=vote_class,correct=vote['vote_correct'],top_tie=vote['top_tie'],
                plurality_margin=vote['plurality_margin']),responsibility_label=e['responsibility_label'],team_impact=e['team_impact']))
    from .weighted_responsibility import POLICY as responsibility_policy
    return dict(schema=POLICY['input_schema'],root_team_state_id=canonical['root_team_state_id'],
        responsibility_definition=dict(classification=responsibility_policy['classification'],
            label_correct_counts=responsibility_policy['label_correct_counts'],
            weights=responsibility_policy['weights'],input=responsibility_policy['input']),
        target_member=canonical['target_member'],root_prompts=canonical['root_prompts'],aggregation=canonical['aggregation'],
        selected_pattern=dict(generalized_gradient=canonical['pattern']['generalized_gradient'],
            support_ids=canonical['pattern']['support_ids'],diagnosed_support_ids=[e['example_id'] for e in examples]),examples=examples)

class GroupOutputFailure(GeneratedOutputFailure):pass

def invalid(code):raise GroupOutputFailure(code)

def validate_experience(value,canonical,rows):
    verify_canonical(canonical)
    if not isinstance(value,dict) or set(value)!={'status',*KINDS}:invalid('GROUP_OUTPUT_SCHEMA')
    if value['status'] not in ('SUPPORTED_INSIGHT','NO_SUPPORTED_INSIGHT'):invalid('GROUP_OUTPUT_STATUS')
    if len(json.dumps(value,ensure_ascii=True,separators=(',',':')))>POLICY['maximum_output_characters']:invalid('GROUP_OUTPUT_OVER_LIMIT')
    indexed={e['example_id']:e for e in canonical['examples']};t=canonical['target_member'];n=0
    from .pattern_primitives import guard_abstraction
    from ..current_contract import PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
    private=[e['reference_solution'] for e in canonical['examples']]
    private += [m['trajectory']['observed_response'] for e in canonical['examples'] for m in e['members']]
    for kind in KINDS:
        item=value[kind]
        if item is None:continue
        n+=1
        if not isinstance(item,dict) or set(item)!={'observation','hypothesis','modification_implication','support'}:invalid('GROUP_ENTRY_SCHEMA')
        texts=[item[k] for k in ('observation','hypothesis','modification_implication')]
        if any(not isinstance(x,str) or not x.strip() or len(x)>300 for x in texts):invalid('GROUP_FIELD_OVER_LIMIT_OR_TYPE')
        if sum(map(len,texts))>900:invalid('GROUP_ENTRY_OVER_LIMIT')
        if not isinstance(item['support'],list) or not 1<=len(item['support'])<=3:invalid('GROUP_SUPPORT_INVALID')
        seen=set()
        for support in item['support']:
            if not isinstance(support,dict) or set(support)!={'example_id','member_ids'}:invalid('GROUP_SUPPORT_INVALID')
            ident=support['example_id'];ms=support['member_ids']
            if not isinstance(ident,str) or ident not in indexed or ident in seen:invalid('GROUP_UNKNOWN_OR_DUPLICATE_EXAMPLE')
            seen.add(ident)
            if (not isinstance(ms,list) or not ms or any(type(m) is not int or m not in range(5) for m in ms)
                    or len(set(ms))!=len(ms)):invalid('GROUP_UNKNOWN_OR_DUPLICATE_MEMBER')
            correct={m['member_id'] for m in indexed[ident]['members'] if m['correct']}
            peers=set(ms)-{t};positive=kind in ('correct_peer_contrast','positive_experience')
            if not peers or (positive and not peers<=correct) or (not positive and peers&correct):invalid('GROUP_PEER_CLASS_MISMATCH')
            if kind.endswith('contrast') and t not in ms:invalid('GROUP_CONTRAST_TARGET_REQUIRED')
        # Apply the existing source-aware abstraction and verbatim-copy guards.
        # These reject unsupported specifics; they do not prove semantic fidelity.
        for text in texts:
            try:guard_abstraction(text,rows,abstraction_guard_version=PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)
            except GeneratedOutputFailure:invalid('GROUP_SPECIFIC_CONTENT_REJECTED')
            if group_diagnostic_source_rejected(text,private_texts=private):invalid('GROUP_SPECIFIC_CONTENT_REJECTED')
        if item['modification_implication'].strip().casefold()==canonical['pattern']['generalized_gradient'].strip().casefold():
            invalid('GROUP_ONLY_PATTERN_REPETITION')
    if (n==0)!=(value['status']=='NO_SUPPORTED_INSIGHT'):invalid('GROUP_STATUS_CONTENT_MISMATCH')
    return deepcopy(value)

class GroupDiagnoser:
    def __init__(self,provider,*,audit_writer=None):self.provider=provider;self.audit_writer=audit_writer
    def diagnose(self,canonical,rows):
        view=group_view(canonical);attempts=[];accepted=None;feedback=None;wire_sizes=[]
        for draw in range(POLICY['maximum_draws']):
            packet=dict(evidence=view,output_contract_feedback=feedback)
            text=INSTRUCTION+JSON_INSTRUCTION+json.dumps(packet,sort_keys=True,separators=(',',':'),ensure_ascii=False)
            sizes=dict(characters=len(text),utf8_bytes=len(text.encode('utf-8')));wire_sizes.append(sizes)
            if sizes['utf8_bytes']>POLICY['maximum_input_bytes']:
                error=SearchContractError('GROUP_DIAGNOSIS_RESOURCE_INFEASIBLE');error.resource_statistics=sizes
                raise error
            try:
                response=self.provider(text)
                try:value=json.loads(response)
                except (ValueError,TypeError):invalid('GROUP_INVALID_JSON')
                accepted=validate_experience(value,canonical,rows)
                attempts.append(dict(draw=draw+1,disposition='ACCEPTED',raw_output=response));break
            except GeneratedOutputFailure as error:
                if not isinstance(error,GroupOutputFailure) and str(error)!='GENERATED_OUTPUT_TRUNCATED':raise
                feedback=dict(category=str(error),instruction='Correct only this output-contract failure; use unchanged evidence.')
                attempts.append(dict(draw=draw+1,disposition='REJECTED',category=str(error),
                    raw_output=response if isinstance(error,GroupOutputFailure) else None))
            finally:
                if self.audit_writer and attempts and attempts[-1]['draw']==draw+1:
                    self.audit_writer(dict(input_text=text,attempt=attempts[-1]))
        status=accepted['status'] if accepted else 'GENERATION_FAILED'
        experience={k:accepted[k] if accepted else None for k in KINDS}
        return dict(source='root_team_snapshot',root_team_state_id=canonical['root_team_state_id'],status=status,
            **experience),dict(status=status,attempts=attempts,accepted=accepted,input_view=view,input_sizes_per_draw=wire_sizes,
                accounting_basis='broker_response_usage_provider_or_conservative; transport_failures_in_durable_ledger',
                input_support_count=len(view['examples']),nonempty_experiences=sum(v is not None for v in experience.values()),
                member_counts=[dict(correct=len(e['correct_members']),wrong=len(e['wrong_members']),invalid=len(e['invalid_members'])) for e in view['examples']])
