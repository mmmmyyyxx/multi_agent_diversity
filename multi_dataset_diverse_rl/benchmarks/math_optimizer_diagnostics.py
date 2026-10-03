"""Content-free diagnostics; never score or admit an optimizer proposal."""
from collections import Counter
import re

from ..evaluation.mutable_prompt_contract import mutable_prompt_violation_reasons
from ..local_optimizers.gepa_proposer_contract import DECISION_PROCEDURE_REFLECTION_TEMPLATE
from ..search.schemas import SearchContractError


DIAGNOSTIC_ID = 'OPTIMIZER_NONTHINKING_WIRE_WITNESS_V1'


def nonthinking_evidence_contract():
    from .. import versions
    return dict(identity=versions.MATH_OPTIMIZER_NONTHINKING_EVIDENCE_VERSION,
        direct='dispatched_false_and_actual_zero_without_contradiction',
        equivalent='dispatched_false_accepted_stop_no_reasoning_no_parser_loss',
        missing_tokens='never_infer_zero',contradiction='provider_control_failure',
        diagnostic_candidate_rejection='record_only_no_output_reuse')


def provider_thinking_indicators(body):
    """Retain JSON paths of positive thinking metadata, never its text."""
    found=[]
    def walk(value,path='$'):
        if isinstance(value,dict):
            for key,item in value.items():
                child=path+'.'+str(key);name=str(key).lower()
                watched=name in {'enable_thinking','thinking','thinking_enabled','thinking_mode','reasoning','reasoning_content','reasoning_text','reasoning_tokens','reasoning_token_count'}
                inactive=item is None or item is False or item==0 or item=='' or item==[] or item=={}
                if isinstance(item,str) and name in {'enable_thinking','thinking','thinking_enabled','thinking_mode'}:
                    inactive=item.lower() in {'false','off','disabled','none','nonthinking','non-thinking'}
                if watched and not inactive:found.append(child)
                walk(item,child)
        elif isinstance(value,list):
            for i,item in enumerate(value):walk(item,path+'['+str(i)+']')
    walk(body)
    return sorted(set(found))


def reflection_input_anatomy(messages):
    if len(messages) != 1 or messages[0]['role'] != 'user':
        raise SearchContractError('REFLECTION_ANATOMY_UNSUPPORTED_MESSAGES')
    text = messages[0]['content']
    template = DECISION_PROCEDURE_REFLECTION_TEMPLATE
    before, tail = template.split('<curr_param>')
    middle, after = tail.split('<side_info>')
    if not text.startswith(before) or not text.endswith(after):
        raise SearchContractError('REFLECTION_ANATOMY_TEMPLATE_MISMATCH')
    parent, sep, side = text[len(before):len(text)-len(after)].partition(middle)
    if not sep:
        raise SearchContractError('REFLECTION_ANATOMY_TEMPLATE_MISMATCH')
    records = re.split(r'(?m)^# Example \d+\n', side)[1:]
    summaries = []
    for record in records:
        parts = re.split(r'(?m)^## (Problem|Reasoning Evidence|Evaluation Outcome|Reasoning Focus|example_id)\n', record)
        fields = {parts[i]:parts[i+1].strip() for i in range(1,len(parts),2)}
        summaries.append(dict(record_chars=len(record), problem_chars=len(fields.get('Problem','')),
            reasoning_evidence_chars=len(fields.get('Reasoning Evidence','')),
            feedback_metadata_chars=sum(len(v) for k,v in fields.items() if k not in {'Problem','Reasoning Evidence'})))
    return dict(total_message_chars=sum(len(m['content']) for m in messages),
        current_parent_prompt_chars=len(parent), reflection_template_chars=len(template),
        reflection_template_literal_chars=len(before)+len(middle)+len(after),
        side_info_total_chars=len(side), reflective_examples=len(records), per_example=summaries,
        side_info_max_record_chars=max((len(r) for r in records),default=0),
        side_info_mean_record_chars=sum(map(len,records))/len(records) if records else 0,
        reasoning_evidence_total_chars=sum(r['reasoning_evidence_chars'] for r in summaries))


def generation_diagnostics(content):
    """Exact stripped nonempty lines; 8-grams are casefolded whitespace words.

    A loop witness is four identical contiguous blocks of at least 40 characters
    or eight words. This is a deterministic lower bound on obvious repetition,
    not a semantic classifier. No text fragments are returned.
    """
    text = content if isinstance(content,str) else ''
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    counts = Counter(lines)
    duplicates = len(lines)-len(counts)
    max_run = run = 0
    long_line_loop = False
    previous = None
    for line in lines:
        run = run+1 if line==previous else 1
        max_run = max(max_run,run)
        long_line_loop = long_line_loop or run>=4 and len(line)>=40
        previous = line
    words = text.casefold().split()
    grams = [tuple(words[i:i+8]) for i in range(max(0,len(words)-7))]
    gram_counts = Counter(grams)
    loop = False
    # Only inspect repeated starts, rather than quadratic arbitrary substrings.
    for gram,n in gram_counts.items():
        if n < 4: continue
        starts = [i for i in range(len(words)-7) if tuple(words[i:i+8])==gram]
        for a,b in zip(starts,starts[1:]):
            width=b-a
            if width>=8 and a+4*width<=len(words) and all(words[a:a+width]==words[a+j*width:a+(j+1)*width] for j in range(1,4)):
                loop=True;break
        if loop:break
    # Covers long repeated lines/paragraphs without eight whitespace words.
    loop = loop or long_line_loop
    fence = re.fullmatch(r'\s*```(.*?)```\s*',text,re.S) if text.count('```')==2 else None
    candidate = fence.group(1).strip() if fence else None
    delimiters=text.count('```')
    first=text.find('```')
    prefix = text[first+3:] if first>=0 else ''
    if '```' in prefix:prefix=prefix.split('```',1)[0]
    fraction=duplicates/len(lines) if lines else 0
    violations=list(mutable_prompt_violation_reasons(candidate)) if candidate is not None else []
    if candidate is not None and len(candidate)>3000:violations.append('over_length')
    if candidate is not None and not candidate:violations.append('empty')
    return dict(output_chars=len(text),output_lines=len(text.splitlines()),nonempty_lines=len(lines),
        unique_lines=len(counts),duplicate_lines=duplicates,duplicate_line_fraction=fraction,
        max_identical_line_run=max_run,repeated_8gram_fraction=(len(grams)-len(gram_counts))/len(grams) if grams else 0,
        obvious_loop_fourfold=loop,repetition_pathology=fraction>=0.5 or max_run>=8 or loop,
        fence_open_count=(delimiters+1)//2,fence_close_count=delimiters//2,
        complete_single_fence=fence is not None,candidate_prefix_chars=len(prefix.strip()),
        candidate_chars=len(candidate) if candidate is not None else None,
        candidate_contract_violations=violations,
        candidate_generation_contract_valid=candidate is not None and not violations)


def optimizer_response_telemetry(request,result,policy,evidence_policy=None):
    from ..governance.token_accounting import serialized_request
    import json
    body=json.loads(serialized_request(request))
    usage=result.get('provider_usage_details') or {}
    details=usage.get('completion_tokens_details') or {}
    reasoning=details.get('reasoning_tokens')
    chars=result.get('provider_reasoning_character_count')
    present=result.get('provider_reasoning_content_present')
    indicators=result.get('provider_thinking_indicators') or []
    contradiction=(type(reasoning) is int and reasoning>0) or (chars or 0)>0 or bool(indicators)
    direct=body.get('enable_thinking') is False and type(reasoning) is int and reasoning==0 and chars in (None,0) and present in (False,True) and not contradiction
    equivalent=False
    if evidence_policy is not None:
        if evidence_policy!=nonthinking_evidence_contract():raise SearchContractError('OPTIMIZER_NONTHINKING_EVIDENCE_POLICY_MISMATCH')
        equivalent=body.get('enable_thinking') is False and reasoning is None and result.get('finish_reason')=='stop' and chars in (None,0) and present in (False,True) and not contradiction and result.get('provider_metadata_loss_audited') is True and result.get('provider_response_accepted') is True
    return dict(generation_policy_identity=policy['identity'],enable_thinking=body.get('enable_thinking'),
        **{k:body.get(k) for k in ('temperature','top_p','top_k','presence_penalty','max_completion_tokens')},
        input_tokens=result.get('provider_reported_input_tokens',result.get('input_tokens')),
        completion_tokens=result.get('provider_reported_output_tokens',result.get('output_tokens')),
        reasoning_tokens=reasoning,provider_reported_text_tokens=details.get('text_tokens'),
        reasoning_content_present=result.get('provider_reasoning_content_present'),
        reasoning_content_chars=chars,content_chars=len(result.get('text') or ''),finish_reason=result.get('finish_reason'),
        nonthinking_wire_confirmed=direct or equivalent,
        nonthinking_evidence_level='DIRECT_LEVEL_A' if direct else 'EQUIVALENT_LEVEL_B' if equivalent else 'CONTRADICTORY' if contradiction else 'UNCONFIRMED',
        provider_thinking_indicators=indicators,
        nonthinking_evidence_policy_identity=evidence_policy['identity'] if evidence_policy else None,
        diagnostics=generation_diagnostics(result.get('text')))
