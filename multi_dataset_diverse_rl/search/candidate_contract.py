"""Single candidate policy: no semantic filters, character caps or legacy modes."""
from dataclasses import dataclass
import json
import re

from .system_prompt import SystemPrompt, CONTRACT
from .schemas import SearchContractError
from .generation_failures import generation_failure

POLICY=dict(identity=CONTRACT,blocks=['role','strategy','answer'],nonempty_strings=True,
    edits_per_candidate=1,maximum_prompt_characters=None,content_filters=False,
    parsing='unique_json_object_optional_complete_fence_v1',
    no_op='record_skip_evaluation_no_regeneration',selection='existing_measured_performance')

@dataclass(frozen=True)
class CandidateEdit:
    prompt: SystemPrompt
    block: str
    change_summary: str
    no_op: bool

def _object(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('ambiguous duplicate JSON key')
        result[k]=v
    return result

def _decode(text):
    """Recover wrappers only; never rewrite mathematical content or JSON escapes."""
    if not isinstance(text,str):raise ValueError('candidate output must be text')
    value=text.strip()
    fence=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n\s*```',value,re.S|re.I)
    if fence:value=fence[1].strip()
    decoder=json.JSONDecoder(object_pairs_hook=_object)
    try:return decoder.decode(value)
    except (ValueError,TypeError):
        # Salvage exactly one object surrounded by prose. Multiple JSON objects,
        # conflicting fenced alternatives and malformed bodies remain ambiguous.
        candidates=[]
        for match in re.finditer(r'\{',value):
            try:obj,end=decoder.raw_decode(value,match.start())
            except ValueError:continue
            if isinstance(obj,dict):candidates.append((match.start(),end,obj))
        outer=[v for v in candidates if not any(a<v[0]<b for a,b,_ in candidates)]
        if len(outer)!=1:raise ValueError('no unique complete candidate object')
        a,b,obj=outer[0]
        if '{' in value[:a] or '}' in value[b:]:raise ValueError('incomplete or competing candidate object')
        return obj

def parse_candidate_edit(text,*,parent,finish_reason='stop'):
    """Resolve one named replacement; identical bytes are a no-op, not an error."""
    if not isinstance(parent,SystemPrompt):raise SearchContractError('CANDIDATE_PARENT_STRUCTURE_REQUIRED')
    try:
        if finish_reason!='stop':raise ValueError('incomplete candidate response')
        value=_decode(text)
        if not isinstance(value,dict):raise ValueError('candidate object required')
        if set(value)=={'block','content'}:
            block,content,summary=value['block'],value['content'],''
        elif set(value) in ({'target_block','new_content'},{'target_block','new_content','change_summary'}):
            block,content,summary=value['target_block'],value['new_content'],value.get('change_summary','')
        else:raise ValueError('one named replacement required')
        if not isinstance(block,str) or block not in ('role','strategy','answer'):
            raise ValueError('unknown block')
        if not isinstance(content,str) or not content.strip() or not isinstance(summary,str):
            raise ValueError('nonempty string replacement required')
        child=parent.edit(block,content)
        return CandidateEdit(child,block,summary,child==parent)
    except SearchContractError:
        # Trusted state/construction failures are not malformed LLM output.
        raise
    except (ValueError,TypeError) as exc:
        raise generation_failure('REFLECTION_OUTPUT_CONTRACT_INVALID') from exc
