"""Strict output types and bounded, source-free feedback for optimizer draws."""
from copy import deepcopy
import json

from .generation_failures import generation_failure
from .schemas import SearchContractError

IDENTITY = 'TYPED_OPTIMIZER_OUTPUT_FEEDBACK_V1'
JSON_INSTRUCTION = (
    'Return valid JSON only: no fences, comments or additional keys. '
    'Use actual JSON null. Inside JSON strings escape every backslash and '
    'double quote correctly; ordinary mathematical prose is also acceptable. '
    'For example, the JSON encoding of a mathematical backslash is: '
    + json.dumps({'notation': r'\alpha'}) + '.\n'
)
_FIELDS = {'output','rendered_system_prompt','disposition','observed_failure','diagnosis','reusable_correction',
    'suggested_block','expected_effect','decision','target_block','new_content','change_summary'}
_CODES = {'JSON_SYNTAX','JSON_INVALID_ESCAPE','TYPE','KEY_SET','ENUM','EMPTY',
    'OVER_LIMIT','MUST_BE_NULL','TRUNCATED','CONTENT_REJECTED'}
_TYPES = {'object','array','string','null','boolean','number','other'}

def _type(value):
    if value is None:return 'null'
    if isinstance(value,bool):return 'boolean'
    if isinstance(value,str):return 'string'
    if isinstance(value,dict):return 'object'
    if isinstance(value,list):return 'array'
    if isinstance(value,(float,int)):return 'number'
    return 'other'

def issue(field,code,value=None,limit=None):
    return dict(field=field,code=code,actual_type=_type(value),
        actual_characters=len(value) if isinstance(value,str) else None,limit=limit)

def validate_feedback(value, *, stage):
    if (not isinstance(value,dict) or set(value)!={'identity','stage','category','issues'}
            or value['identity']!=IDENTITY or value['stage']!=stage
            or not isinstance(value['category'],str) or value['category'] not in {
                'REFERENCE_GRADIENT_OUTPUT_INVALID','PATTERN_GRADIENT_EXTRACTION_INVALID',
                'PATTERN_DISCOVERY_EXAMPLE_LEAKAGE','PATTERN_DISCOVERY_INVALID_ABSTRACTION',
                'GENERATED_OUTPUT_TRUNCATED','REFLECTION_OUTPUT_CONTRACT_INVALID','SYSTEM_PROMPT_OVER_LENGTH'}
            or not isinstance(value['issues'],list) or not 1<=len(value['issues'])<=6):
        raise SearchContractError('OPTIMIZER_OUTPUT_FEEDBACK_INTEGRITY_REQUIRED')
    for row in value['issues']:
        if (not isinstance(row,dict) or set(row)!={'field','code','actual_type','actual_characters','limit'}
                or not isinstance(row['field'],str) or row['field'] not in _FIELDS
                or not isinstance(row['code'],str) or row['code'] not in _CODES
                or not isinstance(row['actual_type'],str) or row['actual_type'] not in _TYPES
                or any(row[k] is not None and (type(row[k]) is not int or row[k]<0)
                    for k in ('actual_characters','limit'))):
            raise SearchContractError('OPTIMIZER_OUTPUT_FEEDBACK_INTEGRITY_REQUIRED')
    return deepcopy(value)

def feedback_for_error(error, *, stage):
    supplied=getattr(error,'output_contract_feedback',None)
    if supplied is not None:return validate_feedback(supplied,stage=stage)
    category=str(error)
    row=issue('output','TRUNCATED') if category=='GENERATED_OUTPUT_TRUNCATED' else issue(
        'reusable_correction' if stage=='gradient' else 'output','TYPE' if stage=='reflection' else 'CONTENT_REJECTED')
    return validate_feedback(dict(identity=IDENTITY,stage=stage,category=category,issues=[row]),stage=stage)

def reject(category,stage,issues):
    error=generation_failure(category)
    error.output_contract_feedback=validate_feedback(dict(identity=IDENTITY,stage=stage,
        category=category,issues=issues[:6]),stage=stage)
    raise error

def parse_output(text, *, stage):
    try:return json.loads(text)
    except (ValueError,TypeError) as error:
        code='JSON_INVALID_ESCAPE' if isinstance(error,json.JSONDecodeError) and error.msg=='Invalid \\escape' else 'JSON_SYNTAX'
        reject('PATTERN_GRADIENT_EXTRACTION_INVALID' if stage=='gradient' else
            'REFLECTION_OUTPUT_CONTRACT_INVALID',stage,[issue('output',code)])

def _enum(issues,value,field,allowed):
    v=value[field]
    if v is None and None in allowed:return
    if not isinstance(v,str):issues.append(issue(field,'TYPE',v))
    elif v not in allowed:issues.append(issue(field,'ENUM',v))

def _text(issues,value,field,limit):
    v=value[field]
    if not isinstance(v,str):issues.append(issue(field,'TYPE',v,limit))
    elif not v.strip():issues.append(issue(field,'EMPTY',v,limit))
    elif len(v)>limit:issues.append(issue(field,'OVER_LIMIT',v,limit))

def validate_gradient_output(value, limits):
    category='REFERENCE_GRADIENT_OUTPUT_INVALID';stage='gradient'
    keys={'disposition','observed_failure','diagnosis','reusable_correction','suggested_block','expected_effect'}
    if not isinstance(value,dict):reject(category,stage,[issue('output','TYPE',value)])
    if set(value)!=keys:reject(category,stage,[issue('output','KEY_SET')])
    issues=[]
    _enum(issues,value,'disposition',{'ACTIONABLE','UNCERTAIN'})
    _enum(issues,value,'suggested_block',{None,'role','strategy','answer'})
    for k in ('observed_failure','diagnosis'):_text(issues,value,k,limits[k])
    if issues:reject(category,stage,issues)
    if value['disposition']=='UNCERTAIN':
        for k in ('reusable_correction','expected_effect'):
            if value[k] is not None:issues.append(issue(k,'MUST_BE_NULL',value[k]))
    else:_text(issues,value,'expected_effect',limits['expected_effect'])
    if issues:reject(category,stage,issues)
    # Correction validity, length, abstraction and source guards remain separate.
    return value

def validate_reflection_output(value):
    category='REFLECTION_OUTPUT_CONTRACT_INVALID';stage='reflection'
    if not isinstance(value,dict):reject(category,stage,[issue('output','TYPE',value)])
    if set(value)!={'target_block','new_content','change_summary'}:
        reject(category,stage,[issue('output','KEY_SET')])
    issues=[]
    _enum(issues,value,'target_block',{'role','strategy','answer'})
    # Rendered total length and content checks remain in the edit guard.
    v=value['new_content']
    if not isinstance(v,str):issues.append(issue('new_content','TYPE',v))
    elif not v.strip():issues.append(issue('new_content','EMPTY',v))
    summary=value['change_summary']
    if not isinstance(summary,str):issues.append(issue('change_summary','TYPE',summary))
    # This metadata does not impose a candidate text or character budget.
    if issues:reject(category,stage,issues)
    return value

def reflection_feedback_prefix(feedback):
    if feedback is None:return ''
    value=validate_feedback(feedback,stage='reflection')
    return ('Previous draw output-contract feedback only; this is not mathematical '
        'evidence or a requested repair. Correct the listed format/type/length errors '
        'while using the complete unchanged task evidence below.\n'
        + json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')
