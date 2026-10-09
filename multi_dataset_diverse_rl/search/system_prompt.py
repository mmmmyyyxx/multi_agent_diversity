"""V2.4 member state: one immutable, canonical three-block System Prompt."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json

from .schemas import SearchContractError

BLOCKS = ('role', 'strategy', 'answer')
IDENTITY = 'STRUCTURED_SYSTEM_PROMPT_V1'
CONTRACT = 'STRUCTURED_SYSTEM_BLOCK_MUTATION_CONTRACT_V1'
MAX_CHARS = 3000


@dataclass(frozen=True)
class SystemPrompt:
    role: str
    strategy: str
    answer: str

    def __post_init__(self):
        if any(not isinstance(getattr(self, key), str) or not getattr(self, key).strip()
               for key in BLOCKS):
            raise SearchContractError('SYSTEM_PROMPT_BLOCK_INVALID')
        if len(self.render()) > MAX_CHARS:
            raise SearchContractError('SYSTEM_PROMPT_OVER_LENGTH')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(BLOCKS):
            raise SearchContractError('SYSTEM_PROMPT_SCHEMA_INVALID')
        return cls(**value)

    def to_dict(self):
        return {key: getattr(self, key) for key in BLOCKS}

    def serialize(self):
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(',', ':'))

    def encode(self, encoding='utf-8'):
        """Existing hash ports consume canonical structured bytes, never rendered text."""
        return self.serialize().encode(encoding)

    def render(self):
        return '\n\n'.join(getattr(self, key) for key in BLOCKS)

    @property
    def prompt_hash(self):
        return sha256(self.encode()).hexdigest()

    @property
    def system_hash(self):
        return sha256(self.render().encode()).hexdigest()

    def block_hash(self, block):
        if block not in BLOCKS:
            raise SearchContractError('SYSTEM_PROMPT_TARGET_BLOCK_INVALID')
        return sha256(getattr(self, block).encode()).hexdigest()

    def edit(self, block, content):
        if block not in BLOCKS:
            raise SearchContractError('SYSTEM_PROMPT_TARGET_BLOCK_INVALID')
        return replace(self, **{block: content})


SEED = SystemPrompt('You are a helpful assistant.', 'Answer the question.',
                    "Put your final answer in the format '### <answer>'")


def require_prompt(value):
    if not isinstance(value, SystemPrompt):
        raise SearchContractError('STRUCTURED_SYSTEM_PROMPT_REQUIRED')
    return value


def solver_messages(prompt, problem):
    require_prompt(prompt)
    if not isinstance(problem, str):
        raise SearchContractError('SOLVER_PROBLEM_TEXT_REQUIRED')
    return [dict(role='system', content=prompt.render()), dict(role='user', content=problem)]


def team_identity(prompts):
    return sha256(json.dumps([require_prompt(p).to_dict() for p in prompts],
                            ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def block_lineage(parent, child):
    """Actual spans and reconstructable parents; only private artifacts retain text."""
    from .optimization_evidence import actual_diff
    require_prompt(parent); require_prompt(child)
    changed = [key for key in BLOCKS if getattr(parent, key) != getattr(child, key)]
    return dict(parent_prompt_hash=parent.prompt_hash, child_prompt_hash=child.prompt_hash,
        parent_system_hash=parent.system_hash, child_system_hash=child.system_hash,
        parent_prompt=parent.to_dict(), child_prompt=child.to_dict(),
        edited_blocks=changed, compound_edit=len(changed) > 1,
        block_edits=[dict(edited_block=key, parent_block_hash=parent.block_hash(key),
            child_block_hash=child.block_hash(key),
            actual_block_diff=actual_diff(getattr(parent, key), getattr(child, key))) for key in changed])


def candidate_failed_checks(prompt, *, parent_prompt, examples, max_chars=MAX_CHARS):
    from ..evaluation.semantic_mutable_contract import _FIXED_ANSWER_PAYLOAD
    from ..local_optimizers.example_text import contains_supplied_example_text
    from .optimization_evidence import executability_checks
    import re
    if not isinstance(prompt, SystemPrompt):
        return ('invalid_structure',)
    reasons = []
    text = prompt.render()
    if len(text) > max_chars:
        reasons.append('over_length')
    if prompt == parent_prompt:
        reasons.append('parent_no_op')
    if _FIXED_ANSWER_PAYLOAD.search(text) or re.search(
            r'\b(?:always|regardless\s+of\s+the\s+problem)\s+(?:answer|return|output)\s+["\']?[-+]?\d'
            r'|\b(?:output|return|print)\s+(?:(?:only|exactly|always)\s+)?(?:###\s*|\\boxed\{)?[-+]?\d', text, re.I):
        reasons.append('fixed_answer_payload')
    if contains_supplied_example_text(text, examples):
        reasons.append('example_copying')
    reasons.extend(executability_checks(text))
    return tuple(dict.fromkeys(reasons))
