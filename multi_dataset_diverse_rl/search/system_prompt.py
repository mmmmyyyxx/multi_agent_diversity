"""V2.4 member state: one immutable, canonical three-block System Prompt."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json

from .schemas import SearchContractError

BLOCKS = ('role', 'strategy', 'answer')
IDENTITY = 'STRUCTURED_SYSTEM_PROMPT_V2'
CONTRACT = 'STRUCTURAL_SINGLE_BLOCK_CANDIDATE_CONTRACT_V2'


@dataclass(frozen=True)
class SystemPrompt:
    role: str
    strategy: str
    answer: str

    def __post_init__(self):
        if any(not isinstance(getattr(self, key), str) or not getattr(self, key).strip()
               for key in BLOCKS):
            raise SearchContractError('SYSTEM_PROMPT_BLOCK_INVALID')

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
                    'On the last line, write "Final answer:" followed by your mathematical answer.')


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


def candidate_failed_checks(prompt, *, parent_prompt, examples=()):
    """Structure and edit scope only. Text and performance are not legality rules.

    Identical parents are structurally valid. Controllers record NO_OP and skip
    their evaluation, without requesting another legal-candidate draw.
    """
    if not isinstance(prompt, SystemPrompt):
        return ('invalid_structure',)
    if not isinstance(parent_prompt, SystemPrompt):
        return ('invalid_parent_structure',)
    if any(not isinstance(getattr(prompt,k),str) or not getattr(prompt,k).strip() for k in BLOCKS):
        return ('empty_or_nonstring_block',)
    if sum(getattr(prompt,k)!=getattr(parent_prompt,k) for k in BLOCKS)>1:
        return ('multiple_block_edit',)
    return ()
