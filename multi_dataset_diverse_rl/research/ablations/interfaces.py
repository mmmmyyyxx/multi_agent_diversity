"""Modules see Optimize only. State belongs to one arm/repeat, never the Solver."""
from dataclasses import dataclass
from copy import deepcopy
from threading import Lock
from typing import Any, Protocol

from ...search.system_prompt import SystemPrompt
import json


@dataclass(frozen=True)
class OptimizeContext:
    cell_id: str
    seed: int
    repeat_id: int
    ordinal: int
    prompt: SystemPrompt
    rows: tuple
    observations: dict


@dataclass(frozen=True)
class PreparedFeedback:
    errors: list
    structured_feedback: dict | None
    module_context: Any
    audit: dict
    state: Any


@dataclass(frozen=True)
class OptimizeOutcome:
    ordinal: int
    parent: SystemPrompt
    candidate: SystemPrompt | None
    retained: SystemPrompt
    before: dict
    candidate_observations: dict | None
    decision: str
    # Deliberately no Validation rows, predictions, scores, or Test capability.


class AblationModule(Protocol):
    def binding(self) -> dict: ...
    def initial_state(self) -> Any: ...
    def prepare(self, context: OptimizeContext, state: Any, provider) -> PreparedFeedback: ...
    def observe(self, outcome: OptimizeOutcome, state: Any) -> Any: ...


class ModuleProvider:
    """Teacher-only capability; no evaluation or held-out loader is exposed."""
    def __init__(self, broker, maximum_calls=0):
        self.__broker = broker
        self.__maximum_calls = maximum_calls
        self.__calls = 0
        self.__lock = Lock()
        self.execution_policy = deepcopy(broker.execution_policy)
        self.__bounded = bool(getattr(broker, 'scope', {}).get('generation_recovery'))
        self.__counts = deepcopy(getattr(broker, 'scope', {}).get('continuation', {}).get(
            'unfinished_module_draws', {}).get(getattr(broker, 'cell_id', None), {}).get(str(getattr(broker, 'round_index', 0)), {}))

    def draw_counts(self):
        with self.__lock:
            return deepcopy(self.__counts)

    def complete(self, *, role, split, **kwargs):
        from ...search.schemas import SearchContractError
        if split != 'optimize' or role not in ('pattern_gradient', 'pattern_cluster'):
            raise SearchContractError('ABLATION_MODULE_PROVIDER_ROLE_FORBIDDEN')
        with self.__lock:
            if self.__bounded:
                packet = json.loads(kwargs['messages'][-1]['content'])
                ident = packet['example']['example_id'] if role == 'pattern_gradient' else 'whole_partition'
                key = role + ':' + ident
                if self.__counts.get(key, 0) >= 3:
                    from ...search.generation_failures import generation_failure
                    # Native extraction marks this item nonactionable; other items survive.
                    raise generation_failure('GENERATED_OUTPUT_TRUNCATED')
                self.__counts[key] = self.__counts.get(key, 0) + 1
            if self.__calls >= self.__maximum_calls:
                raise SearchContractError('ABLATION_MODULE_SEMANTIC_CALL_BUDGET')
            self.__calls += 1
        return self.__broker.complete(role=role, split=split, **kwargs)
