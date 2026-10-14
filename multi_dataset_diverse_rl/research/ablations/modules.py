"""Explicit module factories; missing Memory policies never degrade to no module."""
from copy import deepcopy

from ...search.schemas import SearchContractError
from ..single_prompt import raw_error
from .interfaces import PreparedFeedback


class NoModule:
    def binding(self):
        return dict(identity='ABLATION_NO_MODULE_V1', semantic_calls_per_opportunity=0,
                    maximum_state_bytes=4, implementation='ablations.modules.NoModule')

    def initial_state(self):
        return None

    def prepare(self, context, state, provider):
        return PreparedFeedback([raw_error(r, context.observations[r['stable_example_id']])
                                 for r in context.rows], None, None, dict(mode='raw_errors'), None)

    def observe(self, outcome, state):
        return state


class ModuleRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, name, factory):
        if name in self._factories or not callable(factory):
            raise SearchContractError('ABLATION_MODULE_REGISTRATION')
        instance = factory()
        for method in ('binding', 'initial_state', 'prepare', 'observe'):
            if not callable(getattr(instance, method, None)):
                raise SearchContractError('ABLATION_MODULE_INTERFACE')
        binding = instance.binding()
        if (not isinstance(binding, dict) or not isinstance(binding.get('identity'), str)
                or type(binding.get('semantic_calls_per_opportunity')) is not int
                or binding['semantic_calls_per_opportunity'] < 0
                or type(binding.get('maximum_state_bytes')) is not int
                or binding['maximum_state_bytes'] < 4):
            raise SearchContractError('ABLATION_MODULE_BINDING')
        self._factories[name] = factory

    def create(self, name):
        if name not in self._factories:
            raise SearchContractError('ABLATION_MODULE_NOT_REGISTERED:' + str(name))
        return self._factories[name]()

    def binding(self, name):
        return deepcopy(self.create(name).binding())


def default_registry():
    from .pattern import LargestPatternModule,GradientOnlyModule
    registry = ModuleRegistry()
    registry.register('none', NoModule)
    registry.register('largest_pattern', LargestPatternModule)
    registry.register('formal_gradient', GradientOnlyModule)
    return registry
