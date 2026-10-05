"""Compatibility exports are lazy; current schemas never load legacy backends."""
from importlib import import_module

_EXPORTS={'LocalOptimizerBackend': ('.base', 'LocalOptimizerBackend'), 'LocalPromptOptimizer': ('.base', 'LocalPromptOptimizer'), 'Layer2EvidencePromptOptimizer': ('.base', 'Layer2EvidencePromptOptimizer'), 'LocalSolverEvaluator': ('.base', 'LocalSolverEvaluator'), 'LocalSolverObservation': ('.base', 'LocalSolverObservation'), 'NativeFeedPromptOptimizer': ('.base', 'NativeFeedPromptOptimizer'), 'OptimizationContextProvider': ('.base', 'OptimizationContextProvider'), 'BackendRuntimeConfig': ('.backend_registry', 'BackendRuntimeConfig'), 'ConfiguredLayer1Backend': ('.backend_registry', 'ConfiguredLayer1Backend'), 'Layer1Backend': ('.backend_registry', 'Layer1Backend'), 'Layer1BackendRegistry': ('.backend_registry', 'Layer1BackendRegistry'), 'default_backend_registry': ('.backend_registry', 'default_backend_registry'), 'LocalEvidenceExample': ('.schemas', 'LocalEvidenceExample'), 'LocalOptimizationResult': ('.schemas', 'LocalOptimizationResult'), 'LocalOptimizationTask': ('.schemas', 'LocalOptimizationTask'), 'LocalOptimizerBudget': ('.schemas', 'LocalOptimizerBudget'), 'LocalPromptCandidate': ('.schemas', 'LocalPromptCandidate'), 'OpaqueOptimizerState': ('.schemas', 'OpaqueOptimizerState'), 'RunMode': ('..saturation', 'RunMode'), 'SaturationConfig': ('..saturation', 'SaturationConfig'), 'StopReason': ('..saturation', 'StopReason')}
__all__=list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(import_module(module,__name__),attribute)
    globals()[name]=value
    return value
