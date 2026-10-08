"""Conservative current execution closure, including CLI and package bootstrap."""
import ast
from .source_identity import local_imports

CURRENT_ENTRYPOINTS=(
    'scripts/run_experiment.py',
    'multi_dataset_diverse_rl/benchmarks/math_evidence_binding.py',
    'multi_dataset_diverse_rl/search/current_composition.py',
)
LEGACY_TREATMENT_CLASSES=frozenset({
    'ResponsibilityPatternDiscoveryV3','PatternConditionedEvidenceV4',
    'PatternMemoryOptimizer','PatternMemoryEngine','PatternLayer1Config',
    'StructuredLongTermMemoryProviderV1','StrategyExperienceMemoryV2','StructuredActionMemoryV3',
    'GEPATeamExposureOptimizer','GEPATeamCandidateExposureEngine','V2GEPABridge',
    'PatternDiagnosticV1','FocusedPatternDiagnosticV2',
})


def current_dependency_graph(root):
    root=root.resolve();pending=[root/path for path in CURRENT_ENTRYPOINTS];seen=set();edges=[]
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        for target in local_imports(root,path):
            edges.append((path.relative_to(root).as_posix(),target.relative_to(root).as_posix()))
            if target not in seen:pending.append(target)
    legacy=sorted(path.relative_to(root).as_posix() for path in seen if 'legacy' in path.relative_to(root).parts)
    definitions=[]
    for path in seen:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.ClassDef) and node.name in LEGACY_TREATMENT_CLASSES:
                definitions.append(dict(path=path.relative_to(root).as_posix(),name=node.name))
    return dict(analysis='CONSERVATIVE_AST_ALL_LOCAL_IMPORTS_WITH_INITIALIZERS',
        roots=list(CURRENT_ENTRYPOINTS),modules=sorted(path.relative_to(root).as_posix() for path in seen),
        edges=[dict(source=a,target=b) for a,b in sorted(set(edges))],
        legacy_namespace_dependencies=legacy,legacy_class_definitions=sorted(definitions,key=lambda d:(d['path'],d['name'])),
        loc=sum(len(path.read_text(encoding='utf-8-sig').splitlines()) for path in seen),
        class_count=sum(isinstance(node,ast.ClassDef) for path in seen for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig')))))
