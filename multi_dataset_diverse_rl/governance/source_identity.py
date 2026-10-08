"""Offline Unified identities with separate scientific, governance and data scopes.

Historical source-identity v5 remains in its original tool for replay. This
closure conservatively includes current static local imports and package
initializers. Explicit replay implementations and compatibility shims do not
determine new experiment identity. It never imports provider code.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Iterable


def local_imports(root: Path, path: Path, *, initializers: bool = True) -> set[Path]:
    relative = path.relative_to(root).with_suffix('')
    parts = list(relative.parts)
    package = parts[:-1]
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[:len(package)-node.level+1] if node.level else []
            name = '.'.join(prefix + ([node.module] if node.module else []))
            names.add(name)
            names.update(name + '.' + alias.name for alias in node.names)
    found: set[Path] = set()
    for name in names:
        if not name.startswith('multi_dataset_diverse_rl'):
            continue
        module = root.joinpath(*name.split('.'))
        for candidate in (module.with_suffix('.py'), module/'__init__.py'):
            if candidate.is_file():
                found.add(candidate)
        for length in range(1, len(name.split('.'))) if initializers else ():
            init = root.joinpath(*name.split('.')[:length], '__init__.py')
            if init.is_file():
                found.add(init)
    return found


HISTORICAL_CONTROL_PATHS = {
    'multi_dataset_diverse_rl/__init__.py',
    'multi_dataset_diverse_rl/system.py',
    'multi_dataset_diverse_rl/local_optimizers/mars_native.py',
    'multi_dataset_diverse_rl/local_optimizers/backend_registry.py',
    'multi_dataset_diverse_rl/team_search/controller.py',
    'multi_dataset_diverse_rl/search/legacy_bbh_replay.py',
}

# These receipt constructors now reconstruct only the preserved V2.1 bundle.
# Shared benchmark validity/decoding primitives remain current authority.
HISTORICAL_RECEIPT_PATHS = {
    'multi_dataset_diverse_rl/benchmarks/gradient_contract_receipt.py',
    'multi_dataset_diverse_rl/benchmarks/gradient_recovery_contract.py',
    'multi_dataset_diverse_rl/benchmarks/numeric_admissibility_contract.py',
    'multi_dataset_diverse_rl/benchmarks/numeric_calibration_contract.py',
    'multi_dataset_diverse_rl/benchmarks/operational_pilot_contract.py',
    'multi_dataset_diverse_rl/benchmarks/partition_completion_contract.py',
}


def current_scientific_files(root: Path, *, execution_closure: bool = False) -> list[Path]:
    composition=root/'multi_dataset_diverse_rl/search/current_composition.py'
    if composition.is_file():
        roots=[composition,root/'multi_dataset_diverse_rl/current_contract.py']
        if execution_closure:roots.append(root/'scripts/run_experiment.py')
    else:
        roots=[p for p in (root/'multi_dataset_diverse_rl/search').rglob('*.py')
            if '__pycache__' not in p.parts and 'legacy' not in p.parts and p.name!='legacy_bbh_replay.py']
    roots.append(root/'multi_dataset_diverse_rl/versions.py')
    pending = list(roots)
    if execution_closure:
        for entry in roots:
            for parent in entry.parents:
                if parent == root:
                    break
                init=parent/'__init__.py'
                if init.is_file():pending.append(init)
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        if not execution_closure and path.relative_to(root).as_posix() in HISTORICAL_CONTROL_PATHS:
            continue
        seen.add(path)
        pending.extend(local_imports(root, path, initializers=execution_closure)-seen)
    # Benchmark contracts and governance dependencies receive independent hashes.
    result = list(seen) if execution_closure else [p for p in seen if not p.relative_to(root).as_posix().startswith(
        ('multi_dataset_diverse_rl/benchmarks/', 'multi_dataset_diverse_rl/governance/'))]
    spec = root/'docs/design/CURRENT_SPEC.md'
    if spec.is_file():
        result.append(spec)
    semantic_contract = root/'docs/design/TRANSITION_TARGET_OR_TEAM_PROGRESS_V3.md'
    if not semantic_contract.is_file():
        # Preserve synthetic and explicit original-source workspace hashing.
        semantic_contract = root/'docs/design/UNIFIED_METHOD_SEMANTIC_CONTRACT.md'
    if semantic_contract.is_file():
        result.append(semantic_contract)
    cluster_output_contract = root/'docs/design/PATTERN_CLUSTER_OUTPUT_POLICY_V1.md'
    if cluster_output_contract.is_file():
        result.append(cluster_output_contract)
    operational_contract=root/'docs/design/V22_OPERATIONAL_PILOT_CEILING_V1.md'
    if operational_contract.is_file():
        result.append(operational_contract)
    risk_contract = root/'docs/design/SHARED_RISK_MEMORY_V4.md'
    if risk_contract.is_file():
        result.append(risk_contract)
    pattern_contract = root/'docs/design/PATTERN_GRADIENT_DISCOVERY_V4.md'
    if pattern_contract.is_file():
        result.append(pattern_contract)
    numeric_contract = root/'docs/design/NUMERIC_PROVENANCE_GUARD_V5.md'
    if numeric_contract.is_file():
        result.append(numeric_contract)
    return sorted(result, key=lambda p:p.relative_to(root).as_posix())


def hash_scope(root: Path, files: Iterable[Path]) -> dict:
    entries = []
    for path in sorted(set(files), key=lambda p:p.relative_to(root).as_posix()):
        if path.is_file():
            raw=path.read_bytes()
            entries.append({'path':path.relative_to(root).as_posix(),
                            'sha256':hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest(),
                            'raw_sha256':hashlib.sha256(raw).hexdigest()})
    canonical=[{'path':r['path'],'sha256':r['sha256']} for r in entries]
    encoded = json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()
    return {'sha256':hashlib.sha256(encoded).hexdigest(), 'files':entries}


def is_legacy_forwarder(root: Path, path: Path) -> bool:
    """Recognize a forwarding module, never hide a substantive current import."""
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    statements=[node for node in tree.body if not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))]
    return bool(statements) and all(isinstance(node,ast.ImportFrom) for node in statements) and any(
        'legacy' in target.relative_to(root).parts for target in local_imports(root,path,initializers=False))


def current_authority_files(root: Path, area: str) -> list[Path]:
    """Keep current contracts; exclude explicit replay and forwarding shims."""
    result=[]
    for path in (root/'multi_dataset_diverse_rl'/area).rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or 'legacy' in path.parts:
            continue
        if path.suffix not in {'.py','.json','.md','.txt'}:
            continue
        if path.name=='legacy_bbh_replay.py':
            continue
        if path.relative_to(root).as_posix() in HISTORICAL_RECEIPT_PATHS:
            continue
        if path.suffix=='.py' and is_legacy_forwarder(root,path):
            continue
        result.append(path)
    return result


def build_unified_source_identity(workspace: Path, dataset_manifest: Path | None = None,
                                  config_paths: Iterable[Path] = ()) -> dict:
    root = workspace.resolve()
    configs = [p.resolve() if p.is_absolute() else root/p for p in config_paths]
    dataset = (dataset_manifest.resolve() if dataset_manifest.is_absolute()
               else root/dataset_manifest) if dataset_manifest is not None else None
    for path in configs + ([dataset] if dataset is not None else []):
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('identity input must be an existing repository file')
        if path.relative_to(root).as_posix().startswith(('reports/', 'docs/archive/')):
            raise ValueError('evidence/archive cannot be an identity input')
        if path.suffix.lower() not in {'.json','.yaml','.yml'}:
            raise ValueError('config/dataset-manifest identity inputs must be structured JSON/YAML')
    scientific_files=current_scientific_files(root)
    execution_files=current_scientific_files(root, execution_closure=True)
    bootstrap_files=[p for p in execution_files if p not in scientific_files and
                     not p.relative_to(root).as_posix().startswith('multi_dataset_diverse_rl/benchmarks/')]
    scientific = hash_scope(root, scientific_files+configs)
    governance = hash_scope(root, [root/'AGENTS.md',
        root/'experiments/registry.yaml', root/'experiments/lineage.yaml',
        root/'experiments/current_frontier.yaml', root/'experiments/manifest_schema_index.json',
        root/'docs/design/invariants.yaml', root/'docs/failures/registry.yaml',
        *list((root/'docs/workflows').rglob('*.md')),
        *current_authority_files(root,'governance'),
        *list((root/'experiments/schema').rglob('*.json')), *bootstrap_files])
    benchmark = hash_scope(root, [*current_authority_files(root,'benchmarks'),
        root/'requirements-benchmark-evaluators.txt',
        root/'requirements-ifbench-evaluators.txt'])
    data = hash_scope(root, [dataset] if dataset is not None else [])
    return {'source_identity_version':'unified_source_identity_v1',
            'scientific_source_hash':scientific['sha256'],
            'governance_hash':governance['sha256'],
            'benchmark_contract_hash':benchmark['sha256'],
            'dataset_manifest_hash':data['sha256'],
            'dataset_manifest_bound':dataset is not None,
            'scientific_method_identity':scientific['sha256'],
            'scopes':{'scientific':scientific,'governance':governance,
                      'benchmark':benchmark,'dataset_manifest':data},
            'operational_bootstrap_paths':[p.relative_to(root).as_posix() for p in sorted(bootstrap_files)],
            'closure_policy':'current graph static dependencies are scientific; current package bootstrap is governed operational scope; benchmark contracts are hashed separately; explicit legacy namespaces and historical forwarding shims are excluded; dynamic imports require freeze-specific review',
            'content_hash_policy':'canonical LF text for portable scope hashes; raw_sha256 receipts preserve execution-byte evidence separately',
            'ready_to_run':False}
