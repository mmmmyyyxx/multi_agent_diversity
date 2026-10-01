"""Offline Unified identities with separate scientific, governance and data scopes.

Historical source-identity v5 remains in its original tool for replay. This
closure conservatively includes static local imports and package initializers,
including compatibility dependencies. It never imports provider code.
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


def current_scientific_files(root: Path, *, execution_closure: bool = False) -> list[Path]:
    roots = [p for p in (root/'multi_dataset_diverse_rl/search').rglob('*.py')
             if '__pycache__' not in p.parts and p.name != 'legacy_bbh_replay.py']
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
        *list((root/'multi_dataset_diverse_rl/governance').rglob('*.py')),
        *list((root/'experiments/schema').rglob('*.json')), *bootstrap_files])
    benchmark = hash_scope(root, [*[p for p in (root/'multi_dataset_diverse_rl/benchmarks').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.md','.txt'}],
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
            'closure_policy':'active graph static dependencies are scientific; package bootstrap and historical controller imports are governed operational dependencies, included in governance_hash; benchmark dependencies hashed separately; dynamic imports require freeze-specific review',
            'content_hash_policy':'canonical LF text for portable scope hashes; raw_sha256 receipts preserve execution-byte evidence separately',
            'ready_to_run':False}
