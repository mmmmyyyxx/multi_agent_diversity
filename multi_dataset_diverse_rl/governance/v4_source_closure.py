"""Conservative static source closure for the current V4 diagnostic entry.

Resolve repository-local imports to a fixed point. Dynamic provider and solver
contract entrypoints are explicit roots; report and test code is not included.
This inventory is an execution identity, not an algorithm selector.
"""

from __future__ import annotations

import ast
from pathlib import Path


ENTRYPOINTS = (
    "scripts/run_experiment.py",
    "scripts/prepare_online_transfer_diagnostic_v4.py",
    "multi_dataset_diverse_rl/production_transfer_diagnostic.py",
    "multi_dataset_diverse_rl/team_search/v4_opportunity.py",
    "multi_dataset_diverse_rl/governance/production_execution.py",
)
REQUIRED_ACTIVE = (
    "multi_dataset_diverse_rl/production_transfer_diagnostic.py",
    "multi_dataset_diverse_rl/experiment.py",
    "multi_dataset_diverse_rl/native_feed.py",
    "multi_dataset_diverse_rl/team_search/v4_opportunity.py",
    "multi_dataset_diverse_rl/team_search/feasibility.py",
    "multi_dataset_diverse_rl/team_search/system_runtime.py",
    "multi_dataset_diverse_rl/team_search/task_builder.py",
    "multi_dataset_diverse_rl/team_search/controller.py",
    "multi_dataset_diverse_rl/team_search/primary_responsibility_scheduler.py",
    "multi_dataset_diverse_rl/team_search/execution_runtime.py",
    "multi_dataset_diverse_rl/local_optimizers/gepa_native.py",
    "multi_dataset_diverse_rl/local_optimizers/gepa_optimizer.py",
    "multi_dataset_diverse_rl/candidate_selection.py",
    "multi_dataset_diverse_rl/system.py",
)
DATA_AND_CONTRACT = (
    "experiments/anti_overfitting_split_v1/fold_assignment.json",
    "experiments/gepa_layer2_local_to_team_transfer_diagnostic_v4/PROTOCOL.md",
)


def _module_file(root: Path, name: str) -> Path | None:
    stem = root.joinpath(*name.split("."))
    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def active_v4_source_paths(root: Path) -> tuple[str, ...]:
    """Return the repository-local import closure plus frozen non-code inputs."""
    root = root.resolve()
    pending = [root / relative for relative in ENTRYPOINTS]
    # Contract modules are selected through runtime composition and therefore
    # must be bound even if a future adapter imports them lazily.
    pending.extend((root / "infrastructure/common_solver_contract_v1").glob("*.py"))
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"invalid V4 source dependency: {path}")
        seen.add(path)
        if path.suffix != ".py":
            continue
        relative = path.relative_to(root)
        module_parts = relative.with_suffix("").parts
        package_parts = module_parts if module_parts[-1] == "__init__" else module_parts[:-1]
        if package_parts and package_parts[-1] == "__init__":
            package_parts = package_parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (
                    (*package_parts[: len(package_parts) - node.level + 1],
                     *(node.module.split(".") if node.module else ()))
                    if node.level else tuple(node.module.split(".")) if node.module else ()
                )
                names = [".".join(base)]
                names.extend(".".join((*base, alias.name)) for alias in node.names)
            for name in names:
                dependency = _module_file(root, name)
                if dependency is not None:
                    pending.append(dependency)
        # Importing a child also executes its package initializers.
        for index in range(1, len(module_parts)):
            initializer = root.joinpath(*module_parts[:index], "__init__.py")
            if initializer.is_file():
                pending.append(initializer)
    relative_paths = {path.relative_to(root).as_posix() for path in seen}
    relative_paths.update(DATA_AND_CONTRACT)
    if not set(REQUIRED_ACTIVE).issubset(relative_paths):
        missing = sorted(set(REQUIRED_ACTIVE) - relative_paths)
        raise ValueError(f"V4 active source closure missing {missing}")
    return tuple(sorted(relative_paths))
