"""One-time deterministic splitter for the pre-refactor system module."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "multi_dataset_diverse_rl" / "system.py"

RANGES = (
    (483, 1726, "optimization/lifecycle.py", "LifecycleMixin"),
    (1727, 2528, "persistence/runtime_state.py", "RuntimeStateMixin"),
    (2529, 3132, "optimization/candidate_schema.py", "CandidateSchemaMixin"),
    (3133, 3499, "evaluation/solver_service.py", "SolverServiceMixin"),
    (3500, 4707, "metrics/rollout_metrics.py", "RolloutMetricsMixin"),
    (4708, 5520, "optimization/target_selector.py", "TargetSelectorMixin"),
    (5521, 6901, "optimization/candidate_generator.py", "CandidateGeneratorMixin"),
    (6902, 7903, "evaluation/candidate_evaluator.py", "CandidateEvaluatorMixin"),
    (7904, 9234, "optimization/prompt_update_controller.py", "PromptUpdateMixin"),
    (9235, 9833, "optimization/training_controller.py", "TrainingControllerMixin"),
    (9834, 10242, "qd/joint_controller.py", "JointControllerMixin"),
    (10243, 10655, "evaluation/dataset_evaluator.py", "DatasetEvaluatorMixin"),
    (10656, 10739, "persistence/artifact_methods.py", "ArtifactMethodsMixin"),
)


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    system_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TraceBeamSearchSystem")
    methods = [node for node in system_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assigned: set[str] = set()
    mixins = []
    for low, high, relative_path, class_name in RANGES:
        selected = [node for node in methods if low <= node.lineno <= high]
        if not selected:
            raise RuntimeError(f"empty split range: {relative_path}")
        for node in selected:
            if node.name in assigned:
                raise RuntimeError(f"duplicate method assignment: {node.name}")
            assigned.add(node.name)
        body = []
        for node in selected:
            start = min([node.lineno, *[decorator.lineno for decorator in node.decorator_list]])
            body.extend(lines[start - 1:node.end_lineno])
            if body and not body[-1].endswith("\n"):
                body[-1] += "\n"
            body.append("\n")
        output = ROOT / "multi_dataset_diverse_rl" / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            '"""Extracted TraceBeamSearchSystem responsibility mixin."""\n\n'
            "from ..system_shared import *\n\n\n"
            f"class {class_name}:\n"
            + "".join(body),
            encoding="utf-8",
        )
        mixins.append((relative_path.replace("/", ".")[:-3], class_name))
    missing = sorted({node.name for node in methods} - assigned)
    if missing:
        raise RuntimeError(f"unassigned methods: {missing}")

    shared = "".join(lines[: system_class.lineno - 1])
    (SOURCE.parent / "system_shared.py").write_text(
        shared + "\n__all__ = [name for name in globals() if not name.startswith('__')]\n",
        encoding="utf-8",
    )
    imports = "\n".join(f"from .{module} import {name}" for module, name in mixins)
    bases = ",\n    ".join(name for _, name in mixins)
    SOURCE.write_text(
        '"""Public orchestration facade for multi-agent prompt search."""\n\n'
        "from .system_shared import *\n"
        f"{imports}\n\n\n"
        "class TraceBeamSearchSystem(\n"
        f"    {bases},\n"
        "):\n"
        "    GENERIC_DISTINCT_PROCEDURE = (\n"
        '        "Use a distinct decision procedure: first state which reasoning route you will use, "\n'
        '        "then approach the problem through boundary checks, reverse validation, or an alternative representation. "\n'
        '        "If that procedure is not useful, fall back to direct reasoning with one explicit verification step."\n'
        "    )\n\n\n"
        "TextualGradientRLSystem = TraceBeamSearchSystem\n",
        encoding="utf-8",
    )
    for package in ("evaluation", "optimization", "qd", "metrics", "persistence"):
        init = SOURCE.parent / package / "__init__.py"
        init.touch(exist_ok=True)
    print(f"split {len(methods)} methods across {len(RANGES)} modules")


if __name__ == "__main__":
    main()
