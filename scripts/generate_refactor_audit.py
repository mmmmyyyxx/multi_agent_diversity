"""Generate repeatable configuration and function inventories for refactoring."""

from __future__ import annotations

import ast
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "multi_dataset_diverse_rl" / "config.py"
SETTING_PATH = ROOT / "scripts" / "experiment_config.py"
SYSTEM_PATH = ROOT / "multi_dataset_diverse_rl" / "system.py"
SCAN_ROOTS = (ROOT / "multi_dataset_diverse_rl", ROOT / "scripts", ROOT / "tests")


SECTION_BY_PREFIX = {
    "task_": "data", "dataset_": "data", "train_": "data", "val_": "data", "test_": "data",
    "agent_model": "models", "optimizer_model": "models", "evaluator_model": "models",
    "teacher_": "generation", "critic_": "generation", "student_": "generation", "optimizer_": "generation",
    "candidate_eval_": "evaluation", "candidate_reuse_": "evaluation", "solver_rollout_": "evaluation",
    "reward_": "quality", "accuracy_guard_": "quality", "invalid_guard_": "quality",
    "candidate_refill_": "archive", "probation_": "archive", "qd_": "archive", "beam_size": "archive",
    "joint_": "joint", "lineage_": "lineage", "method_version": "identity",
    "target_selector_": "identity", "beam_policy_": "identity", "active_team_selector_": "identity",
    "out_dir": "output", "llm_call_": "output", "resume_": "runtime", "max_retr": "runtime",
}


def class_fields(path: Path, class_name: str) -> list[tuple[str, str, str, int]]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            rows = []
            for item in node.body:
                if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                    continue
                default = ast.get_source_segment(source, item.value) if item.value is not None else ""
                annotation = ast.get_source_segment(source, item.annotation) or "Any"
                rows.append((item.target.id, annotation, default or "", item.lineno))
            return rows
    return []


def source_texts() -> dict[str, str]:
    result = {}
    for root in SCAN_ROOTS:
        for path in root.rglob("*.py"):
            result[str(path.relative_to(ROOT))] = path.read_text(encoding="utf-8-sig")
    return result


def canonical_section(name: str) -> str:
    for prefix, section in SECTION_BY_PREFIX.items():
        if name == prefix or name.startswith(prefix):
            return section
    if name in {"agents", "epochs", "update_every", "seed", "temperature", "max_tokens"}:
        return "runtime"
    if "version" in name:
        return "identity"
    return "method"


def classification(name: str, references: int) -> str:
    if references <= 1:
        return "DEAD"
    if name.endswith("_version"):
        return "OUTPUT_ONLY" if references < 4 else "ACTIVE_POLICY"
    if name in {"candidate_eval_pool_actual_size", "candidate_eval_total_count", "candidate_eval_unique_question_count"}:
        return "STATE_NOT_CONFIG"
    if name in {"beam_size", "lineage_commit_epochs", "lineage_provisional_epochs", "lineage_switch_confirmation_epochs"}:
        return "COMPAT_ALIAS"
    if canonical_section(name) in {"runtime", "data", "models", "output"}:
        return "ACTIVE_RUNTIME"
    return "ACTIVE_POLICY"


def write_field_matrix(texts: dict[str, str]) -> None:
    config_fields = class_fields(CONFIG_PATH, "Config")
    setting_names = {name for name, *_ in class_fields(SETTING_PATH, "ExperimentSetting")}
    path = ROOT / "REFACTOR_FIELD_MATRIX.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "field", "definition", "default", "type", "reference_count", "classification",
            "canonical_path", "compatibility", "duplicate_with", "removal_plan",
        ])
        for name, annotation, default, line in config_fields:
            refs = sum(text.count(name) for text in texts.values())
            section = canonical_section(name)
            duplicate = ""
            if name == "beam_size": duplicate = "joint_representative_beam_size"
            if name == "lineage_commit_epochs": duplicate = "lineage_commit_required_snapshots"
            if name == "lineage_switch_confirmation_epochs": duplicate = "lineage_switch_confirmation_snapshots"
            cls = classification(name, refs)
            writer.writerow([
                name, f"multi_dataset_diverse_rl/config.py:{line}", default, annotation, refs, cls,
                f"{section}.{name}", "flat property/CLI" if name in setting_names or cls == "COMPAT_ALIAS" else "flat CLI",
                duplicate, "adapter only" if cls == "COMPAT_ALIAS" else "retain canonical",
            ])


def function_rows(path: Path) -> list[dict[str, object]]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = min([node.lineno, *[item.lineno for item in node.decorator_list]])
        rows.append({
            "name": node.name, "start": start, "end": node.end_lineno,
            "lines": node.end_lineno - start + 1, "async": isinstance(node, ast.AsyncFunctionDef),
        })
    return rows


def function_area(name: str) -> str:
    if any(word in name for word in ("checkpoint", "flush", "write_", "history")): return "persistence"
    if any(word in name for word in ("candidate", "prompt", "teacher", "student", "critic")): return "optimization"
    if any(word in name for word in ("rollout", "solve", "cache", "probe")): return "evaluation"
    if any(word in name for word in ("archive", "lineage", "joint", "mechanism")): return "qd"
    if any(word in name for word in ("metric", "reward", "vote", "coverage", "accuracy")): return "metrics"
    return "orchestration"


def write_function_matrix(texts: dict[str, str]) -> None:
    files = [path for root in SCAN_ROOTS for path in root.rglob("*.py")]
    all_rows = [(path, row) for path in files for row in function_rows(path)]
    counts = Counter(row["name"] for _, row in all_rows)
    system_rows = function_rows(SYSTEM_PATH)
    lines = [
        "# Refactor Function Matrix", "", f"Generated from `{len(files)}` Python files.", "",
        "## TraceBeamSearchSystem methods", "",
        "| Function | Lines | Current location | Target responsibility | Duplicate-name count |", "| --- | ---: | --- | --- | ---: |",
    ]
    for row in sorted(system_rows, key=lambda item: (-int(item["lines"]), str(item["name"]))):
        lines.append(
            f"| `{row['name']}` | {row['lines']} | `system.py:{row['start']}` | "
            f"`{function_area(str(row['name']))}` | {counts[str(row['name'])]} |"
        )
    duplicates = [(name, count) for name, count in counts.items() if count > 1 and not name.startswith("test_")]
    lines.extend(["", "## Duplicate function names", "", "| Name | Count |", "| --- | ---: |"])
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(duplicates, key=lambda item: (-item[1], item[0])))
    (ROOT / "REFACTOR_FUNCTION_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    texts = source_texts()
    write_field_matrix(texts)
    write_function_matrix(texts)
    print("wrote REFACTOR_FIELD_MATRIX.csv and REFACTOR_FUNCTION_MATRIX.md")


if __name__ == "__main__":
    main()
