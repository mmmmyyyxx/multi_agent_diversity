from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_MANIFEST = ROOT / "configs" / "task_level_comparison_strict_bbh_seed42.yaml"
EXPERIMENT_MANIFEST = (
    ROOT / "experiments" / "manifests" / "diversity_matrix_d0_d5.yaml"
)
DESIGN_ROOT = ROOT / "experiments" / "diversity_matrix_d0_d5_20260903"
DEFAULT_PREP_ROOT = ROOT / "runs" / "diversity_matrix_d0_d5_prep_20260903"
DEFAULT_RUN_ROOT = ROOT / "runs" / "diversity_matrix_d0_d5_20260903"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "diversity_matrix_d0_d5_20260903"

RUNTIME_VERSION = "diversity_matrix_d0_d5_online_v1"
AUTH_ENV = "DIVERSITY_MATRIX_D0_D5_AUTHORIZED"
UPDATES = 32
AGENTS = 5
SOURCE_CANDIDATES_PER_TARGET = 2
REVISION_OPPORTUNITIES_PER_VALID_SOURCE = 1
SOLVER_MODEL = "qwen3-14b"
ROLE_MODEL = "qwen3.7-flash"
THINKING = False

ARMS: dict[str, dict[str, Any]] = {
    "D0": {
        "setting": "shared_static_reference",
        "allocation": "none",
        "proposal": "static",
        "target_branches": 0,
        "no_semantic_critic": False,
    },
    "D1": {
        "setting": "shared_generic_evolution",
        "allocation": "canonical_generic_s0",
        "proposal": "generic",
        "target_branches": 1,
        "no_semantic_critic": False,
    },
    "D2": {
        "setting": "experimental_diversity_d2_rr_generic",
        "allocation": "responsibility_round_robin_dual",
        "proposal": "generic",
        "target_branches": 2,
        "no_semantic_critic": False,
    },
    "D3": {
        "setting": "experimental_diversity_d3_w1_generic",
        "allocation": "repairability_adjusted_w1",
        "proposal": "generic",
        "target_branches": 2,
        "no_semantic_critic": False,
    },
    "D4": {
        "setting": "experimental_diversity_d4_rr_responsibility",
        "allocation": "responsibility_round_robin_dual",
        "proposal": "teacher_clean_hard_gate_student",
        "target_branches": 2,
        "no_semantic_critic": True,
    },
    "D5": {
        "setting": "experimental_diversity_d5_w1_responsibility",
        "allocation": "repairability_adjusted_w1",
        "proposal": "teacher_clean_hard_gate_student",
        "target_branches": 2,
        "no_semantic_critic": True,
    },
}
ARM_ORDER = tuple(ARMS)

CONTRASTS = {
    "C1_D3_minus_D2": ("D3", "D2", True),
    "C2_D4_minus_D2": ("D4", "D2", True),
    "C3_D5_minus_D3": ("D5", "D3", True),
    "C5_D5_minus_D1": ("D5", "D1", False),
    "C6_D1_minus_D0": ("D1", "D0", False),
    "C6_D5_minus_D0": ("D5", "D0", False),
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def project_local(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents


def manifest() -> dict[str, Any]:
    value = yaml.safe_load(EXPERIMENT_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment manifest must be a mapping")
    return value


def _seed_numbers_from_text(text: str) -> set[int]:
    expressions = (
        r"(?i)\bseed(?:s)?\s*[:=_-]?\s*(\d{1,4})\b",
        r"(?i)\bseed(\d{1,4})\b",
    )
    return {
        int(match.group(1))
        for expression in expressions
        for match in re.finditer(expression, text)
    }


def seed_registry_scan(*, exclude_paths: Iterable[Path] = ()) -> dict[str, Any]:
    excluded = {path.resolve() for path in exclude_paths}
    used: set[int] = set()
    evidence: list[dict[str, Any]] = []
    tracked = git("ls-files", "experiments", "reports").splitlines()
    for relative in tracked:
        path = (ROOT / relative).resolve()
        if path in excluded or not path.is_file():
            continue
        try:
            values = _seed_numbers_from_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        if values:
            used.update(values)
            evidence.append({
                "source_kind": "tracked_experiment_or_report",
                "path_hash": sha256_json(relative.replace("\\", "/")),
                "seeds": sorted(values),
            })
    runs = ROOT / "runs"
    if runs.exists():
        for path in runs.rglob("run_meta.json"):
            try:
                value = read_json(path)
                seed = int(value.get("config", {}).get("seed"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            used.add(seed)
            evidence.append({
                "source_kind": "private_run_metadata",
                "path_hash": sha256_json(path.relative_to(ROOT).as_posix()),
                "seeds": [seed],
            })
        for path in runs.iterdir():
            values = _seed_numbers_from_text(path.name)
            used.update(values)
            if values:
                evidence.append({
                    "source_kind": "private_run_root_name",
                    "path_hash": sha256_json(path.relative_to(ROOT).as_posix()),
                    "seeds": sorted(values),
                })
    maximum = max(used, default=41)
    selected = [maximum + offset for offset in (1, 2, 3)]
    return {
        "scan_version": "fresh_seed_registry_scan_v1",
        "selection_rule": (
            "smallest_consecutive_triplet_strictly_above_the_maximum_"
            "previously_recorded_experimental_seed"
        ),
        "used_seed_count": len(used),
        "maximum_used_seed": maximum,
        "selected_fresh_seeds": selected,
        "selected_seeds_absent_from_scan": all(seed not in used for seed in selected),
        "evidence_record_count": len(evidence),
        "evidence_hash": sha256_json(evidence),
    }


def classifier(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise ValueError("frozen classifier requires exactly three paired deltas")
    numbers = [float(value) for value in values]
    wins = sum(value > 0 for value in numbers)
    ties = sum(value == 0 for value in numbers)
    losses = sum(value < 0 for value in numbers)
    mean = sum(numbers) / 3
    if wins == 3:
        label = "CONSISTENT_POSITIVE"
    elif mean > 0 and wins >= 2:
        label = "MAJORITY_POSITIVE"
    elif wins == losses == 0:
        label = "NEUTRAL"
    elif mean < 0 and losses >= 2:
        label = "NEGATIVE"
    else:
        label = "MIXED"
    return {
        "label": label,
        "values": numbers,
        "mean": mean,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def entropy(counts: Sequence[int]) -> float:
    total = sum(int(value) for value in counts)
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts if count
    )


def recursive_sanitize(value: Any, *, path: str = "$") -> list[str]:
    forbidden = {
        "prompt", "question", "gold_answer", "model_answer", "raw_response",
        "response", "endpoint", "api_key", "checkpoint", "cache_path",
    }
    problems: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in forbidden or normalized.endswith("_prompt"):
                problems.append(f"forbidden_key:{path}.{key}")
            problems.extend(recursive_sanitize(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            problems.extend(recursive_sanitize(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "final_answer:" in lowered or "dashscope" in lowered:
            problems.append(f"forbidden_text:{path}")
        if re.search(r"(?:[a-zA-Z]:\\|^/)", value):
            problems.append(f"absolute_path:{path}")
    return problems


def source_inventory() -> tuple[list[dict[str, str]], str]:
    relative_paths = git(
        "ls-files", "multi_dataset_diverse_rl", "scripts", "tests",
        "experiments/manifests/diversity_matrix_d0_d5.yaml",
        "experiments/diversity_matrix_d0_d5_20260903",
    ).splitlines()
    rows = []
    for relative in sorted(relative_paths):
        path = ROOT / relative
        rows.append({
            "path": relative.replace("\\", "/"),
            "sha256": sha256_file(path),
        })
    return rows, sha256_json(rows)


def arm_protocol_payload(arm: str) -> dict[str, Any]:
    from multi_dataset_diverse_rl.config import Config
    from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem

    specification = ARMS[arm]
    protocol = PromptEnsembleOptimizationSystem(Config.from_flat(
        experiment_setting=specification["setting"],
        out_dir="runs/diversity_matrix_protocol_probe",
    )).protocol
    return {
        **specification,
        "optimization_enabled": protocol.optimization_enabled,
        "target_selection_policy": protocol.target_selection_policy,
        "sample_pool_policy": protocol.sample_pool_policy,
        "tcs_context_policy": protocol.tcs_context_policy,
        "candidate_acceptance_policy": protocol.candidate_acceptance_policy,
        "candidate_ranking_policy": protocol.candidate_ranking_policy,
        "responsibility_refresh_policy": protocol.responsibility_refresh_policy,
        "service_routing_enabled": protocol.service_routing_enabled,
        "generic_revision_enabled": protocol.generic_revision_enabled,
        "compatibility_repair_enabled": protocol.compatibility_repair_enabled,
        "target_branch_count": protocol.target_branch_count,
        "candidates_per_target_branch": protocol.candidates_per_target_branch,
        "module2_evolution_variant": protocol.module2_evolution_variant,
    }


def completion_registry(path: Path) -> dict[str, Any]:
    if path.is_file():
        return read_json(path)
    return {
        "registry_version": "diversity_matrix_completed_cells_v1",
        "completed": [],
        "incomplete": [],
    }


def record_completion(
    path: Path, *, seed: int, arm: str, status: str, detail: str = ""
) -> None:
    payload = completion_registry(path)
    key = {"seed": int(seed), "arm": str(arm)}
    payload["completed"] = [
        row for row in payload["completed"]
        if (int(row["seed"]), row["arm"]) != (seed, arm)
    ]
    payload["incomplete"] = [
        row for row in payload["incomplete"]
        if (int(row["seed"]), row["arm"]) != (seed, arm)
    ]
    destination = "completed" if status == "COMPLETED" else "incomplete"
    payload[destination].append({**key, "status": status, "detail": detail})
    payload["completed"] = sorted(payload["completed"], key=lambda row: (row["seed"], row["arm"]))
    payload["incomplete"] = sorted(payload["incomplete"], key=lambda row: (row["seed"], row["arm"]))
    write_json(path, payload)


def count_by_member(decisions: Sequence[Mapping[str, Any]], key: str) -> list[int]:
    counts: Counter[int] = Counter()
    for row in decisions:
        if key == "targets":
            counts.update(map(int, row.get("selected_target_ids", ())))
        elif key == "commits" and row.get("target_agent_id") is not None:
            counts[int(row["target_agent_id"])] += int(bool(row.get("accepted_prompt_hash")))
    return [counts[index] for index in range(AGENTS)]
