from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / "v17_formal_5arm_3seed_20260813"
RUN_ROOT = ROOT / "runs" / "v17_formal_5arm_3seed_20260813"
REPORT_ROOT = ROOT / "reports" / "v17_formal_5arm_3seed_20260813"
MANIFEST = ROOT / "configs" / "task_level_comparison_strict_bbh_seed42.yaml"
SPLIT_ROOT = ROOT / "strict_splits_bbh_seed42" / "disambiguation_qa"

SEEDS = (56, 57, 58)
ARMS = {
    "S0": "shared_static_reference",
    "S1": "experimental_v17_formal_generic_2x2_matched",
    "S2": "experimental_v16_efficacy_g_matched",
    "S3": "experimental_v16_efficacy_r_m20",
    "S4": "experimental_v16_efficacy_r_m2f",
}
EXECUTION_ORDER = {
    56: ("S0", "S1", "S2", "S3", "S4"),
    57: ("S2", "S3", "S4", "S0", "S1"),
    58: ("S4", "S0", "S1", "S2", "S3"),
}
EXPECTED_SPLITS = {
    "opt": (75, "608b9b56cd3eaef2f57d982fd503670188459017fbff73b4e555f1d3a0820bed"),
    "val": (50, "9ed73e02600b357a73ee03927353737d09b8017479fb0c23bc1a47b3145bda8d"),
    "test": (125, "480a8b99041366ae8e4527da66d7d98fa3eaa16aebb2dbc70cca29e5c85e9b01"),
}
CALL_CEILING = 8000
TOKEN_CEILING = 3_000_000
UPDATES = 8


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def tracked_source_inventory() -> tuple[list[dict[str, str]], str]:
    paths = git("ls-files", "multi_dataset_diverse_rl", "scripts", "tests").splitlines()
    combined = hashlib.sha256()
    rows: list[dict[str, str]] = []
    for relative in sorted(paths):
        normalized = relative.replace("\\", "/")
        digest = sha256_file(ROOT / relative)
        rows.append({"path": normalized, "sha256": digest})
        combined.update(normalized.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return rows, combined.hexdigest()


def split_freeze() -> dict[str, Any]:
    info = read_json(SPLIT_ROOT / "split_info.json")
    with (ROOT / "strict_splits_bbh_seed42" / "summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        summary = [
            row for row in csv.DictReader(handle)
            if row["task_id"] == "disambiguation_qa"
        ]
    by_split = {row["split"]: row for row in summary}
    result: dict[str, Any] = {}
    id_sets: dict[str, set[int]] = {}
    errors: list[str] = []
    for name, (expected_count, expected_hash) in EXPECTED_SPLITS.items():
        row = by_split.get(name)
        if row is None:
            errors.append(f"missing_summary_split:{name}")
            continue
        ids = [int(value) for value in row["sample_ids"].split(",") if value.strip()]
        id_sets[name] = set(ids)
        recorded = info.get("splits", {}).get(name, {})
        if len(ids) != expected_count or int(recorded.get("num_samples", -1)) != expected_count:
            errors.append(f"split_size:{name}")
        if row.get("hash") != expected_hash or recorded.get("hash") != expected_hash:
            errors.append(f"split_hash:{name}")
        result[name] = {
            "row_count": len(ids),
            "frozen_split_hash": expected_hash,
            "sample_id_set_hash": sha256_json(sorted(ids)),
            "file_sha256": sha256_file(SPLIT_ROOT / f"{name}.csv"),
        }
    overlaps = {
        "opt_val": len(id_sets.get("opt", set()) & id_sets.get("val", set())),
        "opt_test": len(id_sets.get("opt", set()) & id_sets.get("test", set())),
        "val_test": len(id_sets.get("val", set()) & id_sets.get("test", set())),
    }
    if any(overlaps.values()):
        errors.append("split_overlap")
    return {
        "task": "disambiguation_qa",
        "benchmark": "BBH",
        "dataset_format": "mars",
        "manifest_sha256": sha256_file(MANIFEST),
        "split_info_sha256": sha256_file(SPLIT_ROOT / "split_info.json"),
        "splits": result,
        "sample_id_overlaps": overlaps,
        "gate": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def formal_target_schedule(seed: int) -> list[list[int]]:
    start = int(seed) % 5
    return [[(start + 2 * update) % 5, (start + 2 * update + 1) % 5] for update in range(UPDATES)]


def classify_three_seed(deltas: Iterable[float]) -> dict[str, Any]:
    values = tuple(float(value) for value in deltas)
    if len(values) != 3:
        raise ValueError("the frozen classifier requires exactly three deltas")
    wins = sum(value > 0 for value in values)
    ties = sum(value == 0 for value in values)
    losses = sum(value < 0 for value in values)
    mean = sum(values) / 3
    if mean > 0 and wins == 3:
        label = "CONSISTENT_POSITIVE"
    elif mean > 0 and wins > losses and wins < 3:
        label = "MAJORITY_POSITIVE"
    elif mean > 0 and wins <= losses:
        label = "POSITIVE_MEAN_HETEROGENEOUS"
    elif mean <= 0 and wins > losses:
        label = "MIXED_NONPOSITIVE"
    else:
        label = "NOT_SUPPORTED"
    return {
        "label": label, "mean_delta": mean,
        "wins": wins, "ties": ties, "losses": losses,
    }


def source_semantics_diff() -> dict[str, Any]:
    names = git(
        "diff", "--name-only",
        "ccfc213c34d0c419f9e56844e1d4423329cd5f0e",
        "f4be41c960aa9f052ac7d1de2a9cf23bde4fd95f",
    ).splitlines()
    allowed_prefix = "reports/v16_module2_compute_matched_efficacy_20260813/"
    unexpected = [name for name in names if not name.replace("\\", "/").startswith(allowed_prefix)]
    return {
        "from_commit": "ccfc213c34d0c419f9e56844e1d4423329cd5f0e",
        "to_commit": "f4be41c960aa9f052ac7d1de2a9cf23bde4fd95f",
        "changed_files": names,
        "unexpected_execution_semantic_files": unexpected,
        "R_M20_SEMANTICS_CHANGED": bool(unexpected),
        "R_M2F_SEMANTICS_CHANGED": bool(unexpected),
        "MODULE1_SEMANTICS_CHANGED": bool(unexpected),
        "COMMON_SAFE_CHANGED": bool(unexpected),
        "gate": "PASS" if not unexpected else "FAIL",
    }


def formal_seed_prior_use() -> dict[str, Any]:
    patterns = tuple(
        rf"seeds?[^0-9]{{1,3}}{seed}([^0-9]|$)" for seed in SEEDS
    )
    matches: list[str] = []
    for pattern in patterns:
        process = subprocess.run(
            [
                "git", "grep", "-n", "-i", "-E", pattern,
                "f4be41c960aa9f052ac7d1de2a9cf23bde4fd95f", "--",
                ":!strict_splits_bbh_seed42/**", ":!Dataset/**",
            ],
            cwd=ROOT, text=True, encoding="utf-8",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if process.returncode not in (0, 1):
            raise RuntimeError(process.stderr.strip() or "git grep failed")
        matches.extend(line for line in process.stdout.splitlines() if line)
    unique = sorted(set(matches))
    return {
        "starting_commit": "f4be41c960aa9f052ac7d1de2a9cf23bde4fd95f",
        "formal_seeds": list(SEEDS),
        "prior_use_count": len(unique),
        "matched_paths": sorted({line.split(":", 2)[1] for line in unique}),
        "gate": "PASS" if not unique else "FAIL",
    }


def path_within_root(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents


def recursive_sanitize(value: Any, *, path: str = "$") -> list[str]:
    forbidden_keys = {
        "prompt", "question", "gold_answer", "model_answer", "raw_response",
        "response", "trace", "endpoint", "api_key", "checkpoint_payload",
    }
    problems: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in forbidden_keys or normalized.endswith("_prompt"):
                problems.append(f"forbidden_key:{path}.{key}")
            problems.extend(recursive_sanitize(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            problems.extend(recursive_sanitize(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "final_answer:" in lowered or "dashscope_base_url" in lowered:
            problems.append(f"forbidden_text:{path}")
        if ":\\" in value or value.startswith("/"):
            problems.append(f"absolute_path:{path}")
    return problems
