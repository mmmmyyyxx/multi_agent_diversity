from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.audit_diversity_matrix_split_balance import (
    FILES,
    ROOT,
    fit_lexical_clusters,
    load_items,
    static_difficulty,
    all_outcomes,
    standardized_difference,
    text_features,
    total_variation,
)
from scripts.diversity_matrix_d0_d5_support import seed_registry_scan


DESIGN_ROOT = ROOT / "experiments" / "anti_overfitting_split_v1"
DEFAULT_PREP_ROOT = ROOT / "runs" / "anti_overfitting_shadow_gate_v1_prep_20260904"
DEFAULT_RUN_ROOT = ROOT / "runs" / "anti_overfitting_shadow_gate_v1_20260904"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "anti_overfitting_shadow_gate_v1"
SOURCE_ROOT = ROOT / "strict_splits_bbh_seed42" / "disambiguation_qa"
SPLIT_SEED = 20260904
BUCKETS = ("fold_a", "fold_b", "fold_c", "validation", "test")
FOLD_MAP = (
    ("fold_a+fold_b", "fold_c"),
    ("fold_a+fold_c", "fold_b"),
    ("fold_b+fold_c", "fold_a"),
)
ARMS = ("RR_GENERIC_OLD_PROTOCOL", "RR_GENERIC_SHADOW_GATED")
SOLVER_MODEL = "qwen3-8b"
ROLE_MODEL = "qwen3.7-flash"
MAX_UPDATE_OPPORTUNITIES = 32
AUTH_ENV = "ANTI_OVERFITTING_SHADOW_AUTHORIZED"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def source_items() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    grouped = load_items()
    safe = [row for group in grouped.values() for row in group]
    raw: dict[str, dict[str, str]] = {}
    for filename in FILES.values():
        with (SOURCE_ROOT / filename).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                digest = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()
                raw[digest] = dict(row)
    if len(safe) != 250 or len(raw) != 250:
        raise ValueError("DQA dataset inventory must be exactly 250 unique rows")
    return safe, raw


def metadata() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    items, raw = source_items()
    grouped = load_items()
    clusters = fit_lexical_clusters(grouped)
    outcomes = all_outcomes(grouped)
    split_by_hash = {
        str(row["question_hash"]): split
        for split, rows in grouped.items() for row in rows
    }
    for row in items:
        digest = str(row["question_hash"])
        split = split_by_hash[digest]
        correct = sum(int(outcomes[split][seed]["D0"][digest]["vote_correct"]) for seed in (72, 73, 74))
        row["static_difficulty_bin"] = f"correct_{correct}_of_3"
        row["lexical_cluster"] = int(clusters[digest])
    return items, raw


def _stable_rank(digest: str, bucket: str = "") -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{bucket}:{digest}".encode()).hexdigest()


def construct_assignment(items: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Greedy marginal stratification with one frozen ordering and no seed search."""
    dimensions = {
        "label": sorted({str(row["gold_label"]) for row in items}),
        "difficulty": sorted({str(row["static_difficulty_bin"]) for row in items}),
        "cluster": sorted({str(row["lexical_cluster"]) for row in items}),
    }
    totals = {
        (name, value): sum(str(row[{"label": "gold_label", "difficulty": "static_difficulty_bin", "cluster": "lexical_cluster"}[name]]) == value for row in items)
        for name, values in dimensions.items() for value in values
    }
    field = {"label": "gold_label", "difficulty": "static_difficulty_bin", "cluster": "lexical_cluster"}
    rarity = lambda row: sum(1 / max(1, totals[(name, str(row[field[name]]))]) for name in dimensions)
    ordered = sorted(items, key=lambda row: (-rarity(row), _stable_rank(str(row["question_hash"]))))
    assigned = {bucket: [] for bucket in BUCKETS}
    counts = {bucket: Counter() for bucket in BUCKETS}
    for row in ordered:
        choices = []
        for bucket in BUCKETS:
            if len(assigned[bucket]) >= 50:
                continue
            score = 0.0
            for name in dimensions:
                value = str(row[field[name]])
                # Allocate each marginal category to the bucket with the
                # smallest fraction of that category already consumed.  A
                # naive squared-distance-to-final-quota greedy packs early
                # rows into one bucket because every bucket begins below its
                # final quota; this load ratio has the required online
                # balancing behavior.
                score += counts[bucket][(name, value)] / max(1, totals[(name, value)])
            score += len(assigned[bucket]) / len(items)
            choices.append((score, _stable_rank(str(row["question_hash"]), bucket), bucket))
        bucket = min(choices)[2]
        digest = str(row["question_hash"])
        assigned[bucket].append(digest)
        for name in dimensions:
            counts[bucket][(name, str(row[field[name]]))] += 1
    if any(len(rows) != 50 for rows in assigned.values()):
        raise AssertionError("deterministic split did not produce five 50-row buckets")
    if len({digest for rows in assigned.values() for digest in rows}) != 250:
        raise AssertionError("deterministic split is not a partition")
    return {key: sorted(value) for key, value in assigned.items()}


def _distribution(rows: Sequence[Mapping[str, Any]], field: str, values: Sequence[str]) -> dict[str, float]:
    counts = Counter(str(row[field]) for row in rows)
    return {value: counts[value] / len(rows) for value in values}


def balance_report(items: Sequence[Mapping[str, Any]], assignment: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    by_hash = {str(row["question_hash"]): row for row in items}
    grouped = {bucket: [by_hash[digest] for digest in hashes] for bucket, hashes in assignment.items()}
    labels = ("A", "B", "C")
    difficulties = tuple(sorted({str(row["static_difficulty_bin"]) for row in items}))
    clusters = tuple(map(str, range(8)))
    result: dict[str, Any] = {
        "schema_version": "anti_overfitting_split_balance_v1",
        "difficulty_source": "frozen D0 Static predictions, seeds 72/73/74, qwen3-14b; outcomes used only for stratification",
        "lexical_cluster_source": "TF-IDF word 1-2 grams, KMeans k=8 random_state=42 n_init=20; no labels or outcomes",
        "structural_feature_source": "question core sentence aggregate statistics only",
        "buckets": {}, "pairwise": [],
    }
    for bucket, rows in grouped.items():
        result["buckets"][bucket] = {
            "count": len(rows),
            "label": _distribution(rows, "gold_label", labels),
            "static_difficulty": _distribution(rows, "static_difficulty_bin", difficulties),
            "lexical_cluster": _distribution(rows, "lexical_cluster", clusters),
            "mean_static_vote_accuracy": sum(int(str(row["static_difficulty_bin"])[8]) / 3 for row in rows) / len(rows),
            "structural_means": {key: sum(float(row["features"][key]) for row in rows) / len(rows) for key in rows[0]["features"]},
        }
    for left_index, left in enumerate(BUCKETS):
        for right in BUCKETS[left_index + 1:]:
            smds = {
                key: standardized_difference(
                    [float(row["features"][key]) for row in grouped[left]],
                    [float(row["features"][key]) for row in grouped[right]],
                ) for key in grouped[left][0]["features"]
            }
            result["pairwise"].append({
                "left": left, "right": right,
                "label_tv": total_variation(result["buckets"][left]["label"], result["buckets"][right]["label"]),
                "difficulty_tv": total_variation(result["buckets"][left]["static_difficulty"], result["buckets"][right]["static_difficulty"]),
                "cluster_tv": total_variation(result["buckets"][left]["lexical_cluster"], result["buckets"][right]["lexical_cluster"]),
                "max_abs_structural_smd": max(abs(value) for value in smds.values()),
            })
    result["thresholds"] = {"max_label_tv": 0.10, "max_difficulty_tv": 0.12, "max_cluster_tv": 0.22, "max_structural_smd": 0.50}
    observed = {
        "max_label_tv": max(row["label_tv"] for row in result["pairwise"]),
        "max_difficulty_tv": max(row["difficulty_tv"] for row in result["pairwise"]),
        "max_cluster_tv": max(row["cluster_tv"] for row in result["pairwise"]),
        "max_structural_smd": max(row["max_abs_structural_smd"] for row in result["pairwise"]),
    }
    result["observed_maxima"] = observed
    result["gate"] = "PASS" if all(observed[key] <= value for key, value in result["thresholds"].items()) else "HOLD"
    return result


def export_private_splits(raw: Mapping[str, Mapping[str, str]], assignment: Mapping[str, Sequence[str]], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for bucket, hashes in assignment.items():
        path = root / f"{bucket}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(
                {"question": raw[digest]["question"], "answer": raw[digest]["answer"]}
                for digest in hashes
            )


def fresh_seed_freeze() -> dict[str, Any]:
    task_outputs = [
        path for root in (DESIGN_ROOT, DEFAULT_REPORT_ROOT)
        if root.exists() for path in root.rglob("*") if path.is_file()
    ]
    task_outputs.append(
        ROOT / "experiments" / "manifests" / "anti_overfitting_shadow_gate_v1.yaml"
    )
    return seed_registry_scan(exclude_paths=task_outputs)
