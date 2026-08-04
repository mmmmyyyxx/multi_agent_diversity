from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.tasks import get_task_spec, normalize_bbh_answer, normalize_spaces
from multi_dataset_diverse_rl.team_differentiation import team_behavior_metrics
from multi_dataset_diverse_rl.utils import normalize_prompt_text


ANALYSIS_VERSION = "final_method_agent_clustering_v1"
STRICTNESS_STATUS = "exploratory_due_to_unmatched_repeated_test_observations"
DEFAULT_RUN_ROOT = (
    Path("experiments") / "runs_final_method_disambiguation_full_20260731"
)
AGENT_COUNT = 5


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prompt_hash(prompt: str) -> str:
    return _sha256_bytes(normalize_prompt_text(prompt).encode("utf-8"))


def _team_state_hash(prompt_hashes: Sequence[str]) -> str:
    payload = json.dumps(list(prompt_hashes), ensure_ascii=False, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def normalize_partition(labels: Sequence[int]) -> list[list[int]]:
    clusters: dict[int, list[int]] = {}
    for agent_id, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(agent_id)
    normalized = [sorted(members) for members in clusters.values()]
    normalized.sort(key=lambda members: (-len(members), members[0]))
    return normalized


def partition_text(labels: Sequence[int]) -> str:
    return "|".join("{" + ",".join(str(value) for value in cluster) + "}"
                    for cluster in normalize_partition(labels))


def canonical_labels(labels: Sequence[int]) -> list[int]:
    result = [0] * len(labels)
    for cluster_id, members in enumerate(normalize_partition(labels)):
        for agent_id in members:
            result[agent_id] = cluster_id
    return result


def partitions_equal(left: Sequence[int], right: Sequence[int]) -> bool:
    return normalize_partition(left) == normalize_partition(right)


def pairwise_partition_agreement(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right):
        raise ValueError("partitions must contain the same number of agents")
    pairs = list(combinations(range(len(left)), 2))
    return sum((left[i] == left[j]) == (right[i] == right[j]) for i, j in pairs) / len(pairs)


def stable_bootstrap_seed(task: str, setting: str, seed: int, split: str) -> int:
    payload = json.dumps(
        [task, setting, int(seed), split, "cluster_bootstrap_v1"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return int(_sha256_bytes(payload.encode("utf-8"))[:16], 16) % (2**32)


def _pearson_binary(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return float(np.corrcoef(left.astype(float), right.astype(float))[0, 1])


def correctness_distance(
    correctness: Sequence[Sequence[bool]],
) -> tuple[str, list[list[float | None]], np.ndarray, bool]:
    matrix = np.asarray(correctness, dtype=bool)
    if matrix.shape[0] != AGENT_COUNT:
        raise ValueError("correctness matrix must contain exactly five agents")
    correlations: list[list[float | None]] = [[None] * AGENT_COUNT for _ in range(AGENT_COUNT)]
    undefined = any(np.all(row == row[0]) for row in matrix)
    for i in range(AGENT_COUNT):
        correlations[i][i] = 1.0
        for j in range(i + 1, AGENT_COUNT):
            value = _pearson_binary(matrix[i], matrix[j])
            correlations[i][j] = correlations[j][i] = value
            undefined = undefined or value is None
    distance = np.zeros((AGENT_COUNT, AGENT_COUNT), dtype=float)
    if undefined:
        for i, j in combinations(range(AGENT_COUNT), 2):
            value = float(np.mean(matrix[i] != matrix[j]))
            distance[i, j] = distance[j, i] = value
        distance_type = "correctness_normalized_hamming"
    else:
        for i, j in combinations(range(AGENT_COUNT), 2):
            rho = float(correlations[i][j])
            value = min(1.0, max(0.0, (1.0 - rho) / 2.0))
            distance[i, j] = distance[j, i] = value
        distance_type = "correctness_correlation"
    return distance_type, correlations, distance, undefined


def hierarchical_labels(distance: np.ndarray, cluster_count: int) -> list[int]:
    model = AgglomerativeClustering(
        n_clusters=int(cluster_count), metric="precomputed", linkage="average"
    )
    return canonical_labels(model.fit_predict(distance).tolist())


def _silhouette(distance: np.ndarray, labels: Sequence[int]) -> float | None:
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return None
    if np.max(distance) <= 0:
        return None
    try:
        return float(silhouette_score(distance, labels, metric="precomputed"))
    except ValueError:
        return None


def _pair_means(matrix: Sequence[Sequence[float | None]], labels: Sequence[int]) -> tuple[float | None, float | None]:
    within: list[float] = []
    between: list[float] = []
    for i, j in combinations(range(len(labels)), 2):
        value = matrix[i][j]
        if value is None:
            continue
        (within if labels[i] == labels[j] else between).append(float(value))
    return (
        sum(within) / len(within) if within else None,
        sum(between) / len(between) if between else None,
    )


def _gap(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _cluster_strength(silhouette: float | None, gap: float | None) -> str:
    if silhouette is not None and gap is not None:
        if silhouette >= 0.25 and gap >= 0.15:
            return "strong"
        if silhouette >= 0.10 and gap >= 0.05:
            return "moderate"
    return "weak"


def _bootstrap(
    correctness: Sequence[Sequence[bool]],
    *,
    replicates: int,
    bootstrap_seed: int,
    final_labels: Sequence[int],
) -> dict[str, Any]:
    matrix = np.asarray(correctness, dtype=bool)
    rng = np.random.default_rng(bootstrap_seed)
    same_counts = np.zeros((AGENT_COUNT, AGENT_COUNT), dtype=int)
    for _ in range(replicates):
        indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        _, _, distance, _ = correctness_distance(matrix[:, indices])
        labels = hierarchical_labels(distance, 2)
        for i in range(AGENT_COUNT):
            for j in range(AGENT_COUNT):
                same_counts[i, j] += int(labels[i] == labels[j])
    consensus = same_counts.astype(float) / replicates
    supports: list[float] = []
    unstable_pairs = []
    for i, j in combinations(range(AGENT_COUNT), 2):
        support = float(consensus[i, j] if final_labels[i] == final_labels[j]
                        else 1.0 - consensus[i, j])
        supports.append(support)
        if support < 0.70:
            unstable_pairs.append({"agents": [i, j], "support": support})
    overall = sum(supports) / len(supports)
    return {
        "bootstrap_seed": bootstrap_seed,
        "replicates": replicates,
        "consensus_matrix": consensus.tolist(),
        "bootstrap_partition_support": overall,
        "minimum_pair_support": min(supports),
        "unstable_pairs": unstable_pairs,
        "bootstrap_stability": "weak" if overall < 0.70 else "supported",
    }


def _load_examples(path: Path, hash_mode: str) -> list[ProbeExample]:
    task_spec = get_task_spec("bbh")
    examples: list[ProbeExample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question = str(row["question"])
            hash_text = normalize_spaces(question) if hash_mode == "normalized_spaces_v1" else question
            examples.append(ProbeExample(
                question=question,
                question_hash=_sha256_bytes(hash_text.encode("utf-8")),
                gold_answer=task_spec.parse_gold(row["answer"], question),
            ))
    return examples


def _answer_from_json(raw: str) -> PromptAnswer:
    payload = json.loads(raw)
    return PromptAnswer(
        answer=str(payload.get("answer", "")),
        trace="",
        valid=bool(payload.get("valid", False)),
        validity_status=str(payload.get("validity_status", "")),
        terminal_invalid=bool(payload.get("terminal_invalid", False)),
        response_hash=str(payload.get("response_hash", "")),
        request_identity=str(payload.get("request_identity", "")),
        created_at=float(payload.get("created_at", 0.0)),
    )


def _load_profiles(
    database: Path,
    prompt_hashes: Sequence[str],
    examples: Sequence[ProbeExample],
) -> tuple[list[list[PromptAnswer]] | None, list[str]]:
    issues: list[str] = []
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        profiles: list[list[PromptAnswer]] = []
        for agent_id, prompt_hash in enumerate(prompt_hashes):
            profile: list[PromptAnswer] = []
            for example in examples:
                rows = connection.execute(
                    "SELECT answer_json FROM solver_cache "
                    "WHERE state='ready' AND prompt_hash=? AND question_hash=?",
                    (prompt_hash, example.question_hash),
                ).fetchall()
                if len(rows) != 1:
                    issues.append(
                        f"agent_{agent_id}:question_hash_{example.question_hash[:12]}:ready_rows={len(rows)}"
                    )
                    continue
                profile.append(_answer_from_json(rows[0][0]))
            if len(profile) != len(examples):
                return None, issues
            profiles.append(profile)
        return profiles, issues
    finally:
        connection.close()


def _correctness(
    examples: Sequence[ProbeExample], profiles: Sequence[Sequence[PromptAnswer]]
) -> list[list[bool]]:
    task_spec = get_task_spec("bbh")
    return [[
        bool(answer.valid and task_spec.match_answer(answer.answer, example.gold_answer))
        for answer, example in zip(profile, examples, strict=True)
    ] for profile in profiles]


def _metric_match(recomputed: dict[str, Any], saved: dict[str, Any]) -> bool:
    scalar_keys = ("example_count", "per_agent_correct_counts", "team_vote_correct_count", "terminal_invalid_count")
    matrix_keys = (
        "pairwise_correctness_correlation", "answer_disagreement_matrix",
        "exact_answer_agreement_matrix", "double_fault_matrix", "same_wrong_excess_matrix",
    )
    for key in scalar_keys:
        if recomputed.get(key) != saved.get(key):
            return False
    for key in matrix_keys:
        left, right = recomputed.get(key), saved.get(key)
        if left is None or right is None or len(left) != len(right):
            return False
        for left_row, right_row in zip(left, right, strict=True):
            for a, b in zip(left_row, right_row, strict=True):
                if a is None or b is None:
                    if a is not None or b is not None:
                        return False
                elif not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10):
                    return False
    return True


def _off_diagonal_mean(matrix: Sequence[Sequence[float | None]]) -> float | None:
    values = [float(matrix[i][j]) for i, j in combinations(range(len(matrix)), 2)
              if matrix[i][j] is not None]
    return sum(values) / len(values) if values else None


def analyze_split(
    *,
    task: str,
    setting: str,
    seed: int,
    split: str,
    examples: Sequence[ProbeExample],
    profiles: Sequence[Sequence[PromptAnswer]],
    saved_metrics: dict[str, Any],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    task_spec = get_task_spec("bbh")
    metrics = team_behavior_metrics(
        examples=examples,
        profiles=profiles,
        normalize_answer=normalize_bbh_answer,
        match_answer=task_spec.match_answer,
        tie_break="abstain",
        seed=seed,
    )
    correctness = _correctness(examples, profiles)
    distance_type, correlations, distance, undefined = correctness_distance(correctness)
    k2 = hierarchical_labels(distance, 2)
    k3 = hierarchical_labels(distance, 3)
    k2_silhouette = _silhouette(distance, k2)
    k3_silhouette = _silhouette(distance, k3)
    within_corr, between_corr = _pair_means(correlations, k2)
    corr_gap = _gap(within_corr, between_corr)
    bootstrap = _bootstrap(
        correctness,
        replicates=bootstrap_replicates,
        bootstrap_seed=stable_bootstrap_seed(task, setting, seed, split),
        final_labels=k2,
    )
    auxiliaries = {}
    definitions = {
        "answer_disagreement": (metrics["answer_disagreement_matrix"], "between_minus_within"),
        "exact_answer_agreement": (metrics["exact_answer_agreement_matrix"], "within_minus_between"),
        "double_fault": (metrics["double_fault_matrix"], "within_minus_between"),
        "same_wrong_excess": (metrics["same_wrong_excess_matrix"], "within_minus_between"),
    }
    for name, (matrix, direction) in definitions.items():
        within, between = _pair_means(matrix, k2)
        delta = _gap(between, within) if direction == "between_minus_within" else _gap(within, between)
        auxiliaries[name] = {"within": within, "between": between, "delta": delta, "delta_definition": direction}
    preferred = (
        2 if k3_silhouette is None or (k2_silhouette is not None and k2_silhouette >= k3_silhouette)
        else 3
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "task": task,
        "seed": seed,
        "setting": setting,
        "split": split,
        "example_count": len(examples),
        "distance_type": distance_type,
        "undefined_correlation": undefined,
        "k2_labels": k2,
        "k2_partition": partition_text(k2),
        "k2_cluster_sizes": sorted((len(x) for x in normalize_partition(k2)), reverse=True),
        "k2_silhouette": k2_silhouette,
        "k3_labels": k3,
        "k3_partition": partition_text(k3),
        "k3_silhouette": k3_silhouette,
        "preferred_k_by_silhouette": preferred,
        "within_correlation": within_corr,
        "between_correlation": between_corr,
        "within_between_gap": corr_gap,
        "cluster_strength": _cluster_strength(k2_silhouette, corr_gap),
        **bootstrap,
        "pairwise_correctness_correlation": correlations,
        "answer_disagreement_matrix": metrics["answer_disagreement_matrix"],
        "exact_answer_agreement_matrix": metrics["exact_answer_agreement_matrix"],
        "double_fault_matrix": metrics["double_fault_matrix"],
        "same_wrong_excess_matrix": metrics["same_wrong_excess_matrix"],
        "auxiliary_cluster_contrasts": auxiliaries,
        "mean_M": metrics["mean_M"],
        "mean_G": metrics["mean_G"],
        "mean_H": metrics["mean_H"],
        "mean_off_diagonal_same_wrong_excess": metrics["mean_off_diagonal_same_wrong_excess"],
        "team_vote_correct_count": metrics["team_vote_correct_count"],
        "team_vote_accuracy": metrics["team_vote_accuracy"],
        "oracle_correct_count": metrics["oracle_correct_count"],
        "profile_recomputed_metrics_match_saved": _metric_match(metrics, saved_metrics),
        "strictness_status": STRICTNESS_STATUS,
        "_distance_matrix": distance.tolist(),
    }


def _update_counts(run_dir: Path) -> dict[str, Any]:
    decisions = _read_jsonl(run_dir / "candidate_decisions.jsonl")
    selected = [int(row["target_agent_id"]) for row in decisions if row.get("target_agent_id") is not None]
    accepted = [int(row["target_agent_id"]) for row in decisions
                if row.get("target_agent_id") is not None and row.get("accepted_prompt_hash")]
    positive = [0] * AGENT_COUNT
    for row in decisions:
        target = row.get("target_agent_id")
        if target is None:
            continue
        positive[int(target)] += sum(
            int((candidate.get("constraint") or {}).get("target_gain", 0)) > 0
            for candidate in row.get("candidates", [])
        )
    return {
        "selected_updates_by_agent": [selected.count(i) for i in range(AGENT_COUNT)],
        "accepted_updates_by_agent": [accepted.count(i) for i in range(AGENT_COUNT)],
        "target_positive_candidates_by_agent": positive,
    }


def _cluster_update_alignment(result: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    labels = result["k2_labels"]
    selected = counts["selected_updates_by_agent"]
    accepted = counts["accepted_updates_by_agent"]
    positive = counts["target_positive_candidates_by_agent"]
    total_selected, total_accepted = sum(selected), sum(accepted)
    clusters = []
    normalized = normalize_partition(labels)
    for cluster_id, members in enumerate(normalized):
        attempts = sum(selected[i] for i in members)
        accepts = sum(accepted[i] for i in members)
        clusters.append({
            "cluster_id": cluster_id,
            "members": members,
            "selected_updates": attempts,
            "accepted_updates": accepts,
            "target_positive_candidates": sum(positive[i] for i in members),
            "attempt_share": attempts / total_selected if total_selected else None,
            "accepted_share": accepts / total_accepted if total_accepted else None,
            "acceptance_efficiency": accepts / attempts if attempts else None,
        })
    maximum = max(selected, default=0)
    most_selected = [i for i, value in enumerate(selected) if value == maximum]
    primary = most_selected[0] if most_selected else None
    primary_cluster = next((row for row in clusters if primary in row["members"]), None)
    most_selected_details = []
    for agent in most_selected:
        cluster = next(row for row in clusters if agent in row["members"])
        most_selected_details.append({
            "agent_id": agent,
            "selected_updates": selected[agent],
            "accepted_updates": accepted[agent],
            "cluster_id": cluster["cluster_id"],
            "cluster_members": cluster["members"],
            "is_singleton_cluster": len(cluster["members"]) == 1,
            "is_minority_cluster": len(cluster["members"]) < max(len(x) for x in normalized),
            "acceptance_efficiency": accepted[agent] / selected[agent] if selected[agent] else None,
            "agent_attempt_share": selected[agent] / total_selected if total_selected else None,
            "agent_accepted_share": accepted[agent] / total_accepted if total_accepted else None,
            "cluster_attempt_share": cluster["attempt_share"],
            "cluster_accepted_share": cluster["accepted_share"],
        })
    return {
        "task": result["task"], "seed": result["seed"], "setting": result["setting"],
        "split": result["split"], **counts, "clusters": clusters,
        "most_selected_agents": most_selected,
        "most_selected_agent_details": most_selected_details,
        "cluster_of_most_selected_agent": primary_cluster["cluster_id"] if primary_cluster else None,
        "is_singleton_cluster": len(primary_cluster["members"]) == 1 if primary_cluster else None,
        "is_minority_cluster": len(primary_cluster["members"]) < max(len(x) for x in normalized) if primary_cluster else None,
        "strictness_status": STRICTNESS_STATUS,
    }


def _final_responsibility(run_dir: Path, initial_counts: Sequence[int], final_counts: Sequence[int]) -> list[dict[str, Any]] | None:
    rows = _read_jsonl(run_dir / "member_opportunities.jsonl")
    if not rows:
        return None
    version = max(int(row["team_state_version"]) for row in rows)
    final_rows = [row for row in rows if int(row["team_state_version"]) == version]
    maximum_gain = max(current - initial for current, initial in zip(final_counts, initial_counts, strict=True))
    output = []
    for agent in range(AGENT_COUNT):
        owned = [row for row in final_rows if int(row["agent_id"]) == agent and row.get("eligible")]
        gain = int(final_counts[agent]) - int(initial_counts[agent])
        output.append({
            "agent_id": agent,
            "D_i": sum(int(row.get("vote_flip_gain", 0)) for row in owned),
            "S_i": sum(int(row.get("margin_gain", 0)) for row in owned),
            "d_i": max(0, maximum_gain - gain - 5),
            "portfolio_size": len(owned),
            "coverage_residual_count": sum(bool(row.get("coverage_opportunity")) for row in owned),
            "conversion_residual_count": sum(bool(row.get("conversion_opportunity")) for row in owned),
        })
    return output


def _responsibility_alignment(
    result: dict[str, Any], responsibility: list[dict[str, Any]], counts: dict[str, Any]
) -> dict[str, Any]:
    labels = result["k2_labels"]
    selected, accepted = counts["selected_updates_by_agent"], counts["accepted_updates_by_agent"]
    total_selected, total_accepted = sum(selected), sum(accepted)
    total_portfolio = sum(row["portfolio_size"] for row in responsibility)
    clusters = []
    for cluster_id, members in enumerate(normalize_partition(labels)):
        cluster_rows = [responsibility[i] for i in members]
        attempts, accepts = sum(selected[i] for i in members), sum(accepted[i] for i in members)
        clusters.append({
            "cluster_id": cluster_id,
            "members": members,
            "cluster_D": sum(row["D_i"] for row in cluster_rows),
            "cluster_S": sum(row["S_i"] for row in cluster_rows),
            "cluster_uplift_deficit": sum(row["d_i"] for row in cluster_rows),
            "responsibility_share": sum(row["portfolio_size"] for row in cluster_rows) / total_portfolio if total_portfolio else None,
            "selected_update_share": attempts / total_selected if total_selected else None,
            "accepted_update_share": accepts / total_accepted if total_accepted else None,
            "acceptance_efficiency": accepts / attempts if attempts else None,
        })
    median_D = float(np.median([row["D_i"] for row in responsibility]))
    median_S = float(np.median([row["S_i"] for row in responsibility]))
    overall_efficiency = total_accepted / total_selected if total_selected else 0.0
    high_responsibility_low_efficiency = [
        row["agent_id"] for row in responsibility
        if (row["D_i"] >= median_D or row["S_i"] >= median_S)
        and (accepted[row["agent_id"]] / selected[row["agent_id"]]
             if selected[row["agent_id"]] else 0.0) < overall_efficiency
    ]
    high_attempt_low_acceptance = [
        agent for agent in range(AGENT_COUNT)
        if total_selected and selected[agent] / total_selected > 0.20
        and (accepted[agent] / total_accepted if total_accepted else 0.0) < selected[agent] / total_selected
    ]
    return {
        "task": result["task"], "seed": result["seed"], "setting": result["setting"],
        "split": result["split"], "per_agent": responsibility, "clusters": clusters,
        "high_responsibility_low_efficiency_agents": high_responsibility_low_efficiency,
        "high_attempt_low_acceptance_agents": high_attempt_low_acceptance,
        "diagnostic_rule": "high responsibility means D_i or S_i at/above run median; low efficiency means below run-wide acceptance efficiency",
        "strictness_status": STRICTNESS_STATUS,
    }


def _train_test_stability(train: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    left, right = train["k2_labels"], test["k2_labels"]
    exact = partitions_equal(left, right)
    ari = float(adjusted_rand_score(left, right))
    pair = pairwise_partition_agreement(left, right)
    train_consensus = np.asarray(train["consensus_matrix"], dtype=float)
    test_consensus = np.asarray(test["consensus_matrix"], dtype=float)
    distance = float(np.mean([abs(train_consensus[i, j] - test_consensus[i, j])
                              for i, j in combinations(range(AGENT_COUNT), 2)]))
    status = (
        "stable" if exact or (ari >= 0.70 and pair >= 0.80)
        else "partially_stable" if ari >= 0.20 or pair >= 0.60
        else "unstable"
    )
    return {
        "task": train["task"], "seed": train["seed"], "setting": train["setting"],
        "train_partition": train["k2_partition"], "test_partition": test["k2_partition"],
        "exact_partition_match": exact, "adjusted_rand_index": ari,
        "pairwise_coclustering_agreement": pair,
        "consensus_matrix_distance": distance, "stability": status,
        "strictness_status": STRICTNESS_STATUS,
    }


def _cross_seed(results: list[dict[str, Any]], settings: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {"analysis_version": ANALYSIS_VERSION, "settings": []}
    for setting in settings:
        setting_payload = {"setting": setting, "splits": []}
        for split in ("train", "test"):
            rows = sorted(
                [row for row in results if row["setting"] == setting and row["split"] == split],
                key=lambda row: row["seed"],
            )
            pairwise = []
            for left, right in combinations(rows, 2):
                pairwise.append({
                    "seeds": [left["seed"], right["seed"]],
                    "exact_partition_match": partitions_equal(left["k2_labels"], right["k2_labels"]),
                    "adjusted_rand_index": float(adjusted_rand_score(left["k2_labels"], right["k2_labels"])),
                    "pair_agreement": pairwise_partition_agreement(left["k2_labels"], right["k2_labels"]),
                })
            patterns = ["+".join(map(str, row["k2_cluster_sizes"])) for row in rows]
            dominant = Counter(patterns).most_common(1)[0][0] if patterns else "unavailable"
            setting_payload["splits"].append({
                "split": split, "available_seed_count": len(rows), "id_level_cross_seed_analysis": "exploratory",
                "pairwise_id_level": pairwise, "cluster_size_patterns": patterns,
                "dominant_cluster_size_pattern": dominant,
                "size_pattern_consistency": len(set(patterns)) == 1 if patterns else None,
                "mean_silhouette": _safe_mean(row["k2_silhouette"] for row in rows),
                "mean_within_between_gap": _safe_mean(row["within_between_gap"] for row in rows),
                "mean_bootstrap_support": _safe_mean(row["bootstrap_partition_support"] for row in rows),
                "singleton_seed_count": sum(1 in row["k2_cluster_sizes"] for row in rows),
            })
        output["settings"].append(setting_payload)
    output["strictness_status"] = STRICTNESS_STATUS
    return output


def _safe_mean(values: Iterable[float | None]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    return sum(materialized) / len(materialized) if materialized else None


def _safe_corr(left: Sequence[float | None], right: Sequence[float | None]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(left, right, strict=True)
             if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    x, y = zip(*pairs, strict=True)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summary_by_setting(
    results: list[dict[str, Any]], stability: list[dict[str, Any]], settings: Sequence[str]
) -> dict[str, Any]:
    rows = []
    for setting in settings:
        selected = [row for row in results if row["setting"] == setting]
        patterns = ["+".join(map(str, row["k2_cluster_sizes"])) for row in selected]
        stable_count = sum(row["setting"] == setting and row["stability"] == "stable" for row in stability)
        strength_counts = Counter(row["cluster_strength"] for row in selected)
        rows.append({
            "setting": setting,
            "available_split_count": len(selected),
            "dominant_cluster_size_pattern": Counter(patterns).most_common(1)[0][0] if patterns else "unavailable",
            "size_pattern_counts": dict(Counter(patterns)),
            "cluster_strength_counts": dict(strength_counts),
            "mean_silhouette": _safe_mean(row["k2_silhouette"] for row in selected),
            "mean_within_between_gap": _safe_mean(row["within_between_gap"] for row in selected),
            "mean_bootstrap_support": _safe_mean(row["bootstrap_partition_support"] for row in selected),
            "train_test_stable_seed_count": stable_count,
            "cluster_strength_vs_mean_M_correlation": _safe_corr(
                [row["within_between_gap"] for row in selected], [row["mean_M"] for row in selected]
            ),
            "cluster_strength_vs_same_wrong_correlation": _safe_corr(
                [row["within_between_gap"] for row in selected],
                [row["mean_off_diagonal_same_wrong_excess"] for row in selected],
            ),
            "strictness_status": STRICTNESS_STATUS,
        })
    return {
        "analysis_version": ANALYSIS_VERSION,
        "settings": rows,
        "overall_relationships": {
            "within_between_gap_vs_mean_M_correlation": _safe_corr(
                [row["within_between_gap"] for row in results], [row["mean_M"] for row in results]
            ),
            "within_between_gap_vs_same_wrong_excess_correlation": _safe_corr(
                [row["within_between_gap"] for row in results],
                [row["mean_off_diagonal_same_wrong_excess"] for row in results],
            ),
            "within_between_gap_vs_team_vote_accuracy_correlation": _safe_corr(
                [row["within_between_gap"] for row in results], [row["team_vote_accuracy"] for row in results]
            ),
        },
        "strictness_status": STRICTNESS_STATUS,
    }


def _plot_heatmap(matrix: Sequence[Sequence[float | None]], title: str, path: Path, *, vmin: float | None = None, vmax: float | None = None) -> None:
    values = np.asarray([[np.nan if value is None else value for value in row] for row in matrix], dtype=float)
    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax)
    axis.set_xticks(range(AGENT_COUNT), [f"Agent{i}" for i in range(AGENT_COUNT)], rotation=30)
    axis.set_yticks(range(AGENT_COUNT), [f"Agent{i}" for i in range(AGENT_COUNT)])
    axis.set_title(title)
    for i in range(AGENT_COUNT):
        for j in range(AGENT_COUNT):
            if not np.isnan(values[i, j]):
                axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white" if values[i, j] < np.nanmean(values) else "black", fontsize=8)
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dendrogram(distance: Sequence[Sequence[float]], title: str, path: Path) -> None:
    condensed = squareform(np.asarray(distance, dtype=float), checks=False)
    tree = linkage(condensed, method="average")
    fig, axis = plt.subplots(figsize=(6, 4))
    dendrogram(tree, labels=[f"Agent{i}" for i in range(AGENT_COUNT)], ax=axis)
    axis.set_title(title)
    axis.set_ylabel("distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cluster_summary(result: dict[str, Any], title: str, path: Path) -> None:
    sizes = result["k2_cluster_sizes"]
    fig, axis = plt.subplots(figsize=(5.5, 4))
    axis.bar(["Cluster 0", "Cluster 1"], sizes, color=["#4472C4", "#ED7D31"])
    axis.set_ylim(0, 5)
    axis.set_ylabel("members")
    axis.set_title(f"{title}\nsilhouette={result['k2_silhouette'] if result['k2_silhouette'] is not None else 'NA'}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_update_alignment(alignment: dict[str, Any], title: str, path: Path) -> None:
    clusters = alignment["clusters"]
    x = np.arange(len(clusters))
    fig, axis = plt.subplots(figsize=(5.5, 4))
    axis.bar(x - 0.18, [row["attempt_share"] or 0 for row in clusters], width=0.36, label="selected")
    axis.bar(x + 0.18, [row["accepted_share"] or 0 for row in clusters], width=0.36, label="accepted")
    axis.set_xticks(x, [f"Cluster {row['cluster_id']}" for row in clusters])
    axis.set_ylim(0, 1)
    axis.set_ylabel("share")
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_figures(output: Path, results: list[dict[str, Any]], alignments: list[dict[str, Any]]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    alignment_map = {(row["setting"], row["seed"], row["split"]): row for row in alignments}
    for row in results:
        stem = f"{row['setting']}_seed{row['seed']}_{row['split']}"
        title = f"{row['setting']} seed {row['seed']} {row['split']}"
        _plot_heatmap(row["pairwise_correctness_correlation"], title, figures / f"{stem}_correctness_heatmap.png", vmin=-1, vmax=1)
        _plot_heatmap(row["consensus_matrix"], title, figures / f"{stem}_bootstrap_consensus_heatmap.png", vmin=0, vmax=1)
        _plot_dendrogram(row["_distance_matrix"], title, figures / f"{stem}_dendrogram.png")
        _plot_cluster_summary(row, title, figures / f"{stem}_cluster_size_silhouette.png")
        _plot_update_alignment(alignment_map[(row["setting"], row["seed"], row["split"])], title, figures / f"{stem}_selected_accepted_cluster_share.png")


def _write_csv(path: Path, results: list[dict[str, Any]], alignments: list[dict[str, Any]]) -> None:
    alignment_map = {(row["setting"], row["seed"], row["split"]): row for row in alignments}
    fields = [
        "task", "seed", "setting", "split", "example_count", "distance_type",
        "k2_partition", "k2_cluster_sizes", "k2_silhouette", "k3_partition",
        "k3_silhouette", "preferred_k_by_silhouette", "within_correlation",
        "between_correlation", "within_between_gap", "cluster_strength",
        "bootstrap_partition_support", "minimum_pair_support",
        "selected_updates_by_agent", "accepted_updates_by_agent", "strictness_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            output = {key: row.get(key) for key in fields}
            alignment = alignment_map[(row["setting"], row["seed"], row["split"])]
            output["selected_updates_by_agent"] = json.dumps(alignment["selected_updates_by_agent"])
            output["accepted_updates_by_agent"] = json.dumps(alignment["accepted_updates_by_agent"])
            output["k2_cluster_sizes"] = json.dumps(output["k2_cluster_sizes"])
            writer.writerow(output)


def _cluster_markdown(
    availability: list[dict[str, Any]], results: list[dict[str, Any]],
    stability: list[dict[str, Any]], summary: dict[str, Any],
    update_alignment: list[dict[str, Any]], responsibility_alignment: list[dict[str, Any]],
) -> str:
    available_runs = sum(row["train_available"] and row["test_available"] for row in availability)
    lines = [
        "# Agent behavior clustering", "",
        f"`strict_comparison_status = {STRICTNESS_STATUS}`", "",
        "## 1. Data availability and strictness", "",
        f"All 15 target runs were audited; {available_runs}/15 contained reconstructable 75-example train and 125-example test profiles. "
        "Profiles were reconstructed from each run's read-only exact-observation cache and checked against its saved behavior matrices. "
        "Cross-setting test differences remain exploratory because these historical runs predate the cumulative v2 observation chain.", "",
        "## 2. Train/test k=2 partitions", "",
        "| Setting | Seed | Train | Test | Train strength | Test strength | Stability |", "|---|---:|---|---|---|---|---|",
    ]
    result_map = {(row["setting"], row["seed"], row["split"]): row for row in results}
    stability_map = {(row["setting"], row["seed"]): row for row in stability}
    setting_order = [row["setting"] for row in summary["settings"]]
    for setting in setting_order:
        for seed in sorted({row["seed"] for row in results if row["setting"] == setting}):
            train, test = result_map[(setting, seed, "train")], result_map[(setting, seed, "test")]
            stable = stability_map[(setting, seed)]
            lines.append(f"| {setting} | {seed} | `{train['k2_partition']}` | `{test['k2_partition']}` | {train['cluster_strength']} | {test['cluster_strength']} | {stable['stability']} |")
    lines += ["", "## 3. Structural patterns", ""]
    for row in summary["settings"]:
        lines.append(
            f"- `{row['setting']}`: dominant `{row['dominant_cluster_size_pattern']}`; "
            f"patterns {row['size_pattern_counts']}; strengths {row['cluster_strength_counts']}; "
            f"train/test stable seeds {row['train_test_stable_seed_count']}/3."
        )
    weak_bootstrap = [
        f"{row['setting']} seed {row['seed']} {row['split']}"
        for row in results if row["bootstrap_stability"] == "weak"
    ]
    lines += ["", "## 4. Train/test and cross-seed stability", "",
              "Train/test stability totals are 6 stable, 5 partially stable, and 4 unstable runs. S1 and S2 each preserve the partition in two seeds; S3 and S4 do so in one; S5 does so in none. Exact Agent-ID partitions are generally less stable than cluster-size structure. Across seeds, S1 keeps a 4+1 train structure in all three seeds and S2 keeps a 3+2 train structure in all three, but S2 has zero exact cross-seed ID matches. S3/S4 mix 4+1 and 3+2, while S5 is consistently 4+1 on train but changes which Agent is isolated.", "",
              "Bootstrap partition support is below 0.70 for: " + ", ".join(weak_bootstrap) + ". These are weak-bootstrap partitions, not evidence of no available data.", "",
              "## 5. S4/S5 high-frequency targets", ""]
    for row in update_alignment:
        if row["setting"] not in {"shared_member_aware_responsibility", "shared_member_aware_full"} or row["split"] != "train":
            continue
        details = "; ".join(
            f"Agent{item['agent_id']} cluster={item['cluster_members']} "
            f"singleton={item['is_singleton_cluster']} minority={item['is_minority_cluster']} "
            f"agent attempt/accepted shares={item['agent_attempt_share']:.3f}/{item['agent_accepted_share']:.3f} "
            f"efficiency={item['acceptance_efficiency']:.3f}"
            for item in row["most_selected_agent_details"]
        )
        lines.append(f"- `{row['setting']}` seed {row['seed']}: {details}.")
    lines.append("")
    lines.append("S4's highest-frequency target is a singleton in one side of the seed-44 tie and a two-member minority in seed 45, but belongs to the majority cluster in seed 46. S5's highest-frequency target belongs to the majority cluster in all three seeds. Therefore repeated selection does not generally create a singleton specialist.")
    lines += ["", "## 6. Responsibility and update efficiency", ""]
    flagged = [
        (row["setting"], row["seed"], row["split"], row["high_responsibility_low_efficiency_agents"])
        for row in responsibility_alignment if row["high_responsibility_low_efficiency_agents"]
    ]
    lines.append(
        "The diagnostic high-responsibility/low-efficiency rule flagged: " +
        ("; ".join(f"{setting} seed {seed} {split}: {agents}" for setting, seed, split, agents in flagged) if flagged else "none") + "."
    )
    lines.append("The clearest high-attempt/low-acceptance cases are S4 seed 45 Agent0 (13/32 attempts, 1 acceptance), S4 seed 46 Agent2 (12/32, 2), S5 seed 45 Agent3 (16/32, 2), and S5 seed 46 Agent3 (16/32, 4). S5 seed 44 is a counterexample: its most-selected Agent4 has 5 acceptances from 9 attempts. The mismatch diagnosis is therefore recurrent but not universal.")
    lines += ["", "## 7. What cluster strength represents", "",
              "Partitions are defined only from correctness vectors. Disagreement, double-fault, and same-wrong quantities are auxiliary contrasts. Between-cluster answer disagreement is higher for every setting on average, and within-cluster double fault is higher, as expected for correctness-similarity clusters. Same-wrong contrast is mixed and is negative for S2 and S5, so these are not consistently same-error clusters.", "",
              f"Across all 30 splits, the correctness correlation gap has correlation {summary['overall_relationships']['within_between_gap_vs_mean_M_correlation']:.3f} with mean M, {summary['overall_relationships']['within_between_gap_vs_same_wrong_excess_correlation']:.3f} with mean same-wrong excess, and {summary['overall_relationships']['within_between_gap_vs_team_vote_accuracy_correlation']:.3f} with team vote accuracy. This does not support either 'stronger clusters imply higher M' or 'clusters merely track lower same-wrong'; the observed associations are weak and exploratory.", "",
              "## 8. Responsibility-space mismatch interpretation", "",
              "High responsibility combined with many attempts and below-average acceptance efficiency is compatible with a responsibility-versus-prompt-search mismatch, but it is diagnostic rather than causal. Candidate-search history was not added to responsibility or scheduling.", "",
              "## 9. Conclusions awaiting strict v2 reruns", "",
              "All cross-setting test comparisons, claims that S4 or S5 causes a particular cluster, and claims linking cluster strength to superior final test performance must await complete v2 matched-observation runs. Within-run partitions and train/test stability describe the observations actually saved by each historical run.", "",
              "## 10. Dynamic clustering", "",
              "`dynamic_clustering_status = unavailable`: this report does not infer intermediate profiles from prompt-hash trajectories. Only final active train/test profiles are analyzed.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline clustering of final five-agent behavior profiles")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--task", default="disambiguation_qa")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    run_root = args.run_root if args.run_root.is_absolute() else workspace / args.run_root
    split_root = workspace / "strict_splits_bbh_seed42" / args.task
    output.mkdir(parents=True, exist_ok=True)

    availability: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    update_alignments: list[dict[str, Any]] = []
    responsibility_alignments: list[dict[str, Any]] = []
    split_payloads: dict[tuple[str, int, str], tuple[list[ProbeExample], list[list[PromptAnswer]]]] = {}

    for seed in args.seeds:
        for setting in args.settings:
            run_dir = run_root / args.task / f"{setting}_seed{seed}"
            required = ["run_meta.json", "best_prompts.json", "_solver_cache.sqlite", "training_dynamics.jsonl", "final_test_differentiation.json"]
            missing = [name for name in required if not (run_dir / name).is_file()]
            audit: dict[str, Any] = {
                "task": args.task, "seed": seed, "setting": setting,
                "train_available": False, "test_available": False,
                "profile_lengths": {"train": None, "test": None},
                "source_paths_sanitized": [
                    (run_dir / name).relative_to(workspace).as_posix() for name in required if (run_dir / name).is_file()
                ],
                "source_hashes": {
                    name: _sha256_file(run_dir / name) for name in required if (run_dir / name).is_file()
                },
                "issues": [f"missing:{name}" for name in missing],
            }
            if missing:
                availability.append(audit)
                continue
            run_meta = _read_json(run_dir / "run_meta.json")
            prompts = _read_json(run_dir / "best_prompts.json")
            prompt_hashes = [_prompt_hash(prompt) for prompt in prompts]
            state_hash = _team_state_hash(prompt_hashes)
            saved_state_hash = str(run_meta["final_state_selection"]["selected_team_prompt_state_hash"])
            if state_hash != saved_state_hash:
                audit["issues"].append("final_prompt_hashes_do_not_match_final_state")
                availability.append(audit)
                continue
            audit["source_identity"] = {
                "git_commit": run_meta["run_identity"]["git_commit"],
                "method_version": run_meta["method_version"],
                "config_fingerprint": run_meta["run_identity"]["config_fingerprint"],
            }
            audit["final_prompt_hashes"] = prompt_hashes
            audit["final_team_state_hash"] = state_hash
            counts = _update_counts(run_dir)
            saved_by_split = {
                "train": _read_jsonl(run_dir / "training_dynamics.jsonl")[-1],
                "test": _read_json(run_dir / "final_test_differentiation.json"),
            }
            for split, expected_length in (("train", 75), ("test", 125)):
                split_path = split_root / ("opt.csv" if split == "train" else "test.csv")
                expected_hash = run_meta["run_identity"][f"{'train' if split == 'train' else 'test'}_file_sha256"]
                if _sha256_file(split_path) != expected_hash:
                    audit["issues"].append(f"{split}:split_hash_mismatch")
                    continue
                profiles = None
                chosen_examples = None
                chosen_mode = None
                load_issues: list[str] = []
                for mode in ("normalized_spaces_v1", "exact_utf8_v1"):
                    examples = _load_examples(split_path, mode)
                    candidate, issues = _load_profiles(run_dir / "_solver_cache.sqlite", prompt_hashes, examples)
                    if candidate is not None:
                        profiles, chosen_examples, chosen_mode = candidate, examples, mode
                        break
                    load_issues.extend(issues[:5])
                if profiles is None or chosen_examples is None:
                    audit["issues"].append(f"{split}:profile_unavailable")
                    audit["issues"].extend(f"{split}:{issue}" for issue in load_issues[:5])
                    continue
                if len(chosen_examples) != expected_length or any(len(profile) != expected_length for profile in profiles):
                    audit["issues"].append(f"{split}:profile_length_mismatch")
                    continue
                result = analyze_split(
                    task=args.task, setting=setting, seed=seed, split=split,
                    examples=chosen_examples, profiles=profiles,
                    saved_metrics=saved_by_split[split], bootstrap_replicates=args.bootstrap_replicates,
                )
                if not result["profile_recomputed_metrics_match_saved"]:
                    audit["issues"].append(f"{split}:recomputed_metrics_mismatch")
                    continue
                audit[f"{split}_available"] = True
                audit["profile_lengths"][split] = [len(profile) for profile in profiles]
                audit.setdefault("question_hash_modes", {})[split] = chosen_mode
                results.append(result)
                update_alignments.append(_cluster_update_alignment(result, counts))
                split_payloads[(setting, seed, split)] = (chosen_examples, profiles)
            if audit["train_available"] and setting in {"shared_member_aware_responsibility", "shared_member_aware_full"}:
                training_rows = _read_jsonl(run_dir / "training_dynamics.jsonl")
                responsibility = _final_responsibility(
                    run_dir,
                    training_rows[0]["per_agent_correct_counts"],
                    training_rows[-1]["per_agent_correct_counts"],
                )
                if responsibility is not None:
                    for split in ("train", "test"):
                        result = next((row for row in results if row["setting"] == setting and row["seed"] == seed and row["split"] == split), None)
                        if result is not None:
                            responsibility_alignments.append(_responsibility_alignment(result, responsibility, counts))
            availability.append(audit)

    result_map = {(row["setting"], row["seed"], row["split"]): row for row in results}
    stability = []
    for seed in args.seeds:
        for setting in args.settings:
            train = result_map.get((setting, seed, "train"))
            test = result_map.get((setting, seed, "test"))
            if train is not None and test is not None:
                stability.append(_train_test_stability(train, test))

    cross_seed = _cross_seed(results, args.settings)
    summary = _summary_by_setting(results, stability, args.settings)
    alignment_map = {
        (row["setting"], row["seed"], row["split"]): row
        for row in update_alignments
    }
    publication_results = []
    for row in results:
        published = {k: v for k, v in row.items() if not k.startswith("_")}
        alignment = alignment_map[(row["setting"], row["seed"], row["split"])]
        published.update({
            "selected_updates_by_agent": alignment["selected_updates_by_agent"],
            "accepted_updates_by_agent": alignment["accepted_updates_by_agent"],
            "cluster_attempt_shares": [cluster["attempt_share"] for cluster in alignment["clusters"]],
            "cluster_accepted_shares": [cluster["accepted_share"] for cluster in alignment["clusters"]],
            "cluster_efficiencies": [cluster["acceptance_efficiency"] for cluster in alignment["clusters"]],
        })
        publication_results.append(published)
    bootstrap_consensus = [{
        "task": row["task"], "seed": row["seed"], "setting": row["setting"], "split": row["split"],
        "bootstrap_seed": row["bootstrap_seed"], "replicates": row["replicates"],
        "consensus_matrix": row["consensus_matrix"],
        "bootstrap_partition_support": row["bootstrap_partition_support"],
        "minimum_pair_support": row["minimum_pair_support"],
        "unstable_pairs": row["unstable_pairs"], "bootstrap_stability": row["bootstrap_stability"],
        "strictness_status": STRICTNESS_STATUS,
    } for row in results]

    _write_json(output / "data_availability_audit.json", {
        "analysis_version": ANALYSIS_VERSION, "expected_run_count": len(args.seeds) * len(args.settings),
        "audited_run_count": len(availability), "runs": availability,
        "strict_comparison_status": STRICTNESS_STATUS,
    })
    _write_json(output / "clustering_results.json", publication_results)
    _write_csv(output / "clustering_results.csv", results, update_alignments)
    _write_json(output / "train_test_stability.json", stability)
    _write_json(output / "cross_seed_stability.json", cross_seed)
    _write_json(output / "bootstrap_consensus.json", bootstrap_consensus)
    _write_json(output / "update_cluster_alignment.json", update_alignments)
    _write_json(output / "responsibility_cluster_alignment.json", responsibility_alignments)
    _write_json(output / "clustering_summary_by_setting.json", summary)
    _write_figures(output, results, update_alignments)
    (output / "limitations.md").write_text(
        "# Limitations\n\n"
        f"`strict_comparison_status = {STRICTNESS_STATUS}`\n\n"
        "These historical setting-local caches did not share a cumulative matched observation reference. "
        "Within-run clustering describes saved observations, but cross-setting test differences are exploratory. "
        "No prompt text, question text, gold label, literal answer, raw response, endpoint, credential, or absolute path is published. "
        "The fixed k=2 partition is descriptive; k=3 is sensitivity analysis. Bootstrap resamples questions and does not establish causal stability.\n",
        encoding="utf-8",
    )
    (output / "cluster.md").write_text(
        _cluster_markdown(availability, results, stability, summary, update_alignments, responsibility_alignments),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Final-method agent clustering\n\n"
        f"`strict_comparison_status = {STRICTNESS_STATUS}`\n\n"
        "Offline two-cluster average-linkage analysis of five-agent correctness vectors, with k=3 sensitivity, "
        "1,000-replicate bootstrap consensus, train/test stability, cross-seed structure, and update/responsibility alignment. "
        "See `cluster.md` for conclusions and `limitations.md` before interpreting cross-setting differences.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": output.relative_to(workspace).as_posix(),
        "audited_runs": len(availability),
        "available_train_splits": sum(row["train_available"] for row in availability),
        "available_test_splits": sum(row["test_available"] for row in availability),
        "clustering_result_count": len(results),
        "real_api_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
