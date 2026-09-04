from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "strict_splits_bbh_seed42" / "disambiguation_qa"
RUN_ROOT = ROOT / "runs" / "diversity_matrix_d0_d5_20260903"
FORMER_TEST_ROOT = ROOT / "runs" / "diversity_matrix_d0_d5_former_test125_20260904_retry1"
COMBINED_REPORT = ROOT / "reports" / "diversity_matrix_d0_d5_combined_validation175_20260904"
DEFAULT_OUT = ROOT / "reports" / "diversity_matrix_split_balance_audit_20260904"
SEEDS = (72, 73, 74)
ARMS = ("D0", "D1", "D2", "D3", "D4", "D5")
SPLITS = ("train75", "validation50", "former_test125")
FILES = {"train75": "opt.csv", "validation50": "val.csv", "former_test125": "test.csv"}
PRONOUNS = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs", "it", "its"}
NEGATIONS = {"not", "no", "never", "neither", "nor", "without", "cannot", "can't", "didn't", "wasn't", "weren't"}
CONJUNCTIONS = {"and", "but", "or", "while", "although", "because", "whereas", "after", "before", "when"}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LABEL_RE = re.compile(r"\(?\b([ABC])\b\)?", re.IGNORECASE)
OPTION_RE = re.compile(r"^\s*\(([ABC])\)\s*(.+?)\s*$", re.MULTILINE)
AUDIT_VERSION = "diversity_matrix_split_balance_audit_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qhash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def label(value: str) -> str:
    match = LABEL_RE.search(value.strip())
    return match.group(1).upper() if match else "INVALID"


def core_sentence(question: str) -> str:
    value = question
    if "Sentence:" in value:
        value = value.split("Sentence:", 1)[1]
    if "Options:" in value:
        value = value.split("Options:", 1)[0]
    return " ".join(value.split())


def text_features(question: str) -> dict[str, float]:
    core = core_sentence(question)
    words = [word.lower() for word in WORD_RE.findall(core)]
    options = [text for _, text in OPTION_RE.findall(question)]
    option_words = [len(WORD_RE.findall(text)) for text in options]
    entity_like = sum(
        token[0].isupper()
        for token in WORD_RE.findall(core)[1:]
        if token
    )
    return {
        "question_chars": float(len(question)),
        "core_chars": float(len(core)),
        "core_words": float(len(words)),
        "entity_like_tokens": float(entity_like),
        "pronoun_count": float(sum(word in PRONOUNS for word in words)),
        "negation_count": float(sum(word in NEGATIONS for word in words)),
        "conjunction_count": float(sum(word in CONJUNCTIONS for word in words)),
        "option_count": float(len(options)),
        "mean_option_words": float(statistics.mean(option_words)) if option_words else 0.0,
        "max_option_words": float(max(option_words)) if option_words else 0.0,
    }


def load_items() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for split, filename in FILES.items():
        with (DATA_ROOT / filename).open(newline="", encoding="utf-8-sig") as handle:
            source = list(csv.DictReader(handle))
        rows = []
        for row in source:
            digest = qhash(row["question"])
            if digest in seen:
                raise ValueError("split question sets overlap")
            seen.add(digest)
            rows.append({
                "question_hash": digest,
                "question": row["question"],
                "gold_label": label(row["answer"]),
                "features": text_features(row["question"]),
                "core": core_sentence(row["question"]),
            })
        result[split] = rows
    expected = {"train75": 75, "validation50": 50, "former_test125": 125}
    if {key: len(value) for key, value in result.items()} != expected:
        raise ValueError("unexpected split inventory")
    return result


def plurality_outcome(answers: Sequence[str], gold: str) -> tuple[bool, bool]:
    normalized = [label(answer) for answer in answers]
    valid = [answer for answer in normalized if answer != "INVALID"]
    counts = Counter(valid)
    if not counts:
        return False, False
    top = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == top]
    vote_correct = len(winners) == 1 and winners[0] == gold
    oracle_correct = gold in valid
    return vote_correct, oracle_correct


def train_outcomes(items: Sequence[Mapping[str, Any]], seed: int, arm: str) -> dict[str, dict[str, bool]]:
    checkpoint = read_json(RUN_ROOT / f"seed{seed}" / arm / "training_checkpoint.json")
    profiles = checkpoint["active_profiles"]
    if len(profiles) != 5 or any(len(profile) != len(items) for profile in profiles):
        raise ValueError(f"training profile inventory mismatch: {seed}:{arm}")
    result = {}
    for index, item in enumerate(items):
        answers = [str(profile[index].get("answer", "")) for profile in profiles]
        vote, oracle = plurality_outcome(answers, str(item["gold_label"]))
        result[str(item["question_hash"])] = {"vote_correct": vote, "oracle_correct": oracle}
    return result


def evaluation_outcomes(split: str, seed: int, arm: str) -> dict[str, dict[str, bool]]:
    if split == "validation50":
        path = RUN_ROOT / "validation" / f"seed{seed}" / arm / "validation_rows_sanitized.jsonl"
    elif split == "former_test125":
        path = FORMER_TEST_ROOT / f"seed{seed}" / arm / "former_test_rows_sanitized.jsonl"
    else:
        raise ValueError(split)
    rows = read_jsonl(path)
    return {
        str(row["example_id_hash"]): {
            "vote_correct": bool(row["vote_correct"]),
            "oracle_correct": int(row["G"]) > 0,
        }
        for row in rows
    }


def all_outcomes(items: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[int, dict[str, dict[str, dict[str, bool]]]]]:
    result: dict[str, dict[int, dict[str, dict[str, dict[str, bool]]]]] = {}
    for split in SPLITS:
        result[split] = {}
        expected = {str(row["question_hash"]) for row in items[split]}
        for seed in SEEDS:
            result[split][seed] = {}
            for arm in ARMS:
                outcomes = train_outcomes(items[split], seed, arm) if split == "train75" else evaluation_outcomes(split, seed, arm)
                if set(outcomes) != expected:
                    raise ValueError(f"outcome question set mismatch: {split}:{seed}:{arm}")
                result[split][seed][arm] = outcomes
    return result


def fit_lexical_clusters(items: Mapping[str, Sequence[Mapping[str, Any]]], count: int = 8) -> dict[str, int]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    ordered = [row for split in SPLITS for row in items[split]]
    matrix = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=600,
        lowercase=True, sublinear_tf=True,
    ).fit_transform(str(row["core"]) for row in ordered)
    fitted = KMeans(n_clusters=count, random_state=42, n_init=20).fit_predict(matrix)
    raw_members: dict[int, list[str]] = {}
    for row, cluster in zip(ordered, fitted):
        raw_members.setdefault(int(cluster), []).append(str(row["question_hash"]))
    stable_order = sorted(raw_members, key=lambda key: hashlib.sha256("".join(sorted(raw_members[key])).encode()).hexdigest())
    relabel = {raw: index for index, raw in enumerate(stable_order)}
    return {str(row["question_hash"]): relabel[int(cluster)] for row, cluster in zip(ordered, fitted)}


def proportions(counts: Mapping[str, int], keys: Iterable[str]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: counts.get(key, 0) / total for key in keys}


def total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def js_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    middle = {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2 for key in keys}
    def kl(source: Mapping[str, float]) -> float:
        return sum(value * math.log2(value / middle[key]) for key, value in source.items() if value > 0 and middle[key] > 0)
    return 0.5 * (kl(left) + kl(right))


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def standardized_difference(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean, left_sd = mean_std(left)
    right_mean, right_sd = mean_std(right)
    pooled = math.sqrt((left_sd ** 2 + right_sd ** 2) / 2)
    return 0.0 if pooled == 0 else (left_mean - right_mean) / pooled


def split_distributions(items: Mapping[str, Sequence[Mapping[str, Any]]], clusters: Mapping[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    label_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        labels = Counter(str(row["gold_label"]) for row in items[split])
        for value in ("A", "B", "C"):
            label_rows.append({"split": split, "label": value, "count": labels[value], "proportion": labels[value] / len(items[split])})
        cluster_counts = Counter(str(clusters[str(row["question_hash"])]) for row in items[split])
        for value in map(str, range(8)):
            cluster_rows.append({"split": split, "cluster": f"cluster_{value}", "count": cluster_counts[value], "proportion": cluster_counts[value] / len(items[split])})
        for feature in items[split][0]["features"]:
            values = [float(row["features"][feature]) for row in items[split]]
            feature_rows.append({
                "split": split, "feature": feature, "mean": statistics.mean(values),
                "median": statistics.median(values), "std": statistics.stdev(values),
                "minimum": min(values), "maximum": max(values),
            })
    return label_rows, cluster_rows, feature_rows


def pairwise_balance(items: Mapping[str, Sequence[Mapping[str, Any]]], clusters: Mapping[str, int]) -> list[dict[str, Any]]:
    rows = []
    for left, right in (("train75", "validation50"), ("train75", "former_test125"), ("validation50", "former_test125")):
        left_labels = proportions(Counter(str(row["gold_label"]) for row in items[left]), ("A", "B", "C"))
        right_labels = proportions(Counter(str(row["gold_label"]) for row in items[right]), ("A", "B", "C"))
        left_clusters = proportions(Counter(str(clusters[str(row["question_hash"])]) for row in items[left]), map(str, range(8)))
        right_clusters = proportions(Counter(str(clusters[str(row["question_hash"])]) for row in items[right]), map(str, range(8)))
        smds = {}
        for feature in items[left][0]["features"]:
            smds[feature] = standardized_difference(
                [float(row["features"][feature]) for row in items[left]],
                [float(row["features"][feature]) for row in items[right]],
            )
        max_feature = max(smds, key=lambda key: abs(smds[key]))
        rows.append({
            "comparison": f"{left}_vs_{right}",
            "label_total_variation": total_variation(left_labels, right_labels),
            "label_js_divergence_bits": js_divergence(left_labels, right_labels),
            "cluster_total_variation": total_variation(left_clusters, right_clusters),
            "cluster_js_divergence_bits": js_divergence(left_clusters, right_clusters),
            "max_abs_structural_smd": abs(smds[max_feature]),
            "max_smd_feature": max_feature,
            "signed_smd": smds[max_feature],
        })
    return rows


def static_difficulty(items: Mapping[str, Sequence[Mapping[str, Any]]], outcomes: Mapping[str, Any], clusters: Mapping[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    cluster_rows = []
    for split in SPLITS:
        by_hash = {str(row["question_hash"]): row for row in items[split]}
        groups: dict[str, list[str]] = {"ALL": list(by_hash)}
        for value in ("A", "B", "C"):
            groups[f"label_{value}"] = [digest for digest, row in by_hash.items() if row["gold_label"] == value]
        for group, hashes in groups.items():
            values = [int(outcomes[split][seed]["D0"][digest]["vote_correct"]) for seed in SEEDS for digest in hashes]
            rows.append({"split": split, "stratum": group, "item_count": len(hashes), "seed_item_count": len(values), "static_vote_accuracy": statistics.mean(values)})
        correct_seed_counts = Counter()
        for digest in by_hash:
            count = sum(int(outcomes[split][seed]["D0"][digest]["vote_correct"]) for seed in SEEDS)
            correct_seed_counts[count] += 1
        for count in range(4):
            rows.append({"split": split, "stratum": f"correct_in_{count}_of_3_seeds", "item_count": correct_seed_counts[count], "seed_item_count": "", "static_vote_accuracy": count / 3})
        for cluster in range(8):
            hashes = [digest for digest in by_hash if clusters[digest] == cluster]
            values = [int(outcomes[split][seed]["D0"][digest]["vote_correct"]) for seed in SEEDS for digest in hashes]
            cluster_rows.append({
                "split": split, "cluster": f"cluster_{cluster}", "item_count": len(hashes),
                "static_vote_accuracy": statistics.mean(values) if values else "",
            })
    return rows, cluster_rows


def arm_performance(items: Mapping[str, Sequence[Mapping[str, Any]]], outcomes: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        size = len(items[split])
        means: dict[str, float] = {}
        arm_values: dict[str, tuple[list[float], list[float]]] = {}
        for arm in ARMS:
            votes = []
            oracles = []
            for seed in SEEDS:
                values = outcomes[split][seed][arm].values()
                votes.append(sum(int(row["vote_correct"]) for row in values) / size)
                oracles.append(sum(int(row["oracle_correct"]) for row in values) / size)
            means[arm] = statistics.mean(votes)
            arm_values[arm] = votes, oracles
        ranking = {arm: index + 1 for index, arm in enumerate(sorted(ARMS, key=lambda arm: (-means[arm], arm)))}
        for arm in ARMS:
            votes, oracles = arm_values[arm]
            rows.append({
                "split": split, "arm": arm, "seed72_vote": votes[0], "seed73_vote": votes[1], "seed74_vote": votes[2],
                "mean_vote": statistics.mean(votes), "mean_oracle": statistics.mean(oracles), "vote_rank": ranking[arm],
            })
    return rows


def arm_label_performance(items: Mapping[str, Sequence[Mapping[str, Any]]], outcomes: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        by_label = {
            value: [str(row["question_hash"]) for row in items[split] if row["gold_label"] == value]
            for value in ("A", "B", "C")
        }
        for arm in ARMS:
            for value, hashes in by_label.items():
                votes = [
                    int(outcomes[split][seed][arm][digest]["vote_correct"])
                    for seed in SEEDS for digest in hashes
                ]
                rows.append({
                    "split": split,
                    "arm": arm,
                    "label": value,
                    "item_count": len(hashes),
                    "seed_item_count": len(votes),
                    "mean_vote_accuracy": statistics.mean(votes),
                })
    return rows


def label_standardization(
    items: Mapping[str, Sequence[Mapping[str, Any]]],
    label_performance: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    index = {
        (str(row["split"]), str(row["arm"]), str(row["label"])): float(row["mean_vote_accuracy"])
        for row in label_performance
    }
    mix = {
        split: proportions(
            Counter(str(row["gold_label"]) for row in items[split]),
            ("A", "B", "C"),
        )
        for split in ("validation50", "former_test125")
    }
    rows = []
    for arm in ARMS:
        val = sum(mix["validation50"][value] * index[("validation50", arm, value)] for value in ("A", "B", "C"))
        former = sum(mix["former_test125"][value] * index[("former_test125", arm, value)] for value in ("A", "B", "C"))
        val_rates_on_former_mix = sum(
            mix["former_test125"][value] * index[("validation50", arm, value)]
            for value in ("A", "B", "C")
        )
        composition = val_rates_on_former_mix - val
        within = former - val_rates_on_former_mix
        if not math.isclose(composition + within, former - val, abs_tol=1e-12):
            raise AssertionError("label standardization does not telescope")
        rows.append({
            "arm": arm,
            "validation50_vote": val,
            "former_test125_vote": former,
            "raw_gap": former - val,
            "validation_rates_on_former_label_mix": val_rates_on_former_mix,
            "label_composition_component": composition,
            "within_label_residual_component": within,
        })
    return rows


def method_split_interactions(performance: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["split"], row["arm"]): row for row in performance}
    d0_gap = float(index[("former_test125", "D0")]["mean_vote"]) - float(index[("validation50", "D0")]["mean_vote"])
    rows = []
    for arm in ARMS:
        train = float(index[("train75", arm)]["mean_vote"])
        val = float(index[("validation50", arm)]["mean_vote"])
        former = float(index[("former_test125", arm)]["mean_vote"])
        combined = (50 * val + 125 * former) / 175
        raw_gap = former - val
        rows.append({
            "arm": arm, "train_vote": train, "validation50_vote": val, "former_test125_vote": former,
            "combined175_vote": combined, "former_minus_validation": raw_gap,
            "difference_in_difference_vs_D0": raw_gap - d0_gap,
            "train_minus_combined175": train - combined,
        })
    d0_train_gap = next(float(row["train_minus_combined175"]) for row in rows if row["arm"] == "D0")
    for row in rows:
        row["train_optimism_difference_vs_D0"] = float(row["train_minus_combined175"]) - d0_train_gap
    return rows


def classify(balance: Sequence[Mapping[str, Any]], static: Sequence[Mapping[str, Any]], interactions: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    val_test = next(row for row in balance if row["comparison"] == "validation50_vs_former_test125")
    static_all = {(row["split"], row["stratum"]): float(row["static_vote_accuracy"]) for row in static if row["stratum"] == "ALL"}
    static_gap = static_all[("former_test125", "ALL")] - static_all[("validation50", "ALL")]
    max_did = max(abs(float(row["difference_in_difference_vs_D0"])) for row in interactions if row["arm"] != "D0")
    mean_train_optimism = statistics.mean(float(row["train_optimism_difference_vs_D0"]) for row in interactions if row["arm"] != "D0")
    flags = {
        "label_imbalance": float(val_test["label_total_variation"]) >= 0.15,
        "lexical_cluster_imbalance": float(val_test["cluster_total_variation"]) >= 0.20,
        "structural_feature_imbalance": float(val_test["max_abs_structural_smd"]) >= 0.35,
        "static_difficulty_gap": abs(static_gap) >= 0.06,
        "method_by_split_interaction": max_did >= 0.06,
        "train_optimism": mean_train_optimism >= 0.08,
    }
    imbalance = any(flags[key] for key in ("label_imbalance", "lexical_cluster_imbalance", "structural_feature_imbalance", "static_difficulty_gap"))
    if imbalance and flags["method_by_split_interaction"]:
        verdict = "SPLIT_IMBALANCE_AND_METHOD_INTERACTION_SUPPORTED"
    elif imbalance:
        verdict = "SPLIT_IMBALANCE_SUPPORTED"
    elif flags["train_optimism"]:
        verdict = "SPLITS_STRUCTURALLY_SIMILAR__TRANSFER_GAP_DOMINANT"
    else:
        verdict = "INCONCLUSIVE"
    diagnostics = {
        "thresholds": {"label_tv": 0.15, "cluster_tv": 0.20, "structural_abs_smd": 0.35, "static_gap": 0.06, "method_did": 0.06, "train_optimism": 0.08},
        "flags": flags, "validation50_vs_former_test125_static_gap": static_gap,
        "max_abs_method_difference_in_difference": max_did, "mean_evolved_arm_train_optimism_vs_D0": mean_train_optimism,
    }
    return verdict, diagnostics


def run(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("split-balance report root must be fresh")
    items = load_items()
    clusters = fit_lexical_clusters(items)
    outcomes = all_outcomes(items)
    labels, cluster_distribution, features = split_distributions(items, clusters)
    balance = pairwise_balance(items, clusters)
    static, cluster_static = static_difficulty(items, outcomes, clusters)
    performance = arm_performance(items, outcomes)
    label_performance = arm_label_performance(items, outcomes)
    label_decomposition = label_standardization(items, label_performance)
    interactions = method_split_interactions(performance)
    verdict, diagnostics = classify(balance, static, interactions)
    out.mkdir(parents=True)
    write_csv(out / "label_distribution.csv", labels)
    write_csv(out / "lexical_cluster_distribution.csv", cluster_distribution)
    write_csv(out / "structural_features.csv", features)
    write_csv(out / "pairwise_split_balance.csv", balance)
    write_csv(out / "static_difficulty.csv", static)
    write_csv(out / "cluster_static_difficulty.csv", cluster_static)
    write_csv(out / "arm_split_performance.csv", performance)
    write_csv(out / "arm_label_performance.csv", label_performance)
    write_csv(out / "label_standardization.csv", label_decomposition)
    write_csv(out / "method_split_interaction.csv", interactions)
    summary = {
        "audit_version": AUDIT_VERSION,
        "status": "PASS",
        "verdict": verdict,
        "api_calls": 0,
        "model_calls": 0,
        "historical_artifacts_modified": False,
        "split_sizes": {split: len(items[split]) for split in SPLITS},
        "question_sets_disjoint": True,
        "cluster_method": "TF-IDF word 1-2 grams; KMeans k=8 random_state=42 n_init=20; all 250 questions; no outcomes",
        "outcome_use": "descriptive only after structural features and clusters",
        "diagnostics": diagnostics,
        "label_standardization": {
            row["arm"]: {
                "raw_gap": row["raw_gap"],
                "label_composition_component": row["label_composition_component"],
                "within_label_residual_component": row["within_label_residual_component"],
            }
            for row in label_decomposition
        },
        "dataset_sha256": {split: sha256_file(DATA_ROOT / FILES[split]) for split in SPLITS},
    }
    write_json(out / "summary.json", summary)
    methodology = {
        "audit_version": AUDIT_VERSION,
        "predeclared_interpretation_thresholds": diagnostics["thresholds"],
        "difficulty_source": "D0 frozen predictions only; D1-D5 never define difficulty",
        "cluster_inputs": "question core sentence only; no labels, model outcomes, or split labels used in fitting",
        "privacy": "only aggregate distributions, metrics, and source hashes are published",
        "limitations": [
            "retrospective descriptive audit; does not causally assign all transfer failure to split imbalance",
            "k=8 lexical clusters are an operational sensitivity probe, not semantic ground truth",
            "Combined175 is development evidence and no untouched held-out test remains",
        ],
    }
    write_json(out / "methodology.json", methodology)
    aggregate = {(row["split"], row["arm"]): row for row in performance}
    lines = [
        "# Train75 / Validation50 / FormerTest125 Split Balance Audit",
        "",
        "This is a zero-API retrospective audit. It does not modify or rerun D0-D5.",
        "FormerTest125 is already converted to development validation; no untouched held-out test remains.",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        "## Arm ranking by split",
        "",
        "| Arm | Train75 Vote | Validation50 Vote | FormerTest125 Vote |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {float(aggregate[('train75', arm)]['mean_vote']):.4f} | "
            f"{float(aggregate[('validation50', arm)]['mean_vote']):.4f} | "
            f"{float(aggregate[('former_test125', arm)]['mean_vote']):.4f} |"
        )
    lines += [
        "", "## Interpretation", "",
        "The audit separates split composition from method transfer. Label, lexical-cluster,",
        "structural, and D0 difficulty effects are reported independently. Method-by-split",
        "difference-in-differences subtract the D0 split gap, so a large remaining value is",
        "not explained by overall split difficulty alone.",
        "", "The result is development evidence. It must not be presented as a fresh test confirmation.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
