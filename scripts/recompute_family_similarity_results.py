import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.utils import compute_strategy_family_profile_metrics


SETTINGS = ["shared_div", "bank_div", "shared_baseline", "bank_baseline"]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_config(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    meta = read_json(meta_path)
    cfg = meta.get("config", {}) if isinstance(meta, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def recompute_record(record: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    primary = record.get("primary_family_labels", [])
    secondary = record.get("secondary_family_labels", [])
    if not isinstance(primary, list) or not primary:
        return False
    if not isinstance(secondary, list):
        secondary = None

    metrics = compute_strategy_family_profile_metrics(
        [str(x) for x in primary],
        [str(x) for x in secondary] if isinstance(secondary, list) else None,
        use_dual_family=bool(cfg.get("use_dual_family_labels", True)),
        primary_weight=float(cfg.get("primary_family_weight", 0.7) or 0.7),
        secondary_weight=float(cfg.get("secondary_family_weight", 0.3) or 0.3),
        same_major_weight=float(cfg.get("same_major_family_weight", 0.5) or 0.5),
        macro_diversity_weight=float(cfg.get("macro_diversity_weight", 0.5) or 0.5),
        allow_fallback=True,
    )

    # Keep label/count fields aligned with the recomputed profile as well, since
    # normalization may affect labels and pairwise rho fields depend on the kernel.
    for key in [
        "primary_families",
        "secondary_families",
        "agent_family_distributions",
        "primary_family_counts",
        "weighted_family_distribution",
        "major_family_distribution",
        "per_agent_same_family_count",
        "per_agent_same_family_ratio",
        "per_agent_family_diversity",
        "team_family_homogeneity_rate",
        "team_family_diversity",
        "team_family_entropy",
        "team_major_family_diversity",
        "team_intra_family_diversity",
        "dominant_family_share",
        "dominant_major_family_share",
    ]:
        if key in metrics:
            record[key] = metrics[key]

    record["primary_family_labels"] = metrics.get("primary_families", record.get("primary_family_labels", []))
    record["secondary_family_labels"] = metrics.get("secondary_families", record.get("secondary_family_labels", []))

    normalized_pairs = list(zip(record.get("primary_family_labels", []), record.get("secondary_family_labels", [])))
    pair_counts: Dict[str, int] = {}
    for p, s in normalized_pairs:
        key = f"{p}|{s}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    n = len(normalized_pairs)
    record["all_same_primary"] = bool(len(set(record.get("primary_family_labels", []))) == 1 and n > 0)
    record["all_same_pair"] = bool(len(set(normalized_pairs)) == 1 and n > 0)
    record["primary_dominant_share"] = (
        float(max(record.get("primary_family_counts", {}).values()) / n)
        if isinstance(record.get("primary_family_counts"), dict) and record.get("primary_family_counts") and n
        else 0.0
    )
    record["pair_dominant_share"] = float(max(pair_counts.values()) / n) if pair_counts and n else 0.0
    return True


def recompute_jsonl(path: Path, cfg: Dict[str, Any]) -> int:
    records = read_jsonl(path)
    changed = 0
    for rec in records:
        if recompute_record(rec, cfg):
            changed += 1
    if changed:
        write_jsonl(path, records)
    return changed


def recompute_history(run_dir: Path) -> bool:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return False
    history = read_json(history_path)
    if not isinstance(history, list):
        return False

    train_records = read_jsonl(run_dir / "train_step_logs.jsonl")
    train_by_epoch: Dict[int, List[Dict[str, Any]]] = {}
    for rec in train_records:
        epoch = int(rec.get("epoch", 0) or 0)
        train_by_epoch.setdefault(epoch, []).append(rec)

    test_by_epoch: Dict[int, List[Dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("test_epoch*_predictions.jsonl")):
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        epoch = int(digits or 0)
        test_by_epoch[epoch] = read_jsonl(path)

    updated = False
    for entry in history:
        if not isinstance(entry, dict):
            continue
        epoch = int(entry.get("epoch", 0) or 0)
        train = entry.get("train", {})
        if isinstance(train, dict) and train_by_epoch.get(epoch):
            vals = [float(r.get("team_family_homogeneity_rate", 0.0) or 0.0) for r in train_by_epoch[epoch]]
            if vals:
                train["mean_family_homogeneity_rate"] = sum(vals) / len(vals)
                updated = True
        test = entry.get("test", {})
        if isinstance(test, dict) and test_by_epoch.get(epoch):
            vals = [float(r.get("team_family_homogeneity_rate", 0.0) or 0.0) for r in test_by_epoch[epoch]]
            if vals:
                test["mean_family_homogeneity_rate"] = sum(vals) / len(vals)
                updated = True
    if updated:
        write_json(history_path, history)
    return updated


def write_run_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    extra = sorted({k for row in rows for k in row.keys() if k not in fields})
    fields.extend(extra)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_run_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def update_run_summary_csv(csv_path: Path, by_setting: Dict[str, Dict[str, float]]) -> int:
    rows = read_run_csv(csv_path)
    changed = 0
    for row in rows:
        setting = str(row.get("setting", ""))
        vals = by_setting.get(setting)
        if not vals:
            continue
        for key, val in vals.items():
            row[key] = str(val)
        changed += 1
    if changed:
        write_run_csv(csv_path, rows)
    return changed


def update_jsonl_summary(path: Path, by_setting: Dict[str, Dict[str, float]]) -> int:
    records = read_jsonl(path)
    changed = 0
    for rec in records:
        vals = by_setting.get(str(rec.get("setting", "")))
        if not vals:
            continue
        rec.update(vals)
        changed += 1
    if changed:
        write_jsonl(path, records)
    return changed


def latest_history_values(run_dir: Path) -> Dict[str, float]:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return {}
    history = read_json(history_path)
    if not isinstance(history, list) or not history:
        return {}
    latest = history[-1]
    if not isinstance(latest, dict):
        return {}
    out: Dict[str, float] = {}
    train = latest.get("train", {})
    test = latest.get("test", {})
    if isinstance(train, dict) and "mean_family_homogeneity_rate" in train:
        out["latest_train_mean_family_homogeneity_rate"] = float(train["mean_family_homogeneity_rate"])
    if isinstance(test, dict) and "mean_family_homogeneity_rate" in test:
        out["latest_test_mean_family_homogeneity_rate"] = float(test["mean_family_homogeneity_rate"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute stored family similarity metrics in existing run outputs.")
    parser.add_argument("--runs_root", type=str, default="runs_experiments")
    args = parser.parse_args()

    root = Path(args.runs_root)
    by_setting: Dict[str, Dict[str, float]] = {}
    total_records = 0
    for setting in SETTINGS:
        run_dir = root / setting
        if not run_dir.exists():
            continue
        cfg = run_config(run_dir)
        changed = 0
        for name in ["train_step_logs.jsonl"]:
            changed += recompute_jsonl(run_dir / name, cfg)
        for path in sorted(run_dir.glob("test_epoch*_predictions.jsonl")):
            changed += recompute_jsonl(path, cfg)
        recompute_history(run_dir)
        by_setting[setting] = latest_history_values(run_dir)
        total_records += changed
        print(f"{setting}: recomputed_records={changed}, latest={by_setting[setting]}")

    for name in ["experiment_runs.csv", "experiment_summary.csv"]:
        changed = update_run_summary_csv(root / name, by_setting)
        if changed:
            print(f"{name}: updated_rows={changed}")
    for name in ["experiment_runs.jsonl", "experiment_summary.jsonl"]:
        changed = update_jsonl_summary(root / name, by_setting)
        if changed:
            print(f"{name}: updated_rows={changed}")
    print(f"total_recomputed_records={total_records}")


if __name__ == "__main__":
    main()
