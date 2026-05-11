import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


SETTINGS = [
    "shared_div",
    "bank_div",
    "shared_baseline",
    "bank_baseline",
]


def _setting_from_run_name(name: str) -> str:
    for setting in SETTINGS:
        if name == setting or name.startswith(f"{setting}_seed"):
            return setting
    return ""


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("setting,status\n")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _collect_run_dirs(out_root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for p in sorted(out_root.iterdir()) if out_root.exists() else []:
        if not p.is_dir() or not (p / "run_meta.json").exists():
            continue
        if _setting_from_run_name(p.name):
            run_dirs.append(p)
    return run_dirs


def _call_analyzer(args: argparse.Namespace, run_dirs: List[Path], out_csv: Path, out_md: Path) -> bool:
    analyzer = Path(args.workspace) / "scripts" / "compute_experiment_metrics.py"
    if not analyzer.exists():
        print("[WARN] scripts/compute_experiment_metrics.py not found, skip analyzer step")
        return False
    if not run_dirs:
        print("[WARN] no valid run directories found, skip analyzer step")
        return False

    cmd = [
        args.python,
        str(analyzer),
        "--runs",
        *[str(p) for p in run_dirs],
        "--out_csv",
        str(out_csv),
        "--out_md",
        str(out_md),
        "--summary_embedding_model",
        args.summary_embedding_model,
    ]
    if args.disable_summary_embedding:
        cmd.append("--disable_summary_embedding")
    print("=" * 120)
    print("[ANALYZE]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=args.workspace, check=False)
    return proc.returncode == 0


def _call_plotter(args: argparse.Namespace, csv_path: Path, out_dir: Path) -> bool:
    plotter = Path(args.workspace) / "scripts" / "plot_experiment_results.py"
    if not plotter.exists():
        print("[WARN] scripts/plot_experiment_results.py not found, skip plotting")
        return False
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found, skip plotting")
        return False

    cmd = [
        args.python,
        str(plotter),
        "--csv",
        str(csv_path),
        "--out_dir",
        str(out_dir),
    ]
    print("=" * 120)
    print("[PLOT]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=args.workspace, check=False)
    return proc.returncode == 0


def _write_filtered_run_summary(out_root: Path):
    source = out_root / "experiment_runs.csv"
    rows = _read_csv_rows(source)
    if not rows:
        return
    rows = [r for r in rows if str(r.get("setting", "")) in SETTINGS]
    order = {name: i for i, name in enumerate(SETTINGS)}
    rows.sort(key=lambda r: order.get(str(r.get("setting", "")), 999))
    _write_csv(out_root / "experiment_summary.csv", rows)
    _write_jsonl(out_root / "experiment_summary.jsonl", rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot the four experiment settings from existing run directories."
    )
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out_root", type=str, default="runs_experiments")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--summary_embedding_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--disable_summary_embedding", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    args.workspace = str(workspace)
    out_root = (workspace / args.out_root).resolve()
    out_dir = (workspace / args.out_dir).resolve() if args.out_dir else out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _collect_run_dirs(out_root)
    found = [p.name for p in run_dirs]
    found_settings = {_setting_from_run_name(p.name) for p in run_dirs}
    missing = [name for name in SETTINGS if name not in found_settings]
    print(f"Found runs: {found}")
    if missing:
        print(f"[WARN] Missing expected runs: {missing}")

    _write_filtered_run_summary(out_root)
    out_csv = out_root / "experiment_analysis.csv"
    out_md = out_root / "experiment_analysis.md"
    analyzed = _call_analyzer(args, run_dirs, out_csv, out_md)
    plotted = _call_plotter(args, out_csv, out_dir) if analyzed else False

    print("=" * 120)
    print(f"Analyze status: {'ok' if analyzed else 'skipped_or_failed'}")
    print(f"Plot status   : {'ok' if plotted else 'skipped_or_failed'}")
    print(f"CSV           : {out_csv}")
    print(f"Markdown      : {out_md}")
    print(f"Figure dir    : {out_dir}")


if __name__ == "__main__":
    main()

