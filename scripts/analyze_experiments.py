import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

try:
    from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTINGS, setting_from_run_name, setting_names
    from scripts.experiment_io import read_csv_rows, write_csv, write_jsonl
except ModuleNotFoundError:
    from experiment_config import DEFAULT_EXPERIMENT_SETTINGS, setting_from_run_name, setting_names
    from experiment_io import read_csv_rows, write_csv, write_jsonl


SETTINGS = setting_names(DEFAULT_EXPERIMENT_SETTINGS)


def _setting_from_run_name(name: str) -> str:
    return setting_from_run_name(name, DEFAULT_EXPERIMENT_SETTINGS)


def _collect_run_dirs(out_root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for path in sorted(out_root.rglob("*")) if out_root.exists() else []:
        if path.is_dir() and (path / "run_meta.json").exists() and _setting_from_run_name(path.name):
            run_dirs.append(path)
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
        *[str(path) for path in run_dirs],
        "--out_csv",
        str(out_csv),
        "--out_md",
        str(out_md),
    ]
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
    rows = read_csv_rows(out_root / "experiment_runs.csv")
    if not rows:
        return
    rows = [row for row in rows if str(row.get("setting", "")) in SETTINGS]
    order = {name: i for i, name in enumerate(SETTINGS)}
    rows.sort(key=lambda row: (str(row.get("dataset", "")), order.get(str(row.get("setting", "")), 999), str(row.get("seed", ""))))
    write_csv(out_root / "experiment_summary.csv", rows, empty_fieldnames=["dataset", "setting", "status"])
    write_jsonl(out_root / "experiment_summary.jsonl", rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot the default experiment settings from existing run directories.")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out_root", type=str, default="runs_trace_beam")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    args.workspace = str(workspace)
    out_root = (workspace / args.out_root).resolve()
    out_dir = (workspace / args.out_dir).resolve() if args.out_dir else out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _collect_run_dirs(out_root)
    found = [str(path.relative_to(out_root)) for path in run_dirs]
    found_settings = {_setting_from_run_name(path.name) for path in run_dirs}
    missing = [name for name in SETTINGS if name not in found_settings]
    print(f"Expected settings: {SETTINGS}")
    print(f"Found runs: {found}")
    if missing:
        print(f"[WARN] Missing expected settings: {missing}")

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
