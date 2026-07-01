#!/usr/bin/env python
"""One-click runner for the official P3 prove suite.

This script runs the four-model P3 matrix, repairs degraded traces, re-analyses
the repaired runs, and then launches the two GPT-5.5 validation passes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 120)
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official P3 prove suite end to end.")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--test_path", type=str, default="mmlu_test_200.jsonl")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_prefix", type=str, default="P3")
    parser.add_argument("--raw_runs_root", type=str, default="prove_experiments/p3_runs")
    parser.add_argument("--analysis_runs_root", type=str, default="prove_experiments/p3_analysis_runs")
    parser.add_argument("--summary_csv", type=str, default="prove_experiments/p3_runs/p3_cross_llm_runs.csv")
    parser.add_argument("--skip_p3_runs", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_repair", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_analysis", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_gpt55", type=int, default=0, choices=[0, 1])
    parser.add_argument("--sample_size", type=int, default=776)
    parser.add_argument("--eval_parallelism", type=int, default=8)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    python = args.python
    raw_runs_root = str(Path(args.raw_runs_root))
    analysis_runs_root = str(Path(args.analysis_runs_root))

    if not int(args.skip_p3_runs):
        run(
            [
                python,
                "scripts/run_p4_cross_llm_matrix.py",
                "--workspace",
                str(workspace),
                "--python",
                python,
                "--models_json",
                "prove_experiments/p4_low_cost_models.json",
                "--out_root",
                raw_runs_root,
                "--summary_name",
                str(Path(args.summary_csv).name),
                "--run_prefix",
                str(args.run_prefix),
                "--conditions",
                "same,mixed",
                "--task_type",
                "mmlu",
                "--test_path",
                str(args.test_path),
                "--test_size",
                str(args.test_size),
                "--family_rejudge_on_low_confidence",
                "0",
                "--seed",
                str(args.seed),
                "--eval_parallelism",
                str(args.eval_parallelism),
            ],
            workspace,
        )

    if not int(args.skip_repair):
        run(
            [
                python,
                "scripts/rebuild_clean_prove_runs.py",
                "--runs_root",
                raw_runs_root,
                "--out_root",
                analysis_runs_root,
                "--task_type",
                "mmlu",
                "--repair_attempts",
                "3",
                "--seed",
                str(args.seed),
                "--critic_model",
                "deepseek-chat",
                "--family_expansion_model",
                "deepseek-chat",
                "--family_taxonomy_path",
                "auto",
                "--family_expansion_enabled",
                "0",
                "--use_dual_family_labels",
                "1",
                "--family_rejudge_on_low_confidence",
                "0",
                "--scan_only",
                "0",
            ],
            workspace,
        )

    if not int(args.skip_analysis):
        run(
            [
                python,
                "scripts/analyze_prove_experiments.py",
                "--runs_root",
                analysis_runs_root,
                "--out_csv",
                "prove_experiments/p3_official_prove_summary.csv",
                "--out_md",
                "prove_experiments/p3_official_prove_summary.md",
                "--out_stats_json",
                "prove_experiments/p3_official_prove_stats.json",
            ],
            workspace,
        )
        run(
            [
                python,
                "scripts/analyze_p3_target_compliance.py",
                "--runs_root",
                analysis_runs_root,
                "--out_dir",
                "prove_experiments/p3_target_compliance",
            ],
            workspace,
        )

    if not int(args.skip_gpt55):
        run(
            [
                python,
                "scripts/run_p3_normal_judge_gpt55.py",
                "--runs_root",
                analysis_runs_root,
                "--out_dir",
                "prove_experiments/p3_normal_judge_gpt55",
                "--sample_size",
                str(args.sample_size),
                "--run_gpt",
                "1",
            ],
            workspace,
        )
        run(
            [
                python,
                "scripts/run_p3_prompt_following_validation.py",
                "--runs_root",
                analysis_runs_root,
                "--out_dir",
                "prove_experiments/p3_prompt_following_gpt55",
                "--sample_size",
                str(args.sample_size),
                "--run_gpt",
                "1",
            ],
            workspace,
        )
        run(
            [
                python,
                "scripts/write_p3_gpt55_reports.py",
                "--normal_rows",
                "prove_experiments/p3_normal_judge_gpt55/p3_normal_judge_analysis_rows.csv",
                "--prompt_rows",
                "prove_experiments/p3_prompt_following_gpt55/p3_prompt_following_analysis_rows.csv",
                "--normal_summary",
                "prove_experiments/p3_normal_judge_gpt55/p3_normal_judge_summary.md",
                "--prompt_summary",
                "prove_experiments/p3_prompt_following_gpt55/p3_prompt_following_summary.md",
                "--combined_summary",
                "prove_experiments/p3_gpt55_combined_summary.md",
            ],
            workspace,
        )

    print("\nP3 official suite completed.")


if __name__ == "__main__":
    main()
