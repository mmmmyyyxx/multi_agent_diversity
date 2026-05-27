import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["alias"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _env_name(value: str) -> str:
    return str(value or "").strip()


def _build_cmd(args: argparse.Namespace, model_spec: Dict[str, Any], prompt_name: str, prompt_path: str, out_dir: Path) -> List[str]:
    cmd = [
        args.python,
        "scripts/run_strategy_probe.py",
        "--task_type",
        args.task_type,
        "--test_path",
        args.test_path,
        "--test_size",
        str(args.test_size),
        "--prompts_json",
        prompt_path,
        "--out_dir",
        str(out_dir),
        "--model",
        str(model_spec["model"]),
        "--critic_model",
        args.critic_model,
        "--family_expansion_model",
        args.family_expansion_model,
        "--family_expansion_enabled",
        str(args.family_expansion_enabled),
        "--family_rejudge_on_low_confidence",
        str(args.family_rejudge_on_low_confidence),
        "--family_taxonomy_path",
        args.family_taxonomy_path,
        "--seed",
        str(args.seed),
        "--max_tokens",
        str(args.max_tokens),
        "--critic_max_tokens",
        str(args.critic_max_tokens),
        "--temperature",
        str(args.temperature),
        "--critic_temperature",
        str(args.critic_temperature),
        "--llm_call_timeout",
        str(args.llm_call_timeout),
        "--eval_parallelism",
        str(args.eval_parallelism),
        "--max_retries",
        str(args.max_retries),
        "--retry_sleep",
        str(args.retry_sleep),
        "--critic_api_key_env",
        args.critic_api_key_env,
        "--critic_base_url_env",
        args.critic_base_url_env,
    ]
    solver_key_env = _env_name(model_spec.get("solver_api_key_env", ""))
    solver_base_env = _env_name(model_spec.get("solver_base_url_env", ""))
    if solver_key_env:
        cmd.extend(["--solver_api_key_env", solver_key_env])
    if solver_base_env:
        cmd.extend(["--solver_base_url_env", solver_base_env])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run cross-LLM low-cost strategy-transfer matrix.")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--models_json", type=str, default="prove_experiments/p4_low_cost_models.json")
    parser.add_argument("--out_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--run_prefix", type=str, default="P4")
    parser.add_argument("--summary_name", type=str, default="")
    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--test_path", type=str, default="mmlu_test_200.jsonl")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--same_prompts_json", type=str, default="prove_experiments/prompts/same_elimination_mmlu.json")
    parser.add_argument("--definition_prompts_json", type=str, default="prove_experiments/prompts/same_definition_mmlu.json")
    parser.add_argument("--mixed_prompts_json", type=str, default="prove_experiments/prompts/mixed_strategy_mmlu.json")
    parser.add_argument("--same_exact_prompts_json", type=str, default="prove_experiments/prompts/same_prompt_mmlu.json")
    parser.add_argument("--critic_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_enabled", type=int, default=0, choices=[0, 1])
    parser.add_argument("--family_rejudge_on_low_confidence", type=int, default=1, choices=[0, 1])
    parser.add_argument("--family_taxonomy_path", type=str, default="auto")
    parser.add_argument("--critic_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--critic_base_url_env", type=str, default="OPENAI_BASE_URL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--critic_temperature", type=float, default=0.0)
    parser.add_argument("--llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--eval_parallelism", type=int, default=100)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument(
        "--conditions",
        type=str,
        default="same,mixed",
        help="Comma-separated subset: same,mixed,definition (same_elimination, mixed_strategy, same_definition)",
    )
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dry_run", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    models = _read_json((workspace / args.models_json).resolve())
    if not isinstance(models, list) or not models:
        raise ValueError(f"No model specs found in {args.models_json}")
    out_root = (workspace / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    wanted = {x.strip() for x in args.conditions.split(",") if x.strip()}
    condition_specs = []
    if "same" in wanted or "same_elimination" in wanted:
        condition_specs.append(("same_elimination", args.same_prompts_json))
    if "same_prompt" in wanted or "same_exact" in wanted:
        condition_specs.append(("same_prompt", args.same_exact_prompts_json))
    if "mixed" in wanted or "mixed_strategy" in wanted:
        condition_specs.append(("mixed_strategy", args.mixed_prompts_json))
    if "definition" in wanted or "same_definition" in wanted:
        condition_specs.append(("same_definition", args.definition_prompts_json))

    rows: List[Dict[str, Any]] = []
    run_prefix = str(args.run_prefix or "P4").strip() or "P4"
    summary_name = str(args.summary_name or f"{run_prefix.lower()}_cross_llm_runs.csv").strip()
    summary_path = out_root / summary_name
    for spec in models:
        alias = str(spec.get("alias", spec.get("model", "model"))).replace("/", "_").replace(" ", "_")
        for condition_name, prompts_path in condition_specs:
            run_name = f"{run_prefix}_{condition_name}_{alias}_seed{args.seed}"
            out_dir = out_root / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            if int(args.skip_existing) and (out_dir / "history.json").exists():
                rows.append({"run_name": run_name, "alias": alias, "model": spec.get("model", ""), "condition": condition_name, "status": "skipped_existing", "out_dir": str(out_dir)})
                _write_csv(rows, summary_path)
                continue
            cmd = _build_cmd(args, spec, condition_name, prompts_path, out_dir)
            print("=" * 120)
            print(f"[{run_prefix}] {run_name}")
            print("Command:", " ".join(cmd))
            if int(args.dry_run):
                status = "dry_run"
                rc = 0
                elapsed = 0.0
            else:
                t0 = time.time()
                proc = subprocess.run(cmd, cwd=str(workspace), check=False, env=os.environ.copy())
                elapsed = time.time() - t0
                rc = proc.returncode
                status = "ok" if rc == 0 else "failed"
            rows.append(
                {
                    "run_name": run_name,
                    "alias": alias,
                    "model": spec.get("model", ""),
                    "provider": spec.get("provider", ""),
                    "condition": condition_name,
                    "status": status,
                    "return_code": rc,
                    "elapsed_sec": elapsed,
                    "out_dir": str(out_dir),
                    "solver_api_key_env": spec.get("solver_api_key_env", ""),
                    "solver_base_url_env": spec.get("solver_base_url_env", ""),
                }
            )
            _write_csv(rows, summary_path)
    print(f"{run_prefix} matrix finished: {summary_path}")


if __name__ == "__main__":
    main()
