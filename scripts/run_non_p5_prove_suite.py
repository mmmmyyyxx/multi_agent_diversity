import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_STAGES = ["p234", "summary", "p1", "p7", "p6", "p8"]


def _as_bool(value: int) -> bool:
    return bool(int(value))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(workspace: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else workspace / p


def _rel_or_abs(workspace: Path, path: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(str(x) for x in cmd)


def _mirror_unified_gateway_env() -> Dict[str, str]:
    """Mirror OPENAI env vars to the provider-specific env names used by P4 config."""
    changes: Dict[str, str] = {}
    key = os.getenv("OPENAI_API_KEY", "")
    base = os.getenv("OPENAI_BASE_URL", "") or os.getenv("OPENAI_API_BASE", "")

    pairs = {
        "GEMINI_API_KEY": key,
        "GEMINI_OPENAI_BASE_URL": base,
        "OPENROUTER_API_KEY": key,
        "OPENROUTER_BASE_URL": base,
        "OPENAI_API_BASE": base,
    }
    for name, value in pairs.items():
        if value and not os.getenv(name):
            os.environ[name] = value
            changes[name] = value
    return changes


def _check_env(require_api: bool) -> Dict[str, Any]:
    keys = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "GEMINI_API_KEY",
        "GEMINI_OPENAI_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
    ]
    status = {name: bool(os.getenv(name)) for name in keys}
    status["OPENAI_BASE_URL_VALUE"] = os.getenv("OPENAI_BASE_URL", "")
    if require_api and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. 请先在系统环境变量或当前 PowerShell 中设置云雾 API key。")
    if require_api and not (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")):
        raise ValueError("OPENAI_BASE_URL/OPENAI_API_BASE is not set. 请先设置云雾 OpenAI-compatible base url。")
    return status


def _run_step(
    name: str,
    cmd: List[str],
    workspace: Path,
    dry_run: bool,
    skip_existing: bool,
    skip_if_exists: Optional[Path],
    continue_on_error: bool,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "name": name,
        "cmd": cmd,
        "status": "pending",
        "return_code": None,
        "elapsed_sec": 0.0,
        "skip_if_exists": str(skip_if_exists) if skip_if_exists else "",
    }
    print("=" * 120, flush=True)
    print(f"[SUITE] {name}", flush=True)
    print(_quote_cmd(cmd), flush=True)

    if dry_run:
        rec["status"] = "dry_run"
        return rec

    if skip_existing and skip_if_exists and skip_if_exists.exists():
        rec["status"] = "skipped_existing"
        print(f"[SUITE][SKIP] exists: {skip_if_exists}", flush=True)
        return rec

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(workspace), check=False, env=os.environ.copy())
    rec["elapsed_sec"] = time.time() - t0
    rec["return_code"] = int(proc.returncode)
    rec["status"] = "ok" if proc.returncode == 0 else "failed"
    if proc.returncode != 0 and not continue_on_error:
        raise RuntimeError(f"Stage failed: {name}, return_code={proc.returncode}")
    return rec


def _stage_filter(raw: str) -> List[str]:
    if not raw or raw.strip().lower() in {"all", "*"}:
        return list(DEFAULT_STAGES)
    aliases = {
        "p2": "p234",
        "p3": "p234",
        "p4": "p234",
        "p2p3p4": "p234",
        "p234": "p234",
        "analyze": "summary",
        "analyse": "summary",
    }
    out: List[str] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        key = aliases.get(key, key)
        if key not in DEFAULT_STAGES:
            raise ValueError(f"Unknown stage: {item}. Valid: {','.join(DEFAULT_STAGES)}")
        if key not in out:
            out.append(key)
    return out


def _build_p234_cmd(args: argparse.Namespace, workspace: Path, out_root: Path) -> List[str]:
    return [
        args.python,
        "scripts/run_p4_cross_llm_matrix.py",
        "--workspace",
        ".",
        "--python",
        args.python,
        "--models_json",
        _rel_or_abs(workspace, _resolve(workspace, args.models_json)),
        "--out_root",
        _rel_or_abs(workspace, out_root),
        "--task_type",
        args.task_type,
        "--test_path",
        args.test_path,
        "--test_size",
        str(args.test_size),
        "--same_prompts_json",
        args.same_prompts_json,
        "--mixed_prompts_json",
        args.mixed_prompts_json,
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
        "--critic_api_key_env",
        args.critic_api_key_env,
        "--critic_base_url_env",
        args.critic_base_url_env,
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
        "--max_retries",
        str(args.max_retries),
        "--retry_sleep",
        str(args.retry_sleep),
        "--conditions",
        "same,mixed",
        "--skip_existing",
        str(args.skip_existing),
        "--dry_run",
        str(args.dry_run),
    ]


def _build_summary_cmd(args: argparse.Namespace, workspace: Path, out_root: Path) -> List[str]:
    return [
        args.python,
        "scripts/analyze_prove_experiments.py",
        "--runs_root",
        _rel_or_abs(workspace, out_root),
        "--out_csv",
        _rel_or_abs(workspace, out_root / "prove_summary.csv"),
        "--out_md",
        _rel_or_abs(workspace, out_root / "prove_summary.md"),
        "--out_stats_json",
        _rel_or_abs(workspace, out_root / "prove_stats.json"),
        "--bootstrap_iterations",
        str(args.bootstrap_iterations),
        "--seed",
        str(args.seed),
    ]


def _build_p1_cmd(args: argparse.Namespace, workspace: Path, out_root: Path, p1_out: Path) -> List[str]:
    return [
        args.python,
        "scripts/rejudge_strategy_traces.py",
        "--runs_root",
        _rel_or_abs(workspace, out_root),
        "--out_dir",
        _rel_or_abs(workspace, p1_out),
        "--max_per_run",
        str(args.p1_max_per_run),
        "--repeats",
        str(args.p1_repeats),
        "--critic_model",
        args.critic_model,
        "--family_expansion_model",
        args.family_expansion_model,
        "--family_expansion_enabled",
        str(args.family_expansion_enabled),
        "--family_rejudge_on_low_confidence",
        str(args.p1_family_rejudge_on_low_confidence),
        "--family_taxonomy_path",
        args.family_taxonomy_path,
        "--critic_max_tokens",
        str(args.critic_max_tokens),
        "--max_retries",
        str(args.max_retries),
        "--retry_sleep",
        str(args.retry_sleep),
        "--llm_call_timeout",
        str(args.llm_call_timeout),
        "--seed",
        str(args.seed),
    ]


def _build_p7_cmd(args: argparse.Namespace, workspace: Path, out_root: Path, p7_out: Path) -> List[str]:
    return [
        args.python,
        "scripts/run_gpt_blind_validation.py",
        "--runs_root",
        _rel_or_abs(workspace, out_root),
        "--out_dir",
        _rel_or_abs(workspace, p7_out),
        "--per_bucket",
        str(args.p7_per_bucket),
        "--evaluator_model",
        args.p7_evaluator_model,
        "--evaluate",
        str(args.p7_evaluate),
        "--temperature",
        str(args.p7_temperature),
        "--max_tokens",
        str(args.p7_max_tokens),
        "--max_trace_chars",
        str(args.p7_max_trace_chars),
        "--llm_call_timeout",
        str(args.p7_llm_call_timeout),
        "--max_retries",
        str(args.max_retries),
        "--retry_sleep",
        str(args.retry_sleep),
        "--resume",
        str(args.p7_resume),
        "--bootstrap_iterations",
        str(args.bootstrap_iterations),
        "--seed",
        str(args.seed),
    ]


def _build_p6_cmd(args: argparse.Namespace, workspace: Path, out_root: Path, p6_out: Path, p7_out: Path) -> List[str]:
    annotation_path = p7_out / "p7_gpt55_analysis_rows.csv"
    cmd = [
        args.python,
        "scripts/analyze_taxonomy_granularity.py",
        "--runs_root",
        _rel_or_abs(workspace, out_root),
        "--taxonomy_path",
        args.family_taxonomy_path,
        "--out_dir",
        _rel_or_abs(workspace, p6_out),
        "--same_major_family_weight",
        str(args.same_major_family_weight),
        "--macro_diversity_weight",
        str(args.macro_diversity_weight),
        "--bootstrap_iterations",
        str(args.bootstrap_iterations),
        "--seed",
        str(args.seed),
    ]
    # Always pass the intended P7 annotation path. On a fresh full-suite run the
    # command list is built before P7 creates the file, so an existence check here
    # would silently drop the blind-validation correlation from P6.
    cmd.extend(["--blind_annotations", _rel_or_abs(workspace, annotation_path)])
    return cmd


def _build_p8_cmd(args: argparse.Namespace, workspace: Path, out_root: Path, p8_out: Path) -> List[str]:
    return [
        args.python,
        "scripts/analyze_task_dependence.py",
        "--runs_root",
        _rel_or_abs(workspace, out_root),
        "--dataset_name",
        args.dataset_name,
        "--test_path",
        args.test_path,
        "--test_size",
        str(args.p8_test_size),
        "--out_dir",
        _rel_or_abs(workspace, p8_out),
        "--bootstrap_iterations",
        str(args.bootstrap_iterations),
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all proof experiments except P5 reward sweep, then run analyses."
    )
    parser.add_argument("--workspace", type=str, default=str(_repo_root()))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--stages", type=str, default="all", help="Comma subset: p234,summary,p1,p7,p6,p8")
    parser.add_argument("--out_root", type=str, default="prove_experiments/runs")
    parser.add_argument("--status_json", type=str, default="prove_experiments/non_p5_suite_status.json")
    parser.add_argument("--dry_run", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--continue_on_error", type=int, default=0, choices=[0, 1])
    parser.add_argument("--yunwu_unified_env", type=int, default=1, choices=[0, 1])

    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--dataset_name", type=str, default="mmlu")
    parser.add_argument("--test_path", type=str, default="mmlu_test_200.jsonl")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--p8_test_size", type=int, default=100)
    parser.add_argument("--models_json", type=str, default="prove_experiments/p4_low_cost_models.json")
    parser.add_argument("--same_prompts_json", type=str, default="prove_experiments/prompts/same_elimination_mmlu.json")
    parser.add_argument("--mixed_prompts_json", type=str, default="prove_experiments/prompts/mixed_strategy_mmlu.json")

    parser.add_argument("--critic_model", type=str, default="deepseek-chat")
    parser.add_argument("--family_expansion_model", type=str, default="deepseek-chat")
    parser.add_argument("--family_expansion_enabled", type=int, default=0, choices=[0, 1])
    parser.add_argument("--family_rejudge_on_low_confidence", type=int, default=1, choices=[0, 1])
    parser.add_argument("--family_taxonomy_path", type=str, default="auto")
    parser.add_argument("--critic_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--critic_base_url_env", type=str, default="OPENAI_BASE_URL")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--critic_temperature", type=float, default=0.0)
    parser.add_argument("--llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)

    parser.add_argument("--p1_out_dir", type=str, default="prove_experiments/rejudge_p1")
    parser.add_argument("--p1_max_per_run", type=int, default=25)
    parser.add_argument("--p1_repeats", type=int, default=3)
    parser.add_argument("--p1_family_rejudge_on_low_confidence", type=int, default=0, choices=[0, 1])

    parser.add_argument("--p7_out_dir", type=str, default="prove_experiments/p7_gpt55_blind")
    parser.add_argument("--p7_per_bucket", type=int, default=20)
    parser.add_argument("--p7_evaluator_model", type=str, default="gpt-5.5")
    parser.add_argument("--p7_evaluate", type=int, default=1, choices=[0, 1])
    parser.add_argument("--p7_temperature", type=float, default=0.0)
    parser.add_argument("--p7_max_tokens", type=int, default=1200)
    parser.add_argument("--p7_max_trace_chars", type=int, default=3500)
    parser.add_argument("--p7_llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--p7_resume", type=int, default=1, choices=[0, 1])

    parser.add_argument("--p6_out_dir", type=str, default="prove_experiments/p6_taxonomy")
    parser.add_argument("--same_major_family_weight", type=float, default=0.5)
    parser.add_argument("--macro_diversity_weight", type=float, default=0.5)
    parser.add_argument("--p8_out_dir", type=str, default="prove_experiments/p8_task_dependence")
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    out_root = _resolve(workspace, args.out_root).resolve()
    p1_out = _resolve(workspace, args.p1_out_dir).resolve()
    p7_out = _resolve(workspace, args.p7_out_dir).resolve()
    p6_out = _resolve(workspace, args.p6_out_dir).resolve()
    p8_out = _resolve(workspace, args.p8_out_dir).resolve()
    status_path = _resolve(workspace, args.status_json).resolve()
    stages = _stage_filter(args.stages)

    if _as_bool(args.yunwu_unified_env):
        mirrored = _mirror_unified_gateway_env()
    else:
        mirrored = {}
    env_status = _check_env(require_api=(not _as_bool(args.dry_run)))

    out_root.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace": str(workspace),
        "python": args.python,
        "stages": stages,
        "dry_run": _as_bool(args.dry_run),
        "skip_existing": _as_bool(args.skip_existing),
        "yunwu_env_mirrored": sorted(mirrored.keys()),
        "env_status": env_status,
        "steps": [],
    }
    _write_json(status_path, status)

    commands: List[Dict[str, Any]] = []
    if "p234" in stages:
        commands.append(
            {
                "name": "P2/P3/P4 controlled prompt probes and cross-LLM matrix",
                "cmd": _build_p234_cmd(args, workspace, out_root),
                # The child matrix runner has per-run resume logic. Always invoke it
                # so partial matrices can continue instead of being hidden by an old CSV.
                "skip_if_exists": None,
            }
        )
    if "summary" in stages:
        commands.append(
            {
                "name": "Summary and statistical tests for P2/P3/P4",
                "cmd": _build_summary_cmd(args, workspace, out_root),
                "skip_if_exists": None,
            }
        )
    if "p1" in stages:
        commands.append(
            {
                "name": "P1 judge reliability rejudge",
                "cmd": _build_p1_cmd(args, workspace, out_root, p1_out),
                "skip_if_exists": p1_out / "rejudge_summary.md",
            }
        )
    if "p7" in stages:
        commands.append(
            {
                "name": "P7 GPT blind validation",
                "cmd": _build_p7_cmd(args, workspace, out_root, p7_out),
                "skip_if_exists": p7_out / "p7_gpt55_summary.md",
            }
        )
    if "p6" in stages:
        commands.append(
            {
                "name": "P6 taxonomy granularity sensitivity",
                "cmd": _build_p6_cmd(args, workspace, out_root, p6_out, p7_out),
                "skip_if_exists": None,
            }
        )
    if "p8" in stages:
        commands.append(
            {
                "name": "P8 task dependence analysis",
                "cmd": _build_p8_cmd(args, workspace, out_root, p8_out),
                "skip_if_exists": None,
            }
        )

    for spec in commands:
        rec = _run_step(
            name=spec["name"],
            cmd=spec["cmd"],
            workspace=workspace,
            dry_run=_as_bool(args.dry_run),
            skip_existing=_as_bool(args.skip_existing),
            skip_if_exists=spec.get("skip_if_exists"),
            continue_on_error=_as_bool(args.continue_on_error),
        )
        status["steps"].append(rec)
        status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_json(status_path, status)

    failed = [s for s in status["steps"] if s.get("status") == "failed"]
    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["overall_status"] = "failed" if failed else "ok"
    _write_json(status_path, status)

    print("=" * 120, flush=True)
    print(f"[SUITE] finished status={status['overall_status']}", flush=True)
    print(f"[SUITE] status_json={status_path}", flush=True)
    print(f"[SUITE] summary_md={out_root / 'prove_summary.md'}", flush=True)
    print(f"[SUITE] p1={p1_out / 'rejudge_summary.md'}", flush=True)
    print(f"[SUITE] p7={p7_out / 'p7_gpt55_summary.md'}", flush=True)
    print(f"[SUITE] p6={p6_out / 'p6_granularity_summary.md'}", flush=True)
    print(f"[SUITE] p8={p8_out / 'p8_task_dependence_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
