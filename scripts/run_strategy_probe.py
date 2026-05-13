import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.cli import build_dataset
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import TextualGradientRLSystem
from multi_dataset_diverse_rl.utils import ensure_dir, load_jsonl, set_seed


def _load_prompt_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Prompt spec must be a JSON object: {path}")
    agents = spec.get("agents", [])
    if not isinstance(agents, list) or not agents:
        raise ValueError(f"Prompt spec must contain a non-empty agents list: {path}")
    for i, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent prompt entry {i} is not an object.")
        prompt = str(agent.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Agent prompt entry {i} has empty prompt.")
    return spec


def _normalize_agent_entries(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = []
    for i, agent in enumerate(spec.get("agents", [])):
        target = agent.get("target_family", [])
        if isinstance(target, str):
            target = [target]
        if not isinstance(target, list):
            target = []
        entries.append(
            {
                "agent_id": int(agent.get("agent_id", i)),
                "target_family": [str(x) for x in target],
                "prompt": str(agent.get("prompt", "")).strip(),
            }
        )
    entries.sort(key=lambda x: int(x["agent_id"]))
    return entries


def _override_system_prompts(system: TextualGradientRLSystem, entries: List[Dict[str, Any]]):
    if len(entries) != len(system.agents):
        raise ValueError(f"Prompt count {len(entries)} does not match system agents {len(system.agents)}")

    prompts = [str(entry["prompt"]) for entry in entries]
    system.initial_agent_prompts = list(prompts)
    system.initial_agent_prompt_hashes = [system._prompt_hash(p) for p in prompts]
    for i, prompt in enumerate(prompts):
        agent = system.agents[i]
        agent.initial_prompt = prompt
        agent.current_prompt = prompt
        agent.history = [prompt]
        agent.gradient_history = []
        agent.accept_count = 0
        agent.reject_count = 0
        agent.last_update_record = {}
    system.prompt_history = system._init_prompt_history()
    system.flush_prompt_history()
    system.write_run_meta()


def _write_probe_meta(system: TextualGradientRLSystem, cfg: Config, spec: Dict[str, Any], entries: List[Dict[str, Any]], args: argparse.Namespace):
    out_dir = Path(cfg.out_dir)
    payload = {
        "probe_name": str(spec.get("name", out_dir.name)),
        "description": str(spec.get("description", "")),
        "prompts_json": str(args.prompts_json),
        "target_families_by_agent": {
            str(i): entries[i].get("target_family", []) for i in range(len(entries))
        },
        "prompt_hashes": [system._prompt_hash(str(entry["prompt"])) for entry in entries],
        "config": asdict(cfg),
    }
    with (out_dir / "probe_prompts.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **payload,
                "agents": [
                    {
                        "agent_id": i,
                        "target_family": entries[i].get("target_family", []),
                        "prompt_hash": system._prompt_hash(str(entries[i]["prompt"])),
                        "prompt": str(entries[i]["prompt"]),
                    }
                    for i in range(len(entries))
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    meta_path = out_dir / "run_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    meta["probe"] = payload
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


async def main_async():
    parser = argparse.ArgumentParser(description="Run a controlled per-agent prompt strategy probe.")
    parser.add_argument("--task_type", type=str, default="mmlu", choices=["auto", "gsm8k", "mmlu"])
    parser.add_argument("--test_path", type=str, default="mmlu_test_200.jsonl")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--prompts_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--critic_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--family_expansion_enabled", type=int, default=0, choices=[0, 1])
    parser.add_argument("--family_taxonomy_path", type=str, default="auto")
    parser.add_argument("--use_dual_family_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--primary_family_weight", type=float, default=0.7)
    parser.add_argument("--secondary_family_weight", type=float, default=0.3)
    parser.add_argument("--same_major_family_weight", type=float, default=0.5)
    parser.add_argument("--macro_diversity_weight", type=float, default=0.5)
    parser.add_argument("--family_confidence_threshold", type=float, default=0.3)
    parser.add_argument("--family_rejudge_on_low_confidence", type=int, default=1, choices=[0, 1])
    parser.add_argument("--min_summary_words", type=int, default=60)
    parser.add_argument("--max_summary_tokens", type=int, default=512)
    parser.add_argument("--min_evidence_spans", type=int, default=1)

    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--critic_max_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--critic_temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--transient_retry_forever", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_transient_retries", type=int, default=0)
    parser.add_argument("--max_retry_backoff", type=float, default=30.0)
    parser.add_argument("--llm_call_logging", type=int, default=1, choices=[0, 1])
    parser.add_argument("--llm_call_timeout", type=float, default=120.0)
    parser.add_argument("--solver_api_key_env", type=str, default="")
    parser.add_argument("--solver_base_url_env", type=str, default="")
    parser.add_argument("--critic_api_key_env", type=str, default="")
    parser.add_argument("--critic_base_url_env", type=str, default="")
    args = parser.parse_args()

    spec = _load_prompt_spec(args.prompts_json)
    entries = _normalize_agent_entries(spec)
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    cfg = Config(
        task_type=args.task_type,
        model=args.model,
        critic_model=args.critic_model,
        rewriter_model=args.critic_model,
        family_expansion_model=args.family_expansion_model,
        family_expansion_enabled=bool(int(args.family_expansion_enabled)),
        family_taxonomy_path=args.family_taxonomy_path,
        use_dual_family_labels=bool(int(args.use_dual_family_labels)),
        primary_family_weight=args.primary_family_weight,
        secondary_family_weight=args.secondary_family_weight,
        same_major_family_weight=args.same_major_family_weight,
        macro_diversity_weight=args.macro_diversity_weight,
        family_confidence_threshold=args.family_confidence_threshold,
        family_rejudge_on_low_confidence=bool(int(args.family_rejudge_on_low_confidence)),
        min_summary_words=args.min_summary_words,
        max_summary_tokens=args.max_summary_tokens,
        min_evidence_spans=args.min_evidence_spans,
        test_path=args.test_path,
        test_size=args.test_size,
        agents=len(entries),
        init_mode="shared",
        shared_prompt=entries[0]["prompt"],
        baseline_only=True,
        max_tokens=args.max_tokens,
        critic_max_tokens=args.critic_max_tokens,
        temperature=args.temperature,
        critic_temperature=args.critic_temperature,
        out_dir=args.out_dir,
        seed=args.seed,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        transient_retry_forever=bool(int(args.transient_retry_forever)),
        max_transient_retries=args.max_transient_retries,
        max_retry_backoff=args.max_retry_backoff,
        llm_call_logging=bool(int(args.llm_call_logging)),
        llm_call_timeout=args.llm_call_timeout,
        solver_api_key_env=args.solver_api_key_env,
        solver_base_url_env=args.solver_base_url_env,
        critic_api_key_env=args.critic_api_key_env,
        critic_base_url_env=args.critic_base_url_env,
        lambda_diversity=0.0,
        lambda_homogeneity=0.0,
        lambda_invalid_trace=0.30,
    )

    raw_test = load_jsonl(cfg.test_path, cfg.test_size)
    test_data = build_dataset(raw_test)
    system = TextualGradientRLSystem(cfg)
    _override_system_prompts(system, entries)
    _write_probe_meta(system, cfg, spec, entries, args)

    print(f"Loaded test={len(test_data)} probe={spec.get('name', Path(args.out_dir).name)} agents={len(entries)}")
    test_metrics = await system.evaluate_dataset(test_data, split_name="test_epoch1")
    epoch_record = {
        "epoch": 1,
        "train": {
            "mean_family_homogeneity_rate": 0.0,
            "mean_family_diversity": 0.0,
            "mean_llm_direct_diversity_score": 0.0,
            "vote_acc": 0.0,
        },
        "test": test_metrics,
    }
    system.history.append(epoch_record)
    system.save_state("last_state", extra=epoch_record)
    system.save_state("best_state", extra=epoch_record)
    with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(system.history, f, ensure_ascii=False, indent=2)
    system.flush_update_logs()
    system.flush_train_step_logs()
    system.flush_train_trace_history_logs()
    system.flush_test_trace_history_logs()
    system.flush_reasoning_summary_history_logs()
    system.flush_prompt_history()

    print(
        "Probe complete: "
        f"family_div={test_metrics['mean_family_diversity']:.4f}, "
        f"homo={test_metrics['mean_family_homogeneity_rate']:.4f}, "
        f"vote_acc={test_metrics['vote_acc']:.4f}, "
        f"out_dir={cfg.out_dir}"
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
