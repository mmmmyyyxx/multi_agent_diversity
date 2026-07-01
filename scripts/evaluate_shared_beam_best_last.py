import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.cli import build_dataset
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem
from multi_dataset_diverse_rl.utils import load_jsonl


def _read_json(path: Path, default: Any):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _restore_prompt_history(run_dir: Path, backup: str):
    if backup:
        (run_dir / "prompt_history.json").write_text(backup, encoding="utf-8")


def _restore_best_prompts(system: TraceBeamSearchSystem, best_payload: Dict[str, Any]):
    agents = best_payload.get("agents", [])
    if not isinstance(agents, list) or len(agents) != len(system.agents):
        raise RuntimeError(f"best_prompts has {len(agents) if isinstance(agents, list) else 0} agents, expected {len(system.agents)}")
    for row in agents:
        agent_id = int(row.get("agent_id", -1))
        prompt = str(row.get("prompt", "") or "")
        if not (0 <= agent_id < len(system.agents)) or not prompt:
            raise RuntimeError(f"invalid best prompt row: {row}")
        agent = system.agents[agent_id]
        agent.current_prompt = prompt
        agent.prompt_beam = [system._make_beam_item(prompt, None, {}, None, 0)]


def _restore_last_state(system: TraceBeamSearchSystem, state_payload: Dict[str, Any]):
    agents = state_payload.get("agents", [])
    if not isinstance(agents, list) or len(agents) != len(system.agents):
        raise RuntimeError(f"last_state has {len(agents) if isinstance(agents, list) else 0} agents, expected {len(system.agents)}")
    for row in agents:
        agent_id = int(row.get("agent_id", -1))
        if not (0 <= agent_id < len(system.agents)):
            raise RuntimeError(f"invalid last_state agent row: {row}")
        agent = system.agents[agent_id]
        prompt = str(row.get("current_prompt", "") or agent.current_prompt)
        agent.current_prompt = prompt
        beam = row.get("prompt_beam", [])
        if isinstance(beam, list) and beam:
            agent.prompt_beam = [dict(x) for x in beam if isinstance(x, dict)]
        if not agent.prompt_beam:
            agent.prompt_beam = [system._make_beam_item(prompt, None, {}, None, 0)]
        if str(agent.prompt_beam[0].get("prompt", "")) != prompt:
            agent.prompt_beam[0]["prompt"] = prompt


async def main_async():
    parser = argparse.ArgumentParser(description="Evaluate shared_beam best and last prompts on the test set.")
    parser.add_argument("--run_dir", type=str, default="runs_mmlu_subject_balanced_default_size_4way/shared_beam_seed42")
    parser.add_argument("--eval_solver_call_concurrency", type=int, default=225)
    parser.add_argument("--llm_call_timeout", type=float, default=240.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    meta = _read_json(run_dir / "run_meta.json", {})
    cfg_dict = dict(meta.get("config", {}))
    cfg_dict["out_dir"] = str(run_dir.resolve())
    cfg_dict["eval_solver_call_concurrency"] = int(args.eval_solver_call_concurrency)
    cfg_dict["llm_call_timeout"] = float(args.llm_call_timeout)
    cfg = Config(**cfg_dict)
    cfg.llm_call_logging = True

    prompt_history_backup = ""
    prompt_history_path = run_dir / "prompt_history.json"
    if prompt_history_path.exists():
        prompt_history_backup = prompt_history_path.read_text(encoding="utf-8")

    test_data = build_dataset(load_jsonl(cfg.test_path, cfg.test_size))
    system = TraceBeamSearchSystem(cfg)
    _restore_prompt_history(run_dir, prompt_history_backup)

    best_payload = _read_json(run_dir / "best_prompts.json", {})
    last_payload = _read_json(run_dir / "last_state.json", {})

    print(f"Evaluating best prompts on test: size={len(test_data)}", flush=True)
    _restore_best_prompts(system, best_payload)
    best_metrics = await system.evaluate_dataset(test_data, split_name="test_best")
    _restore_prompt_history(run_dir, prompt_history_backup)
    print("test_best metrics:", json.dumps(best_metrics, ensure_ascii=False), flush=True)

    print(f"Evaluating last prompts on test: size={len(test_data)}", flush=True)
    _restore_last_state(system, last_payload)
    last_metrics = await system.evaluate_dataset(test_data, split_name="test_last")
    _restore_prompt_history(run_dir, prompt_history_backup)
    print("test_last metrics:", json.dumps(last_metrics, ensure_ascii=False), flush=True)

    summary = {
        "best": {
            "source": "best_prompts.json",
            "selected_epoch": best_payload.get("selected_epoch"),
            "best_validation_score": best_payload.get("best_validation_score"),
            "metrics": best_metrics,
        },
        "last": {
            "source": "last_state.json",
            "metrics": last_metrics,
        },
    }
    out_path = run_dir / "test_best_last_metrics.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
