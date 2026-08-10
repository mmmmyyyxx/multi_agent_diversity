from __future__ import annotations

import asyncio
import gc
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import (
    PersistentSolverCache,
)
from multi_dataset_diverse_rl.evaluation.prompt_question import (
    PromptAnswer,
    PromptQuestionEvaluator,
)
from scripts.run_task_level_accuracy import (
    _assert_runner_owned_child_command,
    _build_child_command,
    _prepare_setting_local_cache,
    _solver_cache_snapshot,
    _sqlite_backup,
    _with_runner_owned_paths,
)


def _evaluator(path: Path | None = None) -> PromptQuestionEvaluator:
    return PromptQuestionEvaluator(
        model_request_identity="deterministic-runner-isolation-v1",
        parser_version="deterministic-parser-v1",
        temperature=0.0,
        decoding_seed=46,
        cache_metadata={
            "solver_model": "offline-solver",
            "endpoint_identity": "offline-endpoint",
            "output_contract_version": "offline-contract",
            "max_tokens": 1800,
        },
        shared_cache=(PersistentSolverCache(path) if path is not None else None),
    )


def _seed_cache(path: Path, *, correct_count: int) -> None:
    evaluator = _evaluator()
    cache = PersistentSolverCache(path)
    for index in range(75):
        prompt_hash = "shared-prompt"
        question_hash = f"probe-{index}"
        key = evaluator.key(prompt_hash, question_hash)
        metadata = {
            **evaluator.cache_metadata,
            "model_request_identity": evaluator.model_request_identity,
            "parser_version": evaluator.parser_version,
            "temperature": evaluator.temperature,
            "evaluation_replica_seed": evaluator.decoding_seed,
            "prompt_hash": prompt_hash,
            "question_hash": question_hash,
        }
        if cache._claim_or_read(key, metadata)[0] != "owner":
            raise AssertionError("deterministic seed cache unexpectedly reused a row")
        answer = "A" if index < correct_count else "B"
        cache._store(
            key,
            PromptAnswer(answer, f"offline-trace-{answer}-{index}", True),
        )


async def _read_initialization(local_cache: Path) -> tuple[int, int]:
    evaluator = _evaluator(local_cache)
    provider_calls = 0
    correct = 0
    for index in range(75):
        async def forbidden_provider(*_args):
            nonlocal provider_calls
            provider_calls += 1
            raise AssertionError("frozen observation unexpectedly called provider")

        answer = await evaluator.evaluate(
            question=f"probe-{index}",
            question_hash=f"probe-{index}",
            prompt="shared-prompt",
            prompt_hash="shared-prompt",
            agent_id=0,
            solve=forbidden_provider,
        )
        correct += int(answer.answer == "A")
    return correct, provider_calls


def main() -> int:
    with TemporaryDirectory(prefix="runner-cache-isolation-") as temp:
        root = Path(temp)
        frozen = root / "frozen" / "initial_solver_cache_frozen.sqlite"
        reference = root / "frozen" / "comparison_reference_solver_cache.sqlite"
        hostile = root / "hostile" / "_shared_solver_cache.sqlite"
        run_dir = (
            root
            / "runs"
            / "shared_responsibility_conditioned_dual_target_seed46"
        )
        local = run_dir / "_solver_cache.sqlite"
        manifest = root / "frozen" / "frozen_initialization_manifest.json"

        _seed_cache(frozen, correct_count=62)
        _seed_cache(hostile, correct_count=60)
        _sqlite_backup(frozen, reference)
        manifest.write_text(json.dumps({
            "initialization_snapshot": {
                "initial_member_correct_counts": [62, 62, 62, 62, 62],
            },
        }), encoding="utf-8")
        hostile_hash_before = _solver_cache_snapshot(hostile)["content_hash"]

        clone_audit = _prepare_setting_local_cache(
            comparison_reference_cache_path=reference,
            frozen_cache_path=frozen,
            setting_local_cache_path=local,
            expected_evaluator=_evaluator(),
        )
        child_config = _with_runner_owned_paths(
            {
                "shared_solver_cache_path": str(hostile),
                "frozen_initialization_manifest_path": str(root / "wrong.json"),
            },
            run_dir=run_dir,
            solver_cache_path=local,
            frozen_manifest_path=manifest,
        )
        child_command = _build_child_command(
            child_config, python_executable="offline-python",
        )
        ownership = _assert_runner_owned_child_command(
            child_command,
            run_dir=run_dir,
            frozen_manifest_path=manifest,
            comparison_reference_cache_path=reference,
        )
        correct_count, provider_calls = asyncio.run(
            _read_initialization(local)
        )
        actual_member_counts = [correct_count] * 5
        expected_member_counts = json.loads(
            manifest.read_text(encoding="utf-8")
        )["initialization_snapshot"]["initial_member_correct_counts"]
        hostile_hash_after = _solver_cache_snapshot(hostile)["content_hash"]

        result = {
            "runner_cache_isolation_smoke": {
                **ownership,
                "child_cache_is_setting_local": (
                    Path(child_config["shared_solver_cache_path"]).resolve()
                    == local.resolve()
                ),
                "child_cache_is_not_reference": local.resolve() != reference.resolve(),
                "child_cache_is_not_hostile_root": local.resolve() != hostile.resolve(),
                "frozen_observation_set_matched": clone_audit[
                    "frozen_observation_set_matched"
                ],
                "frozen_ready_entry_count": clone_audit[
                    "frozen_ready_entry_count"
                ],
                "initialization_correct_count": correct_count,
                "initialization_probe_count": 75,
                "frozen_guard_pass": actual_member_counts == expected_member_counts,
                "provider_calls_for_frozen_observations": provider_calls,
                "hostile_root_cache_unchanged": (
                    hostile_hash_before == hostile_hash_after
                ),
                "api_calls": 0,
            },
        }
        checks = result["runner_cache_isolation_smoke"]
        if not all((
            checks["runner_owned_cache_path"],
            checks["runner_owned_frozen_manifest"],
            checks["setting_local_cache_isolated"],
            checks["child_cache_is_setting_local"],
            checks["child_cache_is_not_reference"],
            checks["child_cache_is_not_hostile_root"],
            checks["frozen_observation_set_matched"],
            checks["frozen_guard_pass"],
            checks["provider_calls_for_frozen_observations"] == 0,
            checks["hostile_root_cache_unchanged"],
            checks["api_calls"] == 0,
        )):
            raise AssertionError(json.dumps(result, sort_keys=True))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        gc.collect()
        time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
