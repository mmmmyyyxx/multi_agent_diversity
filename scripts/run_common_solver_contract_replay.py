"""Execute the authorized COMMON_SOLVER_CONTRACT_V1 ExternalValidation50 replay.

This is deliberately a single evaluator entry point for all source methods. It
cannot open Test50 or run optimization. Real provider calls require an explicit
environment authorization token.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI

from infrastructure.common_solver_contract_v1.contract import (
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
    contract_identity,
)
from infrastructure.common_solver_contract_v1.evaluator import (
    CommonSolverEvaluator,
    TransportResponse,
)
from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temp.replace(path)


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    if status is not None:
        return int(status) in CONTRACT_SPEC.retry_status_codes
    return isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError))


def _plurality(labels: list[str | None]) -> str | None:
    valid = [value for value in labels if value]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == best)
    return winners[0] if len(winners) == 1 else None


async def run(registry_path: Path, output: Path) -> dict[str, Any]:
    if os.environ.get("COMMON_SOLVER_REPLAY_AUTHORIZED") != "1":
        raise PermissionError("COMMON_SOLVER_REPLAY_AUTHORIZED=1 is required")
    registry = read_json(registry_path)
    if registry["contract_id"] != COMMON_SOLVER_CONTRACT_ID:
        raise ValueError("contract id mismatch")
    if registry["contract_identity"] != contract_identity():
        raise ValueError("contract identity mismatch")
    if registry["split"] != "ExternalValidation50" or len(registry["cases"]) != 50:
        raise ValueError("only the frozen ExternalValidation50 registry is allowed")
    if registry.get("test50_accessed") or registry.get("selection_or_optimization"):
        raise ValueError("registry violates evaluation isolation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)

    _, key = resolve_api_key("DASHSCOPE_API_KEY")
    _, base = resolve_base_url("DASHSCOPE_BASE_URL")
    if not key or not base:
        raise ValueError("provider credentials are not configured")
    client = AsyncOpenAI(api_key=key, base_url=base)

    async def transport(request: dict) -> TransportResponse:
        response = await client.chat.completions.create(
            **request,
            timeout=CONTRACT_SPEC.timeout_seconds,
        )
        usage = response.usage
        return TransportResponse(
            text=response.choices[0].message.content or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=str(response.choices[0].finish_reason or ""),
        )

    raw_cache: dict[str, str] = {}
    evaluator = CommonSolverEvaluator(
        transport=transport,
        cache=raw_cache,
        retryable=_retryable,
    )
    private_rows: list[dict[str, Any]] = []
    state_summaries: list[dict[str, Any]] = []
    for state in registry["states"]:
        prompts = [str(value) for value in state["ordered_prompts"]]
        correct_count = oracle_count = valid_vote_count = 0
        member_correct = [0 for _ in prompts]
        for case in registry["cases"]:
            labels: list[str | None] = []
            valids: list[bool] = []
            identities: list[str] = []
            for index, prompt in enumerate(prompts):
                result = await evaluator.evaluate(
                    decision_procedure=prompt,
                    question=str(case["question"]),
                )
                answer = result.response.answer if result.response.valid else None
                labels.append(answer)
                valids.append(result.response.valid)
                identities.append(result.request_identity)
                if answer == case["gold"]:
                    member_correct[index] += 1
            vote = _plurality(labels)
            vote_correct = vote == case["gold"]
            oracle_correct = any(label == case["gold"] for label in labels)
            correct_count += int(vote_correct)
            oracle_count += int(oracle_correct)
            valid_vote_count += int(vote is not None)
            private_rows.append(
                {
                    "state_id": state["state_id"],
                    "case_position": case["position"],
                    "case_id": case["case_id"],
                    "member_labels": labels,
                    "member_valid": valids,
                    "request_identities": identities,
                    "vote_label": vote,
                    "vote_correct": vote_correct,
                    "oracle_correct": oracle_correct,
                }
            )
        state_summaries.append(
            {
                "state_id": state["state_id"],
                "source_repo": state["source_repo"],
                "source_seed": state["source_seed"],
                "member_count": len(prompts),
                "vote_correct": correct_count,
                "vote_accuracy": correct_count / 50,
                "oracle_correct": oracle_count,
                "oracle_accuracy": oracle_count / 50,
                "valid_vote_count": valid_vote_count,
                "individual_member_correct": member_correct,
                "individual_member_accuracy": [value / 50 for value in member_correct],
            }
        )
        write_json_atomic(output / "progress_private.json", {"completed_states": state_summaries})
        write_json_atomic(output / "predictions_private.json", private_rows)
        write_json_atomic(output / "exact_request_cache_private.json", raw_cache)

    summary = {
        "status": "PASS",
        "contract_id": COMMON_SOLVER_CONTRACT_ID,
        "contract_identity": contract_identity(),
        "split": "ExternalValidation50",
        "test50_accessed": False,
        "selection_or_optimization": False,
        "states": state_summaries,
        "accounting": evaluator.accounting(),
    }
    write_json_atomic(output / "summary_private.json", summary)
    write_json_atomic(
        output / "execution_manifest.json",
        {
            "status": "COMPLETE",
            "contract_id": COMMON_SOLVER_CONTRACT_ID,
            "contract_identity": contract_identity(),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "test50_accessed": False,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT
        / "runs/common_solver_contract_v1_prep_20260906/private_replay_registry.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = args.registry.resolve()
    output = args.output.resolve()
    if PROJECT_ROOT.resolve() not in output.parents:
        raise ValueError("output must remain inside the Diversity repository")
    result = asyncio.run(run(registry, output))
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
