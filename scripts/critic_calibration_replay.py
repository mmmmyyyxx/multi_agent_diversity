from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config, add_config_arguments, config_from_args
from multi_dataset_diverse_rl.critic_calibration import calibration_context, calibration_items
from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient
from multi_dataset_diverse_rl.persistence.artifacts import ArtifactWriter
from multi_dataset_diverse_rl.tcs import build_critic_request, parse_critic_decision
from multi_dataset_diverse_rl.utils import extract_json_obj


def _excerpt(value: str, limit: int = 600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    half = max(1, (limit - 24) // 2)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


async def run(
    cfg: Config,
    client: RoleAwareLLMClient | None = None,
) -> dict[str, Any]:
    client = client or RoleAwareLLMClient(cfg)
    context = calibration_context()
    rows = []
    for item in calibration_items():
        request = build_critic_request(context, item.proposal)
        decision = None
        parse_error = ""
        attempts = []
        for attempt_index in range(cfg.tcs.critic_json_max_retries + 1):
            user_request = (
                "Audit this calibration proposal."
                if attempt_index == 0
                else (
                    "Your previous response was invalid because "
                    f"{parse_error}. Copy DERIVED_CASE_FACTS exactly and return corrected strict JSON."
                )
            )
            raw = await client.chat(
                cfg.models.evaluator_model,
                request,
                user_request,
                cfg.tcs.critic_temperature,
                cfg.tcs.critic_max_tokens,
                "evaluator",
            )
            parsed = extract_json_obj(raw)
            parse_error = ""
            try:
                if parsed is None:
                    raise ValueError("critic response is not JSON")
                decision = parse_critic_decision(
                    parsed,
                    allowed_case_ids=set(),
                    feedback_max_chars=cfg.tcs.critic_feedback_max_chars,
                )
            except (KeyError, TypeError, ValueError) as exc:
                parse_error = str(exc)
            attempts.append({
                "attempt_index": attempt_index,
                "response_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "json_extracted": parsed is not None,
                "schema_valid": decision is not None,
                "parse_error": parse_error,
                "response_excerpt": _excerpt(raw),
            })
            if decision is not None:
                break
        rows.append({
            "name": item.name,
            "category": item.category,
            "expected_approved": item.expected_approved,
            "schema_valid": decision is not None,
            "actual_approved": decision.approved if decision is not None else None,
            "classification_correct": (
                decision is not None and decision.approved == item.expected_approved
            ),
            "decision": asdict(decision) if decision is not None else None,
            "attempts": attempts,
        })

    good = [row for row in rows if row["category"] == "good"]
    memorizing = [row for row in rows if row["category"] == "memorizing"]
    schema_valid_count = sum(bool(row["schema_valid"]) for row in rows)
    good_acceptance_count = sum(bool(row["actual_approved"]) for row in good)
    memorizing_rejection_count = sum(row["actual_approved"] is False for row in memorizing)
    cost = client.cost_summary()
    report = {
        "ok": bool(
            schema_valid_count == len(rows)
            and good_acceptance_count > 0
            and memorizing_rejection_count == len(memorizing)
            and sum(bool(row["classification_correct"]) for row in rows) == len(rows)
            and cost["solver_calls"] == 0
            and cost["optimizer_calls"] == 0
        ),
        "method_version": cfg.training.method_version,
        "criteria": {
            "all_schema_valid": schema_valid_count == len(rows),
            "good_proposal_acceptance_gt_zero": good_acceptance_count > 0,
            "memorizing_proposal_rejection_rate": (
                memorizing_rejection_count / len(memorizing) if memorizing else 1.0
            ),
            "all_labeled_classifications_correct": (
                sum(bool(row["classification_correct"]) for row in rows) == len(rows)
            ),
            "solver_calls_zero": cost["solver_calls"] == 0,
            "optimizer_calls_zero": cost["optimizer_calls"] == 0,
        },
        "summary": {
            "items": len(rows),
            "schema_valid_count": schema_valid_count,
            "good_acceptance_count": good_acceptance_count,
            "memorizing_rejection_count": memorizing_rejection_count,
            "classification_correct_count": sum(
                bool(row["classification_correct"]) for row in rows
            ),
        },
        "items": rows,
        "cost": cost,
    }
    ArtifactWriter(cfg.persistence.out_dir).write_json(
        "critic_calibration_report.json",
        report,
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Replay a human-labeled Critic calibration set using only the evaluator.",
    )
    return add_config_arguments(value)


def main() -> int:
    report = asyncio.run(run(config_from_args(parser().parse_args())))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
