from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url
from scripts.solver_headroom_screening_support import (
    CANDIDATES, ROLE_MODEL, RUN_ROOT, git, read_json, write_json,
)


async def smoke(client: AsyncOpenAI, model: str) -> dict[str, Any]:
    started = time.time()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return exactly SMOKE_OK."},
                {"role": "user", "content": "Compatibility smoke."},
            ],
            temperature=0,
            max_tokens=16,
            timeout=60,
            extra_body={"enable_thinking": False},
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return {
            "attempted": True,
            "success": True,
            "status_code": 200,
            "error_type": "",
            "finish_reason": str(response.choices[0].finish_reason or ""),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "response_nonempty": bool(text.strip()),
            "response_hash": hashlib.sha256(text.encode()).hexdigest(),
            "latency_seconds": round(time.time() - started, 6),
        }
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        return {
            "attempted": True,
            "success": False,
            "status_code": int(status) if status is not None else None,
            "error_type": type(exc).__name__,
            "finish_reason": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "response_nonempty": False,
            "response_hash": "",
            "latency_seconds": round(time.time() - started, 6),
        }


async def run(freeze_path: Path, out: Path) -> dict[str, Any]:
    if os.environ.get("SOLVER_HEADROOM_SMOKE_AUTHORIZED") != "1":
        raise RuntimeError("availability smoke is not authorized")
    if out.exists():
        raise RuntimeError("fresh Phase A output required")
    freeze = read_json(freeze_path)
    if git("rev-parse", "HEAD") != freeze["execution_commit"]:
        raise RuntimeError("execution commit mismatch")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("worktree must be clean")
    _, key = resolve_api_key("DASHSCOPE_API_KEY")
    _, base = resolve_base_url("DASHSCOPE_BASE_URL")
    if not key or not base:
        raise RuntimeError("provider credentials unavailable")
    client = AsyncOpenAI(api_key=key, base_url=base)
    listed = {str(item.id) for item in (await client.models.list()).data}
    candidate_rows: list[dict[str, Any]] = []
    for priority, (model_key, model) in enumerate(CANDIDATES, start=1):
        is_listed = model in listed
        result = await smoke(client, model) if is_listed else {
            "attempted": False, "success": False, "status_code": None,
            "error_type": "not_in_credential_model_inventory", "finish_reason": "",
            "prompt_tokens": 0, "completion_tokens": 0,
            "response_nonempty": False, "response_hash": "", "latency_seconds": 0.0,
        }
        candidate_rows.append({
            "model_key": model_key,
            "solver_model": model,
            "priority": priority,
            "listed": is_listed,
            "smoke": result,
            "screening_eligible": is_listed and result["success"],
        })
    role_listed = ROLE_MODEL in listed
    role_smoke = await smoke(client, ROLE_MODEL) if role_listed else {
        "attempted": False, "success": False, "status_code": None,
        "error_type": "not_in_credential_model_inventory", "finish_reason": "",
        "prompt_tokens": 0, "completion_tokens": 0,
        "response_nonempty": False, "response_hash": "", "latency_seconds": 0.0,
    }
    payload = {
        "phase": "availability_smoke",
        "gate": "PASS" if any(r["screening_eligible"] for r in candidate_rows)
        and role_listed and role_smoke["success"] else "HOLD",
        "execution_commit": freeze["execution_commit"],
        "candidate_inventory_count": len(CANDIDATES),
        "candidates": candidate_rows,
        "role_model": {
            "model": ROLE_MODEL, "listed": role_listed, "smoke": role_smoke,
        },
        "excluded_model_requested": False,
        "successful_smoke_call_count": sum(
            int(row["smoke"]["success"]) for row in candidate_rows
        ) + int(role_smoke["success"]),
        "test_calls": 0,
        "full_method_run": False,
    }
    write_json(out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path,
        default=RUN_ROOT / "phase_a" / "availability_smoke_private.json",
    )
    args = parser.parse_args()
    payload = asyncio.run(run(args.freeze, args.out))
    print({
        "gate": payload["gate"],
        "eligible": [
            row["solver_model"] for row in payload["candidates"]
            if row["screening_eligible"]
        ],
        "role_model_smoke": payload["role_model"]["smoke"]["success"],
        "test_calls": 0,
    })
    raise SystemExit(0 if payload["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
