from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url
from scripts.solver_headroom_multimodel_support import CANDIDATES, ROLE_MODEL, RUN_ROOT, git, read_json, write_json


async def one(client: AsyncOpenAI, model: str) -> dict[str, Any]:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Return exactly SMOKE_OK."}, {"role": "user", "content": "Compatibility smoke."}],
            temperature=0, max_tokens=16, timeout=60,
            extra_body={"enable_thinking": False},
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return {
            "attempted": True, "success": True, "status_code": 200,
            "error_type": "", "finish_reason": str(response.choices[0].finish_reason or ""),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "response_hash": hashlib.sha256(text.encode()).hexdigest(),
        }
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        return {"attempted": True, "success": False, "status_code": int(status) if status else None, "error_type": type(exc).__name__, "finish_reason": "", "prompt_tokens": 0, "completion_tokens": 0, "response_hash": ""}


async def main_async() -> None:
    if os.environ.get("SOLVER_MULTIMODEL_SMOKE_AUTHORIZED") != "1":
        raise RuntimeError("smoke not authorized")
    out = RUN_ROOT / "phase_a/availability_smoke_private.json"
    if out.exists():
        raise RuntimeError("fresh smoke output required")
    freeze = read_json(RUN_ROOT / "freeze/source_freeze.json")
    if git("rev-parse", "HEAD") != freeze["execution_commit"] or git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("source identity mismatch")
    _, key = resolve_api_key("DASHSCOPE_API_KEY")
    _, base = resolve_base_url("DASHSCOPE_BASE_URL")
    client = AsyncOpenAI(api_key=key, base_url=base)
    listed = {str(item.id) for item in (await client.models.list()).data}
    rows = []
    for priority, (model_key, model) in enumerate(CANDIDATES, 1):
        visible = model in listed
        smoke = await one(client, model) if visible and model != "qwen3-8b" else ({
            "attempted": False, "success": True, "status_code": 200,
            "error_type": "reused_prior_passed_smoke" if model == "qwen3-8b" else "not_listed",
            "finish_reason": "", "prompt_tokens": 0, "completion_tokens": 0,
            "response_hash": "",
        } if visible else {
            "attempted": False, "success": False, "status_code": None,
            "error_type": "not_listed", "finish_reason": "", "prompt_tokens": 0,
            "completion_tokens": 0, "response_hash": "",
        })
        rows.append({"key": model_key, "model": model, "priority": priority, "listed": visible, "smoke": smoke, "static_eligible": visible and smoke["success"]})
    payload = {
        "gate": "PASS" if any(r["static_eligible"] for r in rows) else "HOLD",
        "candidates": rows, "role_model": ROLE_MODEL,
        "qwen3_8b_smoke_reused": True,
        "successful_new_smoke_calls": sum(int(r["smoke"]["success"] and r["smoke"]["attempted"]) for r in rows),
        "test_calls": 0,
    }
    write_json(out, payload)
    print({"gate": payload["gate"], "entrants": [r["model"] for r in rows if r["static_eligible"]], "new_smokes": payload["successful_new_smoke_calls"]})


if __name__ == "__main__":
    asyncio.run(main_async())
