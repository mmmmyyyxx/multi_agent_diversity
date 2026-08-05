from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    mutable_prompt_violation_reasons,
)
from multi_dataset_diverse_rl.utils import normalize_prompt_text
from multi_dataset_diverse_rl.versions import MUTABLE_PROMPT_CONTRACT_VERSION


AUDIT_SCHEMA_VERSION = "mutable_prompt_contamination_hash_audit_v1"


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_run_directory(run_dir: Path) -> dict[str, Any]:
    prompts_path = run_dir / "best_prompts.json"
    prompts = _load_json(prompts_path)
    if not isinstance(prompts, list) or any(
        not isinstance(prompt, str) for prompt in prompts
    ):
        raise ValueError("best_prompts.json must contain a list of strings")
    meta_path = run_dir / "run_meta.json"
    meta = _load_json(meta_path) if meta_path.is_file() else {}
    identity = meta.get("run_identity", {}) if isinstance(meta, dict) else {}
    config = meta.get("config", {}) if isinstance(meta, dict) else {}
    protocol = meta.get("experiment_protocol", {}) if isinstance(meta, dict) else {}
    setting = str(
        identity.get("experiment_setting", "")
        or protocol.get("name", "")
    )
    seed = config.get("seed") if isinstance(config, dict) else None
    run_id_hash = _hash_json(identity or {"directory_name": run_dir.name})
    rows = []
    for agent_id, prompt in enumerate(prompts):
        normalized = normalize_prompt_text(prompt)
        marker_categories = mutable_prompt_violation_reasons(normalized)
        rows.append({
            "run_id_hash": run_id_hash,
            "setting": setting,
            "seed": seed,
            "agent_id": agent_id,
            "prompt_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "contaminated": bool(marker_categories),
            "marker_category": list(marker_categories),
        })
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "mutable_prompt_contract_version": MUTABLE_PROMPT_CONTRACT_VERSION,
        "run_id_hash": run_id_hash,
        "prompt_count": len(rows),
        "contaminated_prompt_count": sum(row["contaminated"] for row in rows),
        "prompt_records": rows,
    }


def discover_run_directories(inputs: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for value in inputs:
        path = value.resolve()
        if (path / "best_prompts.json").is_file():
            found.add(path)
        if path.is_dir():
            found.update(
                candidate.parent
                for candidate in path.rglob("best_prompts.json")
            )
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash-only audit of historical mutable prompt contamination."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    runs = [scan_run_directory(path) for path in discover_run_directories(args.inputs)]
    payload = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "mutable_prompt_contract_version": MUTABLE_PROMPT_CONTRACT_VERSION,
        "run_count": len(runs),
        "prompt_count": sum(row["prompt_count"] for row in runs),
        "contaminated_prompt_count": sum(
            row["contaminated_prompt_count"] for row in runs
        ),
        "runs": runs,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
