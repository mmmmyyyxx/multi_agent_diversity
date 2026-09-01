from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "solver_headroom_multimodel_seed65_20260901"
REPORT_ROOT = ROOT / "reports" / "solver_headroom_multimodel_seed65_20260901"
OLD_ROOT = ROOT / "runs" / "solver_headroom_screening_20260901"
MANIFEST = ROOT / "configs" / "task_level_comparison_strict_bbh_seed42.yaml"
ROLE_MODEL = "qwen3.7-flash"
SEED = 65
CANDIDATES = (
    ("TURBO", "qwen-turbo"),
    ("Q4", "qwen3-4b"),
    ("FLASH", "qwen-flash"),
    ("Q8", "qwen3-8b"),
    ("DSQ7", "deepseek-r1-distill-qwen-7b"),
    ("DSL8", "deepseek-r1-distill-llama-8b"),
    ("DSQ15", "deepseek-r1-distill-qwen-1.5b"),
    ("GLMA", "glm-4.5-air"),
)
STATIC_SETTING = "shared_static_reference"
GENERIC_SETTING = "shared_generic_evolution"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def source_inventory() -> tuple[list[dict[str, str]], str]:
    paths = git("ls-files", "multi_dataset_diverse_rl", "scripts", "tests", "experiments/solver_headroom_multimodel_seed65_20260901").splitlines()
    rows, combined = [], hashlib.sha256()
    for relative in sorted(filter(None, paths)):
        normalized = relative.replace("\\", "/")
        digest = sha256_file(ROOT / relative)
        rows.append({"path": normalized, "sha256": digest})
        combined.update(normalized.encode() + b"\0" + digest.encode() + b"\n")
    return rows, combined.hexdigest()


def model_root(key: str) -> Path:
    return RUN_ROOT / "training" / f"model_{key}" / "seed65"


def run_dir(key: str, arm: str) -> Path:
    setting = STATIC_SETTING if arm == "STATIC" else GENERIC_SETTING
    if key == "Q8":
        return OLD_ROOT / "training" / "model_A" / "seed65" / "disambiguation_qa" / f"{setting}_seed65"
    return model_root(key) / "disambiguation_qa" / f"{setting}_seed65"


def validation_dir(key: str, arm: str) -> Path:
    return RUN_ROOT / "validation" / f"model_{key}" / arm


def candidates_by_key() -> dict[str, str]:
    return dict(CANDIDATES)


def phase_a_rows() -> list[dict[str, Any]]:
    return read_json(RUN_ROOT / "phase_a" / "availability_smoke_private.json")["candidates"]


def entrants() -> list[dict[str, Any]]:
    return [row for row in phase_a_rows() if row["static_eligible"]]


def selected_generic() -> list[dict[str, Any]]:
    return read_json(RUN_ROOT / "static_selection_private.json")["selected"]
