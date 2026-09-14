"""Reflection-evidence-fixed successor to the Level-B GEPA path canary."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_level_b_gepa_real_canary as base  # noqa: E402


base.EXPERIMENT_ID = "level_b_gepa_real_canary_v2"
base.ATTEMPT_ID = "level_b_gepa_real_canary_v2_reflectionfix1_pending_authorization"
base.MANIFEST = ROOT / "experiments/manifests/level_b_gepa_real_canary_v2.yaml"
base.PROTOCOL = ROOT / "experiments/level_b_gepa_real_canary_v2/PROTOCOL.md"
base.DEFAULT_PREP = ROOT / "runs/level_b_gepa_real_canary_v2_prep_reflectionfix1"
base.DEFAULT_RUN = ROOT / "runs/level_b_gepa_real_canary_v2_reflectionfix1"
base.DEFAULT_REPORT = ROOT / "reports/level_b_gepa_real_canary_v2_reflectionfix1"
base.AUTH_ENV = "LEVEL_B_GEPA_REAL_CANARY_V2_AUTHORIZED"

_base_source_paths = base.source_paths
_base_protocol_document = base.protocol_document


def source_paths() -> list[Path]:
    return sorted(
        set(_base_source_paths())
        | {Path("scripts/run_level_b_gepa_real_canary_v2.py")},
        key=lambda path: path.as_posix(),
    )


def protocol_document():
    protocol = _base_protocol_document()
    protocol.update(
        {
            "schema_version": "level_b_gepa_real_canary_protocol_v3",
            "reflection_input_repair": (
                "component_specific_reasoning_evidence_v1"
            ),
            "minimum_technical_success": (
                "contract_valid_changed_proposal_reaches_solver_gt_zero"
            ),
        }
    )
    return protocol


base.source_paths = source_paths
base.protocol_document = protocol_document


if __name__ == "__main__":
    base.main()
