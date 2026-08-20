from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generic_m20_probe_support import state_hash, system_for


A = "A"
B = "B"
C = "C"
D = "D"
CELLS = (A, B, C, D)
CELL_NAMES = {
    A: "RR_GENERIC",
    B: "W1_GENERIC",
    C: "RR_MEMBER_AWARE",
    D: "W1_MEMBER_AWARE",
}
GENERIC_SETTING = "experimental_v17_formal_generic_2x2_matched"
MEMBER_AWARE_SETTING = "experimental_v16_efficacy_g_matched"
AUTHORIZATION_ENV = "V17_MODULE1_2X2_LOW_API_AUTHORIZED"
ALLOWED_CONCLUSIONS = (
    "TARGET_ALLOCATION_DOMINANT",
    "RESIDUAL_CONTEXT_DOMINANT",
    "BOTH_CONTRIBUTE",
    "NEGATIVE_INTERACTION_DOMINANT",
    "NO_CLEAR_LOCAL_CAUSAL_SOURCE",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def targets_for(case: dict[str, Any], cell: str) -> list[int]:
    if cell not in CELLS:
        raise ValueError(f"unknown cell: {cell}")
    key = "round_robin_target_ids" if cell in {A, C} else "w1_target_ids"
    targets = list(map(int, case[key]))
    if len(targets) != 2 or len(set(targets)) != 2:
        raise ValueError("every cell requires two distinct targets")
    return targets


def context_hashes(case: dict[str, Any], cell: str, target: int) -> set[str]:
    if cell in {A, B}:
        return set()
    hashes = set(map(str, case["active_residual_hashes_by_agent"][str(target)]))
    if not hashes:
        raise ValueError("member-aware target has no frozen active residual")
    return hashes


def probe_system(
    case: dict[str, Any], cell: str, *, out_dir: Path, cache_path: Path | str,
    target: int | None = None,
):
    setting = GENERIC_SETTING if cell in {A, B} else MEMBER_AWARE_SETTING
    resolved_target = int(
        targets_for(case, cell)[0] if target is None else target
    )
    local_case = dict(case)
    local_case["target_agent_id"] = resolved_target
    local_case["active_lane"] = case["active_lane_by_agent"][str(resolved_target)]
    system = system_for(
        local_case,
        setting=setting,
        out_dir=out_dir,
        cache_path=cache_path,
        evolution_variant="m20_current_v15",
    )
    # This is the historical V17 generic-evolution path. Its conditioned arm
    # is PeerStateDiagnosisContext, not the M20 SingleLaneDiagnosisContext.
    # The probe enables no compatibility policy, RCRU, or proposal memory.
    if system.protocol.compatibility_repair_enabled:
        raise AssertionError("compatibility repair is forbidden")
    if not system.protocol.generic_revision_enabled:
        raise AssertionError("loss-blind generic revision must be enabled")
    if system.cfg.tcs.proposal_memory_mode != "off":
        raise AssertionError("proposal memory must remain off")
    return system


def choose_would_commit(evaluator: Any, branch_winners: Iterable[Any]) -> Any | None:
    winners = [
        row for row in branch_winners
        if row is not None and getattr(row, "accepted", None) is not None
    ]
    return max(winners, key=evaluator._cross_branch_key, default=None)


def realized_delta(would_commit: bool, parent: int, hypothetical: int) -> int:
    return int(hypothetical - parent) if would_commit else 0


def _pair(value: dict[str, int]) -> tuple[int, int]:
    return int(value["vote"]), int(value["oracle"])


def _sub(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return left[0] - right[0], left[1] - right[1]


def _add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return left[0] + right[0], left[1] + right[1]


def _negative(value: tuple[int, int]) -> bool:
    return value[0] < 0 or (value[0] == 0 and value[1] < 0)


def _magnitude(value: tuple[int, int]) -> tuple[int, int]:
    return abs(value[0]), abs(value[1])


def classify(aggregate: dict[str, dict[str, int]]) -> dict[str, Any]:
    values = {cell: _pair(aggregate[cell]) for cell in CELLS}
    ba = _sub(values[B], values[A])
    dc = _sub(values[D], values[C])
    ca = _sub(values[C], values[A])
    db = _sub(values[D], values[B])
    target = _add(ba, dc)
    context = _add(ca, db)
    interaction = _sub(dc, ba)
    if (
        _negative(interaction)
        and _magnitude(interaction) > _magnitude(target)
        and _magnitude(interaction) > _magnitude(context)
    ):
        conclusion = "NEGATIVE_INTERACTION_DOMINANT"
    elif _negative(target) and _negative(context):
        conclusion = "BOTH_CONTRIBUTE"
    elif _negative(target):
        conclusion = "TARGET_ALLOCATION_DOMINANT"
    elif _negative(context):
        conclusion = "RESIDUAL_CONTEXT_DOMINANT"
    else:
        conclusion = "NO_CLEAR_LOCAL_CAUSAL_SOURCE"
    assert conclusion in ALLOWED_CONCLUSIONS
    return {
        "contrasts": {
            "B-A": {"vote": ba[0], "oracle": ba[1]},
            "D-C": {"vote": dc[0], "oracle": dc[1]},
            "C-A": {"vote": ca[0], "oracle": ca[1]},
            "D-B": {"vote": db[0], "oracle": db[1]},
            "interaction": {"vote": interaction[0], "oracle": interaction[1]},
            "target_main": {"vote": target[0], "oracle": target[1]},
            "context_main": {"vote": context[0], "oracle": context[1]},
        },
        "conclusion": conclusion,
    }


def immutable_state_hash(system: Any) -> str:
    return state_hash(system)
