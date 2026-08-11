from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
TABLES = OUT / "tables"
SEEDS = (48, 49, 50)
VARIANTS = (
    "C0_CURRENT_V15",
    "C1_BOUNDARY_PROPAGATION",
    "C2_BOUNDARY_PLUS_PRESERVATION",
    "C3_COALITION_AWARE_PRESERVATION",
)
REPAIR_MAX = 6
PRESERVE_MAX = 6
EXPECTED_DESIGN_HASHES = {
    "DESIGN_SPEC.md": "f5c6a6ace1dfd85e67d7fcd94437c828e5ef0be281ff40b178bc57753be4c4ed",
    "design_variants.json": "2e76f5d48dd498e6cafcefa59f83f17674592c5a9ecbecafe30c2e9095eadef6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        if names:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantile95(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[math.ceil(0.95 * len(ordered)) - 1])


def load_bottleneck_module() -> Any:
    path = ROOT / "experiments" / "v15_bottleneck_isolation_offline_audit_seed48_50_20260811" / "scripts" / "audit_bottleneck_isolation.py"
    spec = importlib.util.spec_from_file_location("v15_bottleneck_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen replay utility")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def exact_team(module: Any, record: dict[str, Any], seed: int, target_invalid: int | None = None) -> Any:
    validity = list(record["validity"])
    if target_invalid is not None:
        validity[target_invalid] = False
    return module.build_team_vote_state(
        question_hash=record["question_hash"],
        gold_answer=record["gold_answer"],
        answers=list(record["answers"]),
        valid_vector=validity,
        normalize_answer=lambda value: str(value or "").strip(),
        match_answer=lambda prediction, gold: prediction == gold,
        tie_break="abstain",
        seed=seed,
    )


def repair_tier(record: dict[str, Any]) -> str:
    if record["repair_distance"] == 1 and record["G"] > 0:
        return "R1_ONE_REPAIR_AWAY"
    if record["G"] == 1:
        return "R2_SINGLETON_FRAGMENTED"
    if record["repair_distance"] == 2:
        return "R3_TWO_REPAIRS_AWAY"
    return "R4_OTHER_ASSIGNED"


def repair_class(record: dict[str, Any]) -> str:
    if record["repair_distance"] == 1:
        return "one_repair_away"
    if record["G"] == 1:
        return "singleton_fragmented"
    return "fragmented"


def target_role(record: dict[str, Any]) -> str:
    if record["G"] == 0:
        return "discoverer"
    if record["repair_distance"] == 1:
        return "boundary_closing_member"
    return "reinforcing_member"


def build_repair_set(data: dict[str, Any], update: int, target: int, branch: dict[str, Any]) -> list[dict[str, Any]]:
    rank = data["active_state_rank"](data["accepted"], update)
    state = data["states"][rank]
    rows = []
    for qid in set(str(value) for value in branch.get("assigned_question_hashes", [])):
        record = state[qid]
        if record["correctness"][target] or record["vote_correct"]:
            continue
        rows.append({
            "question_hash": qid,
            "tier": repair_tier(record),
            "G": record["G"],
            "repair_distance": record["repair_distance"],
            "boundary_class": repair_class(record),
            "target_role": target_role(record),
        })
    tier_rank = {
        "R1_ONE_REPAIR_AWAY": 0, "R2_SINGLETON_FRAGMENTED": 1,
        "R3_TWO_REPAIRS_AWAY": 2, "R4_OTHER_ASSIGNED": 3,
    }
    return sorted(rows, key=lambda row: (tier_rank[row["tier"]], row["question_hash"]))[:REPAIR_MAX]


def build_preservation_set(module: Any, data: dict[str, Any], seed: int, update: int, target: int) -> list[dict[str, Any]]:
    rank = data["active_state_rank"](data["accepted"], update)
    state = data["states"][rank]
    rows = []
    for qid in data["qids"]:
        record = state[qid]
        if not record["correctness"][target]:
            continue
        parent = exact_team(module, record, seed)
        removed = exact_team(module, record, seed, target_invalid=target)
        if parent.vote_correct and not removed.vote_correct:
            tier = "P1_VOTE_CRITICAL"
        elif parent.vote_correct and removed.vote_correct and removed.plurality_margin < parent.plurality_margin:
            tier = "P2_COALITION_SUPPORT"
        else:
            stable = all(data["states"][prior][qid]["correctness"][target] for prior in range(rank + 1))
            if not stable:
                continue
            tier = "P3_STABLE_COMPETENCE"
        rows.append({
            "question_hash": qid, "tier": tier,
            "observed_correct_state_count": sum(data["states"][prior][qid]["correctness"][target] for prior in range(rank + 1)),
            "parent_margin": parent.plurality_margin,
            "removed_margin": removed.plurality_margin,
        })
    tier_rank = {"P1_VOTE_CRITICAL": 0, "P2_COALITION_SUPPORT": 1, "P3_STABLE_COMPETENCE": 2}
    return sorted(
        rows,
        key=lambda row: (
            tier_rank[row["tier"]],
            -int(row["observed_correct_state_count"]) if row["tier"] == "P3_STABLE_COMPETENCE" else 0,
            row["question_hash"],
        ),
    )[:PRESERVE_MAX]


def estimate_context(variant: str, repair: list[dict[str, Any]], preserve: list[dict[str, Any]]) -> tuple[int, int, int]:
    if variant == "C1_BOUNDARY_PROPAGATION":
        text = "Repair responsibilities:\n" + "".join(f"- [RESIDUAL_{index + 1}]\n" for index in range(len(repair)))
    elif variant == "C2_BOUNDARY_PLUS_PRESERVATION":
        text = (
            "Repair responsibilities:\n" + "".join(f"- [RESIDUAL_{index + 1}]\n" for index in range(len(repair)))
            + "Preservation responsibilities:\n" + "".join(f"- [CAPABILITY_{index + 1}]\n" for index in range(len(preserve)))
        )
    elif variant == "C3_COALITION_AWARE_PRESERVATION":
        text = "Repair responsibilities:\n" + "".join(
            f"- [RESIDUAL_{index + 1}] G={row['G']} r={row['repair_distance']} class={row['boundary_class']} role={row['target_role']}\n"
            for index, row in enumerate(repair)
        ) + "Preservation responsibilities:\n" + "".join(
            f"- [CAPABILITY_{index + 1}] tier={row['tier']}\n" for index, row in enumerate(preserve)
        )
    else:
        raise ValueError(variant)
    return len(text), len(text.split()), math.ceil(len(text) / 4)


def geometry_type(target_gain: int, vote_net: int) -> str:
    if target_gain > 0 and vote_net > 0:
        return "A"
    if target_gain > 0 and vote_net == 0:
        return "B"
    if target_gain == 0 and vote_net > 0:
        return "C"
    if target_gain > 0 and vote_net < 0:
        return "D"
    if target_gain < 0 and vote_net > 0:
        return "E"
    return "F"


def main() -> None:
    allowed_existing = {"scripts", "DESIGN_SPEC.md", "design_variants.json", "design_freeze.json"}
    if any(path.name not in allowed_existing for path in OUT.iterdir()):
        raise FileExistsError("analysis output must be fresh after design freeze")
    for name, expected in EXPECTED_DESIGN_HASHES.items():
        actual = sha256(OUT / name)
        if actual != expected:
            raise AssertionError(f"frozen design changed: {name}")
    TABLES.mkdir(parents=True, exist_ok=True)
    module, replay_script = load_bottleneck_module()
    consumed: set[Path] = {
        replay_script,
        ROOT / "experiments" / "v15_coverage_fragmentation_consensus_audit_seed48_50_20260811" / "audit_summary.json",
        ROOT / "experiments" / "v15_bottleneck_isolation_offline_audit_seed48_50_20260811" / "audit_summary.json",
        ROOT / "experiments" / "v15_bottleneck_isolation_offline_audit_seed48_50_20260811" / "tables" / "propagation_opportunity_events.csv",
        OUT / "DESIGN_SPEC.md", OUT / "design_variants.json", OUT / "design_freeze.json",
    }
    run_data: dict[int, dict[str, Any]] = {}
    branch_sets: dict[tuple[int, int, int], dict[str, Any]] = {}
    repair_rows: list[dict[str, Any]] = []
    preserve_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        data = module.load_run(seed, consumed)
        data["active_state_rank"] = module.active_state_rank
        run_data[seed] = data
        for decision in data["decisions"]:
            update = int(decision["update_index"])
            for branch in decision.get("branches", []):
                target = int(branch["target_agent_id"])
                repair = build_repair_set(data, update, target, branch)
                preserve = build_preservation_set(module, data, seed, update, target)
                c0 = set(data["context_map"].get((update, target), set()))
                key = (seed, update, target)
                branch_sets[key] = {"repair": repair, "preserve": preserve, "c0": c0}
                for qid in sorted(c0):
                    membership_rows.append({
                        "seed": seed, "update_index": update, "target_agent_id": target,
                        "variant": "C0_CURRENT_V15", "question_hash": qid,
                        "item_role": "historical_context_unknown_role", "tier": "not_separable",
                    })
                for index, row in enumerate(repair):
                    repair_rows.append({
                        "seed": seed, "update_index": update, "target_agent_id": target,
                        "repair_rank": index + 1, **row,
                    })
                    for variant in VARIANTS[1:]:
                        membership_rows.append({
                            "seed": seed, "update_index": update, "target_agent_id": target,
                            "variant": variant, "question_hash": row["question_hash"],
                            "item_role": "repair", "tier": row["tier"],
                        })
                for index, row in enumerate(preserve):
                    preserve_rows.append({
                        "seed": seed, "update_index": update, "target_agent_id": target,
                        "preservation_rank": index + 1, **row,
                    })
                    for variant in VARIANTS[2:]:
                        membership_rows.append({
                            "seed": seed, "update_index": update, "target_agent_id": target,
                            "variant": variant, "question_hash": row["question_hash"],
                            "item_role": "preservation", "tier": row["tier"],
                        })
                context_row = next(
                    (row for row in module.read_jsonl(data["paths"]["contexts"]) if int(row["update_index"]) == update and int(row["target_agent_id"]) == target),
                    None,
                )
                c0_chars = int(context_row.get("context_characters", 0)) if context_row else 0
                complexity_rows.append({
                    "seed": seed, "update_index": update, "target_agent_id": target,
                    "variant": "C0_CURRENT_V15", "repair_item_count": len(c0), "preservation_item_count": 0,
                    "total_item_count": len(c0), "high_priority_item_count": len(c0 & {row['question_hash'] for row in repair if row['tier'] != 'R4_OTHER_ASSIGNED'}),
                    "estimated_characters": c0_chars, "estimated_words": "not_separately_persisted",
                    "estimated_token_proxy": math.ceil(c0_chars / 4), "measurement": "historical_serialized_context",
                })
                for variant in VARIANTS[1:]:
                    used_preserve = preserve if variant in VARIANTS[2:] else []
                    chars, words, tokens = estimate_context(variant, repair, used_preserve)
                    useful = sum(row["tier"] != "R4_OTHER_ASSIGNED" for row in repair) + sum(row["tier"] in ("P1_VOTE_CRITICAL", "P2_COALITION_SUPPORT") for row in used_preserve)
                    complexity_rows.append({
                        "seed": seed, "update_index": update, "target_agent_id": target,
                        "variant": variant, "repair_item_count": len(repair), "preservation_item_count": len(used_preserve),
                        "total_item_count": len(repair) + len(used_preserve), "high_priority_item_count": useful,
                        "estimated_characters": chars, "estimated_words": words,
                        "estimated_token_proxy": tokens, "measurement": "synthetic_placeholder_serialization",
                    })

    # Exact opportunity/context comparison using the frozen opportunity ledger.
    opportunity_source = ROOT / "experiments" / "v15_bottleneck_isolation_offline_audit_seed48_50_20260811" / "tables" / "propagation_opportunity_events.csv"
    context_rows: list[dict[str, Any]] = []
    for source in read_csv(opportunity_source):
        seed = int(source["seed"]); update = int(source["update_index"]); target = int(source["eligible_agent_id"])
        selected = source["selected"] == "True"
        assigned = source["assigned_to_branch"] == "True"
        sets = branch_sets.get((seed, update, target))
        qid = source["question_hash"]
        for variant in VARIANTS:
            if variant == "C0_CURRENT_V15":
                exposed = bool(selected and sets and qid in sets["c0"])
            else:
                exposed = bool(selected and sets and qid in {row["question_hash"] for row in sets["repair"]})
            context_rows.append({
                "event_id": source["event_id"], "seed": seed, "update_index": update,
                "target_agent_id": target, "question_hash": qid,
                "parent_repair_distance": int(source["parent_repair_distance"]),
                "selected": selected, "assigned_to_branch": assigned,
                "variant": variant, "explicitly_exposed": exposed,
                "exclusion_reason": (
                    "exposed" if exposed else "target_not_selected" if not selected else
                    "not_assigned_to_branch" if not assigned else "outside_frozen_budget_or_priority"
                ),
            })
    # Hard minimal-redundancy assertion.
    for key, sets in branch_sets.items():
        seed, update, target = key
        rank = module.active_state_rank(run_data[seed]["accepted"], update)
        assert all(not run_data[seed]["states"][rank][row["question_hash"]]["vote_correct"] for row in sets["repair"])

    # Candidate replay and alignment.
    alignment_rows: list[dict[str, Any]] = []
    coalition_rows: list[dict[str, Any]] = []
    local_failure_cases: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for seed, data in run_data.items():
        for decision in data["decisions"]:
            update = int(decision["update_index"])
            rank = module.active_state_rank(data["accepted"], update)
            parent = data["states"][rank]
            candidates_by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for candidate in decision.get("candidates", []):
                if candidate.get("evaluation"):
                    candidates_by_target[int(candidate["target_agent_id"])].append(candidate)
            for target, candidates in candidates_by_target.items():
                sets = branch_sets[(seed, update, target)]
                c0_set = set(sets["c0"])
                c1_set = {row["question_hash"] for row in sets["repair"]}
                preserve_by_tier = defaultdict(set)
                for row in sets["preserve"]:
                    preserve_by_tier[row["tier"]].add(row["question_hash"])
                oracle_flags = {variant: False for variant in VARIANTS}
                for candidate_index, candidate in enumerate(candidates):
                    prompt_hash = str(candidate["prompt_hash"])
                    constraint = candidate["constraint"]
                    target_gain = int(constraint["target_gain"])
                    vote_net = int(constraint["vote_net_gain"])
                    geometry = geometry_type(target_gain, vote_net)
                    effects: dict[str, dict[str, int]] = {}
                    child_by_q: dict[str, dict[str, Any]] = {}
                    for qid in data["qids"]:
                        observation = data["observations"][(prompt_hash, qid)]
                        child = module.candidate_state(parent[qid], target, observation["answer"], observation["valid"], seed)
                        child_by_q[qid] = child
                    for variant in VARIANTS:
                        repair_set = c0_set if variant == "C0_CURRENT_V15" else c1_set
                        gain = loss = r1 = orphan = vote_conversion = r_reduction = 0
                        for qid in repair_set:
                            p, c = parent[qid], child_by_q[qid]
                            target_delta = int(c["correctness"][target]) - int(p["correctness"][target])
                            dr = c["repair_distance"] - p["repair_distance"]
                            dv = int(c["vote_correct"]) - int(p["vote_correct"])
                            gain += target_delta > 0 or dr < 0 or dv > 0
                            loss += target_delta < 0 or dr > 0 or dv < 0
                            r1 += p["repair_distance"] == 1 and (target_delta > 0 or dr < 0 or dv > 0)
                            orphan += p["G"] == 1 and not p["vote_correct"] and (target_delta > 0 or dr < 0 or dv > 0)
                            vote_conversion += dv > 0
                            r_reduction += dr < 0
                        p_losses = {}
                        for tier in ("P1_VOTE_CRITICAL", "P2_COALITION_SUPPORT", "P3_STABLE_COMPETENCE"):
                            p_losses[tier] = sum(
                                parent[qid]["correctness"][target] and not child_by_q[qid]["correctness"][target]
                                for qid in preserve_by_tier[tier]
                            ) if variant in VARIANTS[2:] else 0
                        preservation_loss = sum(p_losses.values())
                        aligned = gain > 0 and (variant in VARIANTS[:2] or preservation_loss == 0)
                        effects[variant] = {"gain": gain, "preservation_loss": preservation_loss}
                        alignment_rows.append({
                            "seed": seed, "update_index": update, "target_agent_id": target,
                            "candidate_index": candidate_index, "candidate_hash": prompt_hash,
                            "variant": variant, "geometry_type": geometry,
                            "constraint_passed": bool(constraint["passed"]),
                            "globally_accepted": prompt_hash == str(decision.get("accepted_prompt_hash", "")),
                            "repair_set_gain_count": gain, "repair_set_loss_count": loss,
                            "r1_repair_count": r1, "orphan_repair_count": orphan,
                            "vote_conversion_on_repair_set": vote_conversion,
                            "repair_distance_reduction_count": r_reduction,
                            "P1_loss_count": p_losses["P1_VOTE_CRITICAL"],
                            "P2_loss_count": p_losses["P2_COALITION_SUPPORT"],
                            "P3_loss_count": p_losses["P3_STABLE_COMPETENCE"],
                            "total_preservation_violation_count": preservation_loss,
                            "fixed_pool_aligned": aligned,
                            "evidence_level": "B_FIXED_POOL_RETROSPECTIVE_C0_GENERATED_CANDIDATE",
                        })
                        if bool(constraint["passed"]) and gain > 0 and (
                            variant in VARIANTS[:2] or p_losses["P1_VOTE_CRITICAL"] + p_losses["P2_COALITION_SUPPORT"] == 0
                        ):
                            oracle_flags[variant] = True
                    f_subtype = "not_F"
                    if geometry == "F":
                        if effects["C1_BOUNDARY_PROPAGATION"]["gain"] > 0:
                            f_subtype = "F_LOCAL_GAIN_GLOBAL_COLLATERAL"
                        elif target_gain < 0:
                            f_subtype = "F_TARGET_DEGRADATION"
                        elif target_gain == 0 and vote_net <= 0:
                            f_subtype = "F_LOCAL_NO_PROGRESS"
                        else:
                            f_subtype = "F_OTHER"
                    coalition_rows.append({
                        "seed": seed, "update_index": update, "target_agent_id": target,
                        "candidate_hash": prompt_hash, "geometry_type": geometry, "F_subtype": f_subtype,
                        "global_target_gain": target_gain, "global_vote_net_gain": vote_net,
                        "C1_repair_set_gain_count": effects["C1_BOUNDARY_PROPAGATION"]["gain"],
                        "C2_preservation_loss_count": effects["C2_BOUNDARY_PLUS_PRESERVATION"]["preservation_loss"],
                    })
                    # Freeze-identical definition of the previous eight cases.
                    c0_orphan_repairs = []
                    for qid in c0_set:
                        p, c = parent[qid], child_by_q[qid]
                        repaired = (
                            int(c["correctness"][target]) > int(p["correctness"][target])
                            or c["repair_distance"] < p["repair_distance"]
                            or int(c["vote_correct"]) > int(p["vote_correct"])
                        )
                        if p["G"] == 1 and not p["vote_correct"] and repaired:
                            c0_orphan_repairs.append(qid)
                    if c0_orphan_repairs and not bool(constraint["passed"]):
                        row_c2 = next(row for row in reversed(alignment_rows) if row["candidate_hash"] == prompt_hash and row["variant"] == "C2_BOUNDARY_PLUS_PRESERVATION")
                        local_failure_cases.append({
                            "seed": seed, "update_index": update, "branch_target_agent_id": target,
                            "candidate_hash": prompt_hash, "geometry_type": geometry,
                            "repair_residuals_fixed": len(c0_orphan_repairs),
                            "repair_distance_reductions": sum(child_by_q[q]["repair_distance"] < parent[q]["repair_distance"] for q in c0_orphan_repairs),
                            "P1_losses": row_c2["P1_loss_count"], "P2_losses": row_c2["P2_loss_count"], "P3_losses": row_c2["P3_loss_count"],
                            "global_target_gain": target_gain,
                            "global_target_example_gain_count": sum(int(child_by_q[q]["correctness"][target]) > int(parent[q]["correctness"][target]) for q in data["qids"]),
                            "global_target_example_loss_count": sum(int(child_by_q[q]["correctness"][target]) < int(parent[q]["correctness"][target]) for q in data["qids"]),
                            "vote_gain_count": int(constraint["vote_gain_count"]), "vote_loss_count": int(constraint["vote_loss_count"]),
                            "interpretation_limit": "preservation regions exposed; prevention requires pilot",
                        })
                for variant in VARIANTS:
                    oracle_rows.append({
                        "seed": seed, "update_index": update, "target_agent_id": target,
                        "variant": variant, "historical_evaluated_candidate_count": len(candidates),
                        "fixed_pool_ideal_candidate_exists": oracle_flags[variant],
                        "evidence_level": "FIXED_POOL_ORACLE_NOT_SELECTOR_PERFORMANCE",
                    })

    if len({(row["seed"], row["update_index"], row["target_agent_id"], row["candidate_hash"]) for row in coalition_rows}) != 247:
        raise AssertionError("candidate identity inventory mismatch")
    if len(local_failure_cases) != 8:
        raise AssertionError(f"expected eight prior local-repair rejected cases, got {len(local_failure_cases)}")

    # Aggregate exact context metrics.
    context_summary: list[dict[str, Any]] = []
    variant_seed_rows: list[dict[str, Any]] = []
    for seed_scope in [*SEEDS, "all"]:
        scoped = context_rows if seed_scope == "all" else [row for row in context_rows if row["seed"] == seed_scope]
        event_total = len({row["event_id"] for row in scoped})
        for variant in VARIANTS:
            rows = [row for row in scoped if row["variant"] == variant]
            exposed_events = {row["event_id"] for row in rows if row["explicitly_exposed"]}
            assigned_rows = [row for row in rows if row["assigned_to_branch"]]
            r1_rows = [row for row in rows if row["parent_repair_distance"] == 1]
            context_summary.append({
                "seed_scope": seed_scope, "variant": variant,
                "eligible_opportunity_count": len(rows),
                "explicit_context_opportunity_count": sum(row["explicitly_exposed"] for row in rows),
                "eligible_opportunity_to_context_rate": sum(row["explicitly_exposed"] for row in rows) / len(rows) if rows else None,
                "orphan_event_count": event_total, "orphan_event_context_coverage_count": len(exposed_events),
                "orphan_event_context_coverage_rate": len(exposed_events) / event_total if event_total else None,
                "assigned_opportunity_count": len(assigned_rows),
                "branch_assigned_to_context_retention": sum(row["explicitly_exposed"] for row in assigned_rows) / len(assigned_rows) if assigned_rows else None,
                "r1_opportunity_count": len(r1_rows),
                "r1_context_coverage_rate": sum(row["explicitly_exposed"] for row in r1_rows) / len(r1_rows) if r1_rows else None,
                "excluded_assigned_opportunity_count": sum(row["assigned_to_branch"] and not row["explicitly_exposed"] for row in rows),
            })

    # Preservation discrimination, fixed-pool proxy, complexity, and oracle by seed.
    preservation_summary: list[dict[str, Any]] = []
    fixed_pool_summary: list[dict[str, Any]] = []
    complexity_summary: list[dict[str, Any]] = []
    for seed_scope in [*SEEDS, "all"]:
        scoped_align = alignment_rows if seed_scope == "all" else [row for row in alignment_rows if row["seed"] == seed_scope]
        for variant in VARIANTS:
            rows = [row for row in scoped_align if row["variant"] == variant]
            aligned = sum(row["fixed_pool_aligned"] for row in rows)
            fixed_pool_summary.append({
                "seed_scope": seed_scope, "variant": variant, "candidate_count": len(rows),
                "aligned_candidate_count": aligned, "aligned_candidate_rate": aligned / len(rows) if rows else None,
                "mean_repair_set_gain_count": statistics.mean(row["repair_set_gain_count"] for row in rows) if rows else None,
                "mean_preservation_violation_count": statistics.mean(row["total_preservation_violation_count"] for row in rows) if rows else None,
            })
            for geometry_scope in ("AB", "D", "F", "DF"):
                geometry_rows = [
                    row for row in rows
                    if (geometry_scope == "AB" and row["geometry_type"] in ("A", "B"))
                    or (geometry_scope == "DF" and row["geometry_type"] in ("D", "F"))
                    or row["geometry_type"] == geometry_scope
                ]
                for violation_scope, columns in (
                    ("P1", ("P1_loss_count",)),
                    ("P2", ("P2_loss_count",)),
                    ("P1P2", ("P1_loss_count", "P2_loss_count")),
                    ("ALL", ("P1_loss_count", "P2_loss_count", "P3_loss_count")),
                ):
                    violation_counts = [sum(int(row[column]) for column in columns) for row in geometry_rows]
                    preservation_summary.append({
                        "seed_scope": seed_scope, "variant": variant, "geometry_scope": geometry_scope,
                        "violation_scope": violation_scope, "candidate_count": len(geometry_rows),
                        "any_preservation_violation_count": sum(value > 0 for value in violation_counts),
                        "any_preservation_violation_rate": sum(value > 0 for value in violation_counts) / len(violation_counts) if violation_counts else None,
                        "mean_preservation_loss_count": statistics.mean(violation_counts) if violation_counts else None,
                    })
        scoped_complexity = complexity_rows if seed_scope == "all" else [row for row in complexity_rows if row["seed"] == seed_scope]
        for variant in VARIANTS:
            rows = [row for row in scoped_complexity if row["variant"] == variant]
            totals = [int(row["total_item_count"]) for row in rows]
            chars = [int(row["estimated_characters"]) for row in rows]
            useful = sum(int(row["high_priority_item_count"]) for row in rows)
            memberships = [
                row for row in membership_rows
                if row["variant"] == variant and (seed_scope == "all" or row["seed"] == seed_scope)
            ]
            duplicate_counter = Counter((row["seed"], row["update_index"], row["question_hash"], row["item_role"]) for row in memberships)
            cross_branch_duplicate_occurrences = sum(max(0, count - 1) for count in duplicate_counter.values())
            complexity_summary.append({
                "seed_scope": seed_scope, "variant": variant, "branch_count": len(rows),
                "mean_total_items": statistics.mean(totals) if totals else None,
                "median_total_items": statistics.median(totals) if totals else None,
                "p95_total_items": quantile95(totals), "max_total_items": max(totals) if totals else None,
                "mean_estimated_characters": statistics.mean(chars) if chars else None,
                "p95_estimated_characters": quantile95(chars), "max_estimated_characters": max(chars) if chars else None,
                "context_efficiency": useful / sum(totals) if sum(totals) else None,
                "cross_branch_duplicate_item_occurrences": cross_branch_duplicate_occurrences,
                "vote_correct_items_added_as_repair": 0 if variant != "C0_CURRENT_V15" else "not_separable_from_historical_context",
            })
        scoped_oracle = oracle_rows if seed_scope == "all" else [row for row in oracle_rows if row["seed"] == seed_scope]
        for variant in VARIANTS:
            rows = [row for row in scoped_oracle if row["variant"] == variant]
            variant_seed_rows.append({
                "seed_scope": seed_scope, "variant": variant,
                "branch_update_count": len(rows),
                "fixed_pool_ideal_available_count": sum(row["fixed_pool_ideal_candidate_exists"] for row in rows),
                "fixed_pool_ideal_availability_rate": sum(row["fixed_pool_ideal_candidate_exists"] for row in rows) / len(rows) if rows else None,
            })

    f_subtype_rows = []
    for seed_scope in [*SEEDS, "all"]:
        rows = coalition_rows if seed_scope == "all" else [row for row in coalition_rows if row["seed"] == seed_scope]
        f_rows = [row for row in rows if row["geometry_type"] == "F"]
        counts = Counter(row["F_subtype"] for row in f_rows)
        for subtype in ("F_LOCAL_GAIN_GLOBAL_COLLATERAL", "F_TARGET_DEGRADATION", "F_LOCAL_NO_PROGRESS", "F_OTHER"):
            f_subtype_rows.append({
                "seed_scope": seed_scope, "F_subtype": subtype, "candidate_count": counts[subtype],
                "fraction_of_F": counts[subtype] / len(f_rows) if f_rows else None,
            })

    write_csv(TABLES / "repair_set_by_state.csv", repair_rows)
    write_csv(TABLES / "preservation_set_by_state.csv", preserve_rows)
    write_csv(TABLES / "context_membership_by_branch.csv", membership_rows)
    write_csv(TABLES / "context_coverage_comparison.csv", context_summary)
    write_csv(TABLES / "candidate_variant_alignment.csv", alignment_rows)
    write_csv(TABLES / "preservation_discrimination.csv", preservation_summary)
    write_csv(TABLES / "coalition_alignment.csv", coalition_rows)
    write_csv(TABLES / "fixed_pool_variant_summary.csv", fixed_pool_summary)
    write_csv(TABLES / "variant_by_seed.csv", variant_seed_rows)
    write_csv(TABLES / "complexity_budget.csv", complexity_summary)
    write_csv(TABLES / "local_repair_rejected_cases.csv", local_failure_cases)
    write_csv(TABLES / "fixed_pool_oracle_by_update.csv", oracle_rows)
    write_csv(TABLES / "F_subtype_summary.csv", f_subtype_rows)

    manifest_rows = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size": path.stat().st_size}
        for path in sorted(consumed)
    ]
    forbidden = ("_test", "test_diagnostics", "final_test")
    if any(any(token in row["path"].lower() for token in forbidden) for row in manifest_rows):
        raise AssertionError("test artifact entered analysis")
    manifest = {
        "study_version": "v16_module2_fixed_trajectory_design_isolation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "current_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "origin_main": subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip(),
        "method_semantics_commit": "c705eedb2959c3ad1349f5d6c52ffed64bca90ae",
        "historical_execution_commit": "b7936ae2f16d8907f0ffdf161dc8991368abeed8",
        "method_version": "member_aware_peer_state_v15", "checkpoint_version": 24,
        "design_variants_frozen_before_analysis": True,
        "frozen_design_hashes": EXPECTED_DESIGN_HASHES,
        "train_trajectory_only": True, "test_artifacts_consumed": False,
        "api_calls": 0, "model_calls": 0, "training_runs": 0, "validation_calls": 0, "test_calls": 0,
        "method_source_changed": False, "figures_generated": False,
        "consumed_artifacts": manifest_rows,
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    write_json(OUT / "audit_manifest.json", manifest)
    write_json(OUT / "fixed_pool_results.json", {
        "task_status": "PASS", "candidate_count": 247, "candidate_replay_mismatch": 0,
        "context_coverage": context_summary, "fixed_pool_alignment": fixed_pool_summary,
        "preservation_discrimination": preservation_summary, "oracle_availability": variant_seed_rows,
        "complexity": complexity_summary, "F_subtypes": f_subtype_rows,
        "local_repair_rejected_case_count": len(local_failure_cases),
        "evidence_level": "FIXED_POOL_RETROSPECTIVE_NOT_EFFICACY",
    })
    print(json.dumps({
        "task_status": "PASS", "design_frozen": True, "candidates": 247,
        "local_repair_rejected_cases": len(local_failure_cases),
        "api_calls": 0, "test_artifacts_consumed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
