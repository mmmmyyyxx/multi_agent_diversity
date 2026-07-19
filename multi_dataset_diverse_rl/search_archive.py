"""Pure bounded-search helpers for the stable QD prompt archive."""

from typing import Any, Dict, Iterable, List, Sequence

from .mechanisms import (
    mechanism_distance,
    mechanism_niche_key,
    mechanisms_are_near_duplicate,
    normalize_mechanism_representation,
)


def _coverage_loss_count(metrics: Dict[str, Any], depth: int) -> int:
    loss_key = f"depth{depth}_loss_count"
    if loss_key in metrics:
        return max(0, int(metrics.get(loss_key, 0) or 0))
    net_key = f"depth{depth}_net_count"
    if net_key in metrics:
        return max(0, -int(metrics.get(net_key, 0) or 0))
    # Legacy fixtures used integer deltas as counts. Fractional deltas are rates
    # and must never be truncated into a false zero-loss classification.
    delta = float(metrics.get(f"depth{depth}_net_delta", 0.0) or 0.0)
    if delta >= 0.0:
        return 0
    if delta.is_integer():
        return -int(delta)
    denominator = int(metrics.get("num_eval_samples", metrics.get("actual_eval_batch_size", 0)) or 0)
    return max(1, int(round(-delta * denominator))) if denominator > 0 else 1


def _operation_sequence(item: Dict[str, Any]) -> List[str]:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
    proposal = item.get("proposal", {}) if isinstance(item.get("proposal", {}), dict) else {}
    representation = metrics.get("mechanism_representation", {})
    values = representation.get("normalized_operation_sequence")
    if values is None:
        raw_steps = proposal.get("mechanism_steps", metrics.get("mechanism_steps", []))
        values = normalize_mechanism_representation(str(item.get("prompt", "")), raw_steps)["normalized_operation_sequence"]
    return [str(value).strip().lower() for value in values if str(value).strip()] if isinstance(values, list) else []


def mechanism_is_novel(
    item: Dict[str, Any],
    parent: Dict[str, Any] | None = None,
    existing: Sequence[Dict[str, Any]] = (),
    *,
    near_duplicate_threshold: float = 0.97,
) -> bool:
    """Require an observed operation change that is not an existing niche duplicate."""
    representation = item.get("metrics", {}).get("mechanism_representation", {})
    sequence = _operation_sequence(item)
    family_kind = str(representation.get("family_kind", "unknown"))
    specificity = float(representation.get("specificity_score", 0.0) or 0.0)
    if not sequence and not (family_kind == "semantic" and specificity > 0.0):
        return False
    if parent is not None:
        parent_representation = parent.get("metrics", {}).get("mechanism_representation", {})
        if sequence and sequence == _operation_sequence(parent):
            return False
        if not sequence and str(representation.get("family_id", "")) == str(parent_representation.get("family_id", "")):
            return False
    return not any(
        mechanisms_are_near_duplicate(
            representation,
            other.get("metrics", {}).get("mechanism_representation", {}),
            near_duplicate_threshold,
        )
        for other in existing
        if other is not item
    )


def cheap_prescreen(
    candidate: Dict[str, Any],
    parent_prompt_hash: str,
    seen_hashes: Iterable[str],
    *,
    parent: Dict[str, Any] | None = None,
) -> List[str]:
    prompt = str(candidate.get("prompt", "")).strip()
    metrics = candidate.get("metrics", {}) if isinstance(candidate.get("metrics", {}), dict) else {}
    proposal = candidate.get("proposal", {}) if isinstance(candidate.get("proposal", {}), dict) else {}
    prompt_hash = str(candidate.get("prompt_hash", ""))
    reasons: List[str] = []
    if not prompt:
        reasons.append("empty_prompt")
    if prompt and prompt[-1] not in ".!?":
        reasons.append("incomplete_prompt")
    if bool(proposal.get("candidate_prompt_over_hard_limit", False)):
        reasons.append("prompt_over_hard_limit")
    if prompt_hash and prompt_hash == parent_prompt_hash:
        reasons.append("parent_duplicate")
    # Callers may retain normalized text during generation and hashes in the
    # archive. Accept either representation so a duplicate cannot slip through.
    seen = {str(value) for value in seen_hashes}
    prompt_signature = " ".join(prompt.lower().split())
    if prompt_hash and (prompt_hash in seen or prompt_signature in seen):
        reasons.append("duplicate_prompt")
    candidate_type = str(proposal.get("candidate_type", metrics.get("candidate_type", "")))
    if candidate_type not in {"task_specific_repair", "mechanism_alternative"}:
        reasons.append("invalid_candidate_type")
    steps = proposal.get("mechanism_steps", metrics.get("mechanism_steps", []))
    if not isinstance(steps, list) or not steps:
        reasons.append("missing_mechanism_steps")
    elif candidate_type == "mechanism_alternative" and not (
        _operation_sequence(candidate)
        or normalize_mechanism_representation(str(candidate.get("prompt", "")), steps).get("family_kind") == "semantic"
    ):
        reasons.append("missing_substantive_mechanism_operation")
    elif candidate_type == "mechanism_alternative" and parent is not None and _operation_sequence(candidate) == _operation_sequence(parent):
        reasons.append("mechanism_operation_unchanged")
    return reasons


def candidate_quality_bucket(item: Dict[str, Any], config: Any) -> str:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
    rejection = str(metrics.get("rejection_reason", ""))
    accuracy_loss = max(0.0, -float(metrics.get("accuracy_delta", 0.0) or 0.0))
    c1_loss = _coverage_loss_count(metrics, 1)
    c2_loss = _coverage_loss_count(metrics, 2)
    novelty = bool(metrics.get("mechanism_novel", False))
    candidate_type = str(metrics.get("candidate_type", ""))
    hard_guard_failed = not bool(metrics.get("hard_guard_passed", True))
    invalid_guard_failed = not bool(metrics.get("invalid_guard_passed", True))
    if rejection or hard_guard_failed or invalid_guard_failed or (candidate_type == "mechanism_alternative" and not novelty) or accuracy_loss > float(config.catastrophic_target_accuracy_loss_epsilon) or c1_loss >= int(config.candidate_c1_catastrophic_loss_questions) or c2_loss >= int(config.candidate_c2_catastrophic_loss_questions):
        return "catastrophic"
    if accuracy_loss == 0.0 and c1_loss == 0 and c2_loss == 0:
        return "safe"
    if (
        bool(config.probation_archive_enabled)
        and accuracy_loss <= float(config.probation_max_accuracy_loss)
        and c1_loss <= int(config.probation_max_c1_loss_questions)
        and c2_loss <= int(config.probation_max_c2_loss_questions)
        and (not bool(config.probation_require_mechanism_novelty) or novelty)
    ):
        return "probation"
    return "catastrophic"


def refill_requirements(items: Sequence[Dict[str, Any]], config: Any) -> Dict[str, Any]:
    safe = [item for item in items if item.get("archive_bucket") == "safe" and not bool(item.get("is_incumbent", False))]
    task_repair = [item for item in safe if str(item.get("metrics", {}).get("candidate_type", "")) == "task_specific_repair"]
    distinct = [item for item in safe if bool(item.get("metrics", {}).get("mechanism_novel", False))]
    missing = []
    if len(safe) < int(config.candidate_refill_min_safe_non_incumbent):
        missing.append("safe_non_incumbent")
    if bool(config.candidate_refill_require_task_repair) and not task_repair:
        missing.append("safe_task_specific_repair")
    if bool(config.candidate_refill_require_distinct_mechanism) and not distinct:
        missing.append("safe_distinct_mechanism")
    return {
        "safe_non_incumbent_count": len(safe),
        "safe_task_repair_count": len(task_repair),
        "safe_distinct_mechanism_count": len(distinct),
        "missing": missing,
        "met": not missing,
    }


def retained_archive_requirements(
    archive: Sequence[Dict[str, Any]], active_hash: str, config: Any,
) -> Dict[str, Any]:
    safe = [item for item in archive if str(item.get("prompt_hash", "")) != active_hash]
    niches = {mechanism_niche_key(item.get("metrics", {}).get("mechanism_representation", {})) for item in safe}
    task_repairs = [item for item in safe if str(item.get("metrics", {}).get("candidate_type", "")) == "task_specific_repair"]
    missing = []
    if len(safe) < int(config.candidate_refill_min_safe_non_incumbent):
        missing.append("retained_safe_non_incumbent_underfilled")
    if len(niches) < min(2, int(config.candidate_refill_min_safe_non_incumbent)):
        missing.append("retained_distinct_niche_underfilled")
    if bool(config.candidate_refill_require_task_repair) and not task_repairs:
        missing.append("retained_task_repair_missing")
    return {
        "retained_safe_count": len(safe),
        "retained_distinct_niche_count": len(niches),
        "retained_task_repair_count": len(task_repairs),
        "missing": missing,
        "met": not missing,
    }


def representative_requirements(
    representatives: Sequence[Dict[str, Any]], archive: Sequence[Dict[str, Any]], active_hash: str, config: Any,
) -> Dict[str, Any]:
    hashes = {str(item.get("prompt_hash", "")) for item in representatives}
    niches = {mechanism_niche_key(item.get("metrics", {}).get("mechanism_representation", {})) for item in representatives}
    expected = min(int(config.joint_representative_beam_size), len(archive))
    non_active = [item for item in representatives if str(item.get("prompt_hash", "")) != active_hash]
    complementary_available = any(item.get("metrics", {}).get("behavior_profile") for item in archive if str(item.get("prompt_hash", "")) != active_hash)
    complementary_retained = any(item.get("metrics", {}).get("behavior_profile") for item in non_active)
    missing = []
    if len(representatives) < expected:
        missing.append("representative_underfilled")
    if active_hash not in hashes:
        missing.append("representative_missing_active")
    if len(archive) > 1 and not non_active:
        missing.append("representative_missing_non_active")
    if complementary_available and not complementary_retained:
        missing.append("missing_behaviorally_distinct_representative")
    return {
        "representative_count": len(representatives),
        "representative_distinct_niche_count": len(niches),
        "representative_behavior_span": _representative_behavior_span(representatives, config),
        "missing": missing,
        "met": not missing,
    }


def search_space_requirements(
    evaluated: Sequence[Dict[str, Any]],
    archive: Sequence[Dict[str, Any]],
    representatives: Sequence[Dict[str, Any]],
    active_hash: str,
    config: Any,
) -> Dict[str, Any]:
    raw = refill_requirements(evaluated, config)
    retained = retained_archive_requirements(archive, active_hash, config)
    representative = representative_requirements(representatives, archive, active_hash, config)
    missing = list(dict.fromkeys([*raw["missing"], *retained["missing"], *representative["missing"]]))
    collision_count = max(0, int(raw["safe_non_incumbent_count"]) - int(retained["retained_safe_count"]))
    if collision_count and not retained["met"]:
        missing = list(dict.fromkeys(["archive_niche_collision", "safe_candidates_collapsed_after_archive", *missing]))
    return {
        **raw,
        **{key: value for key, value in retained.items() if key not in {"missing", "met"}},
        **{key: value for key, value in representative.items() if key not in {"missing", "met"}},
        "raw_requirements_met": bool(raw["met"]),
        "retained_requirements_met": bool(retained["met"]),
        "representative_requirements_met": bool(representative["met"]),
        "archive_collision_count": collision_count,
        "post_archive_refill_triggered": bool(raw["met"] and (not retained["met"] or not representative["met"])),
        "post_archive_refill_reason": ",".join([*retained["missing"], *representative["missing"]]),
        "missing": missing,
        "met": not missing,
    }


def select_safe_archive(items: Sequence[Dict[str, Any]], incumbent_hash: str, size: int) -> List[Dict[str, Any]]:
    safe = [item for item in items if item.get("archive_bucket") == "safe"]
    by_niche: Dict[tuple, Dict[str, Any]] = {}
    for item in safe:
        niche = mechanism_niche_key(item.get("metrics", {}).get("mechanism_representation", {}))
        current = by_niche.get(niche)
        key = (
            float(item.get("metrics", {}).get("candidate_target_accuracy", 0.0) or 0.0),
            float(item.get("metrics", {}).get("depth1_net_delta", 0.0) or 0.0),
            float(item.get("metrics", {}).get("depth2_net_delta", 0.0) or 0.0),
            float(item.get("metrics", {}).get("penalized_reward", item.get("reward", 0.0)) or 0.0),
            -int(item.get("generation", 0) or 0),
        )
        if current is None or key > current[0]:
            by_niche[niche] = (key, item)
    elites = [value[1] for value in by_niche.values()]
    incumbent = next((item for item in safe if str(item.get("prompt_hash", "")) == incumbent_hash), None)
    retained: List[Dict[str, Any]] = []
    if incumbent is not None:
        retained.append(incumbent)
    remaining = [item for item in elites if item not in retained]
    while remaining and len(retained) < int(size):
        def retention_key(row: Dict[str, Any]) -> tuple:
            quality = _archive_quality_key(row)
            if not retained:
                min_distance = 1.0
            else:
                min_distance = min(
                    mechanism_distance(
                        row.get("metrics", {}).get("mechanism_representation", {}),
                        kept.get("metrics", {}).get("mechanism_representation", {}),
                    )["mechanism_distance"]
                    for kept in retained
                )
            return (min_distance, quality)

        chosen = max(remaining, key=retention_key)
        retained.append(chosen)
        remaining.remove(chosen)
    return retained[: int(size)]


def _archive_quality_key(item: Dict[str, Any]) -> tuple:
    metrics = item.get("metrics", {})
    return (
        float(metrics.get("candidate_target_accuracy", 0.0) or 0.0),
        float(metrics.get("depth1_net_delta", 0.0) or 0.0),
        float(metrics.get("depth2_net_delta", 0.0) or 0.0),
        float(metrics.get("plurality_vote_gain_rate", metrics.get("vote_gain_rate", 0.0)) or 0.0),
        float(metrics.get("penalized_reward", item.get("reward", 0.0)) or 0.0),
        -int(item.get("generation", 0) or 0),
        str(item.get("prompt_hash", "")),
    )


def _representative_behavior_span(rows: Sequence[Dict[str, Any]], config: Any) -> float:
    from .behavior_profiles import behavior_distance

    distances = []
    for index, left in enumerate(rows):
        left_profile = left.get("metrics", {}).get("behavior_profile", {})
        for right in rows[index + 1:]:
            right_profile = right.get("metrics", {}).get("behavior_profile", {})
            if left_profile and right_profile:
                distances.append(behavior_distance(
                    left_profile, right_profile,
                    correct_set_weight=float(config.behavior_correct_set_weight),
                    rescue_weight=float(config.behavior_rescue_weight),
                    shared_wrong_weight=float(config.behavior_error_overlap_weight),
                    wrong_answer_dispersion_weight=float(config.behavior_wrong_answer_dispersion_weight),
                    support_shrinkage=float(config.behavior_support_shrinkage),
                    wrong_support_shrinkage=float(config.behavior_wrong_support_shrinkage),
                )["behavior_distance"])
    return max(distances, default=0.0)


def select_joint_representatives(
    archive: Sequence[Dict[str, Any]], active_hash: str, size: int, config: Any = None,
) -> List[Dict[str, Any]]:
    config = config or type("Defaults", (), {
        "behavior_correct_set_weight": 0.4, "behavior_rescue_weight": 0.3,
        "behavior_error_overlap_weight": 0.15, "behavior_wrong_answer_dispersion_weight": 0.15,
        "behavior_support_shrinkage": 5.0, "behavior_wrong_support_shrinkage": 5.0,
    })()
    active = next((item for item in archive if str(item.get("prompt_hash", "")) == active_hash), None)
    retained = [active] if active is not None else []
    remaining = [item for item in archive if item not in retained]
    if remaining and len(retained) < int(size):
        quality = max(remaining, key=_archive_quality_key)
        retained.append(quality)
        remaining.remove(quality)
    while remaining and len(retained) < int(size):
        def marginal_key(item: Dict[str, Any]) -> tuple:
            candidate_profile = item.get("metrics", {}).get("behavior_profile", {})
            behavior = []
            if candidate_profile:
                from .behavior_profiles import behavior_distance
                for kept in retained:
                    kept_profile = kept.get("metrics", {}).get("behavior_profile", {})
                    if kept_profile:
                        behavior.append(behavior_distance(
                            candidate_profile, kept_profile,
                            correct_set_weight=float(config.behavior_correct_set_weight),
                            rescue_weight=float(config.behavior_rescue_weight),
                            shared_wrong_weight=float(config.behavior_error_overlap_weight),
                            wrong_answer_dispersion_weight=float(config.behavior_wrong_answer_dispersion_weight),
                            support_shrinkage=float(config.behavior_support_shrinkage),
                            wrong_support_shrinkage=float(config.behavior_wrong_support_shrinkage),
                        )["behavior_distance"])
            mechanism = min((mechanism_distance(
                item.get("metrics", {}).get("mechanism_representation", {}),
                kept.get("metrics", {}).get("mechanism_representation", {}),
            )["mechanism_distance"] for kept in retained), default=0.0)
            return (min(behavior, default=-1.0), mechanism, _archive_quality_key(item))
        chosen = max(remaining, key=marginal_key)
        retained.append(chosen)
        remaining.remove(chosen)
    return retained[: int(size)]


def select_reproduction_parent(
    active: Dict[str, Any],
    safe_archive: Sequence[Dict[str, Any]],
    probation_archive: Sequence[Dict[str, Any]],
    parent_counts: Dict[str, int],
    *,
    epoch: int,
    min_opportunities: int,
    allow_probation: bool,
) -> tuple[Dict[str, Any] | None, str, str]:
    """Choose the second parent without letting active exploitation starve niches."""
    active_hash = str(active.get("prompt_hash", ""))

    def niche(item: Dict[str, Any]) -> str:
        return repr(mechanism_niche_key(item.get("metrics", {}).get("mechanism_representation", {})))

    def opportunity(item: Dict[str, Any]) -> int:
        return int(parent_counts.get(f"{epoch}:{niche(item)}", 0) or 0)

    if allow_probation:
        probation = sorted(
            list(probation_archive),
            key=lambda item: (opportunity(item), int(item.get("probation_created_update", 0) or 0), str(item.get("prompt_hash", ""))),
        )
        if probation and opportunity(probation[0]) < int(min_opportunities):
            return probation[0], "probation_niche", niche(probation[0])

    safe = [item for item in safe_archive if str(item.get("prompt_hash", "")) != active_hash]
    underused = [item for item in safe if opportunity(item) < int(min_opportunities)]
    if underused:
        chosen = min(underused, key=lambda item: (opportunity(item), str(item.get("prompt_hash", ""))))
        return chosen, "safe_niche", niche(chosen)
    if safe:
        active_representation = active.get("metrics", {}).get("mechanism_representation", {})
        chosen = max(
            safe,
            key=lambda item: (
                mechanism_distance(item.get("metrics", {}).get("mechanism_representation", {}), active_representation)["mechanism_distance"],
                -opportunity(item),
                str(item.get("prompt_hash", "")),
            ),
        )
        return chosen, "safe_niche", niche(chosen)
    return None, "active", ""
