"""Pure bounded-search helpers for the stable QD prompt archive."""

from typing import Any, Dict, Iterable, List, Sequence

from .mechanisms import mechanism_distance, mechanism_niche_key, mechanisms_are_near_duplicate


def _operation_sequence(item: Dict[str, Any]) -> List[str]:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
    proposal = item.get("proposal", {}) if isinstance(item.get("proposal", {}), dict) else {}
    representation = metrics.get("mechanism_representation", {})
    values = representation.get(
        "normalized_operation_sequence",
        proposal.get("mechanism_steps", metrics.get("mechanism_steps", [])),
    )
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
    if not sequence:
        return False
    if parent is not None and sequence == _operation_sequence(parent):
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
    elif candidate_type == "mechanism_alternative" and parent is not None and _operation_sequence(candidate) == _operation_sequence(parent):
        reasons.append("mechanism_operation_unchanged")
    return reasons


def candidate_quality_bucket(item: Dict[str, Any], config: Any) -> str:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
    rejection = str(metrics.get("rejection_reason", ""))
    accuracy_loss = max(0.0, -float(metrics.get("accuracy_delta", 0.0) or 0.0))
    c1_loss = max(0, -int(metrics.get("depth1_net_delta", 0) or 0))
    c2_loss = max(0, -int(metrics.get("depth2_net_delta", 0) or 0))
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


def select_joint_representatives(archive: Sequence[Dict[str, Any]], active_hash: str, size: int) -> List[Dict[str, Any]]:
    active = next((item for item in archive if str(item.get("prompt_hash", "")) == active_hash), None)
    retained = [active] if active is not None else []
    for item in archive:
        if item in retained:
            continue
        if not retained:
            retained.append(item)
        else:
            if not any(mechanisms_are_near_duplicate(
                item.get("metrics", {}).get("mechanism_representation", {}),
                kept.get("metrics", {}).get("mechanism_representation", {}),
            ) for kept in retained):
                retained.append(item)
        if len(retained) >= int(size):
            break
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
