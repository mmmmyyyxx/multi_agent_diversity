"""Resume-safe candidate channel funnel accounting."""

from typing import Any, Dict, Mapping, MutableMapping, MutableSet


CANDIDATE_CHANNELS = (
    "teacher_critic_student",
    "open_mechanism_exploration",
    "incumbent",
    "other",
)
CANDIDATE_FUNNEL_STAGES = (
    "generation_call_count",
    "raw_candidate_count",
    "schema_valid_candidate_count",
    "prescreen_pass_count",
    "evaluated_candidate_count",
    "safe_count",
    "probation_count",
    "catastrophic_count",
    "archive_retained_count",
    "representative_selected_count",
    "active_selected_count",
)


def empty_candidate_channel_funnel() -> Dict[str, Dict[str, int]]:
    return {
        channel: {stage: 0 for stage in CANDIDATE_FUNNEL_STAGES}
        for channel in CANDIDATE_CHANNELS
    }


def normalize_candidate_channel(item: Mapping[str, Any]) -> str:
    source = str(item.get("candidate_source", item.get("source", "")) or "").lower()
    architecture = str(item.get("optimizer_architecture", "") or "").lower()
    if source == "teacher_critic_student" or architecture == "teacher_critic_student":
        return "teacher_critic_student"
    if source == "open_mechanism_exploration" or architecture == "open_mechanism_exploration":
        return "open_mechanism_exploration"
    pool_source = str(item.get("candidate_pool_source", "") or "").lower()
    beam_slot = str(item.get("beam_slot", item.get("metrics", {}).get("beam_slot", "")) or "").lower()
    if pool_source in {"existing_beam", "current_active_fallback"} or source in {
        "existing_beam", "current_active_fallback", "incumbent",
    } or beam_slot == "incumbent":
        return "incumbent"
    return "other"


def candidate_funnel_identity(item: Mapping[str, Any], agent_id: int) -> str:
    prompt_hash = str(item.get("prompt_hash", "") or item.get("prompt", "") or "")
    generation = int(item.get("generation", 0) or 0)
    return f"a{int(agent_id)}:g{generation}:p{prompt_hash}"


def record_funnel_event(
    funnel: MutableMapping[str, Dict[str, int]],
    seen: MutableMapping[str, MutableSet[str]],
    *,
    channel: str,
    stage: str,
    identity: str,
    amount: int = 1,
) -> bool:
    channel = channel if channel in CANDIDATE_CHANNELS else "other"
    if stage not in CANDIDATE_FUNNEL_STAGES:
        raise ValueError(f"unknown candidate funnel stage: {stage}")
    funnel.setdefault(channel, {name: 0 for name in CANDIDATE_FUNNEL_STAGES})
    seen_key = f"{channel}:{stage}"
    bucket = seen.setdefault(seen_key, set())
    if identity in bucket:
        return False
    bucket.add(identity)
    funnel[channel][stage] = int(funnel[channel].get(stage, 0) or 0) + max(0, int(amount))
    return True


def record_candidate_stage(
    funnel: MutableMapping[str, Dict[str, int]],
    seen: MutableMapping[str, MutableSet[str]],
    item: Mapping[str, Any],
    *,
    agent_id: int,
    stage: str,
) -> bool:
    return record_funnel_event(
        funnel,
        seen,
        channel=normalize_candidate_channel(item),
        stage=stage,
        identity=candidate_funnel_identity(item, agent_id),
    )


def record_candidate_classification(
    funnel: MutableMapping[str, Dict[str, int]],
    seen: MutableMapping[str, MutableSet[str]],
    item: Mapping[str, Any],
    *,
    agent_id: int,
    stage: str,
) -> bool:
    if stage not in {"safe_count", "probation_count", "catastrophic_count"}:
        raise ValueError(f"invalid candidate classification stage: {stage}")
    channel = normalize_candidate_channel(item)
    identity = candidate_funnel_identity(item, agent_id)
    classification_seen = seen.setdefault(f"{channel}:classification", set())
    if identity in classification_seen:
        return False
    classification_seen.add(identity)
    return record_funnel_event(
        funnel, seen, channel=channel, stage=stage, identity=identity,
    )


def serialize_funnel_seen(seen: Mapping[str, Any]) -> Dict[str, list[str]]:
    return {str(key): sorted(str(value) for value in values) for key, values in seen.items()}


def restore_funnel_seen(payload: Mapping[str, Any]) -> Dict[str, set[str]]:
    return {
        str(key): {str(value) for value in values}
        for key, values in payload.items()
        if isinstance(values, list)
    }


def validate_candidate_channel_funnel(funnel: Mapping[str, Mapping[str, int]]) -> None:
    for channel in CANDIDATE_CHANNELS:
        counts = funnel.get(channel, {})
        evaluated = int(counts.get("evaluated_candidate_count", 0) or 0)
        classified = sum(int(counts.get(stage, 0) or 0) for stage in (
            "safe_count", "probation_count", "catastrophic_count",
        ))
        assert classified <= evaluated
        assert int(counts.get("archive_retained_count", 0) or 0) <= int(counts.get("safe_count", 0) or 0)
        if channel != "incumbent":
            assert int(counts.get("representative_selected_count", 0) or 0) <= int(counts.get("archive_retained_count", 0) or 0)
        assert int(counts.get("active_selected_count", 0) or 0) <= int(counts.get("representative_selected_count", 0) or 0)
