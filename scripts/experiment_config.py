from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from multi_dataset_diverse_rl.config import Config


@dataclass(frozen=True)
class ExperimentSetting:
    name: str
    overrides: Mapping[str, Any]

    def resolved_overrides(self) -> dict[str, Any]:
        return {"experiment_setting": self.name, **dict(self.overrides)}


COMMON = {
    "method_version": "member_aware_peer_state_v2",
    "agents": 5,
    "initialization_mode": "shared_identical",
    "vote_tie_break": "abstain",
}

SETTING_NAMES = (
    "shared_baseline",
    "shared_independent_accuracy",
    "shared_peer_state_vote_first",
    "shared_peer_state_member_pareto",
    "shared_member_aware_responsibility",
    "shared_member_aware_full",
)

ALL_EXPERIMENT_SETTINGS = [ExperimentSetting(name, COMMON) for name in SETTING_NAMES]
DEFAULT_EXPERIMENT_SETTINGS = ALL_EXPERIMENT_SETTINGS
DEFAULT_EXPERIMENT_SETTING_NAMES = list(SETTING_NAMES)


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def select_settings(
    raw: str,
    settings: Iterable[ExperimentSetting] = ALL_EXPERIMENT_SETTINGS,
) -> list[ExperimentSetting]:
    available = {setting.name: setting for setting in settings}
    names = list(available) if not raw or raw == "all" else parse_csv_list(raw)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown experiment setting: {missing}")
    return [available[name] for name in names]


def setting_names(settings: Iterable[ExperimentSetting] = ALL_EXPERIMENT_SETTINGS) -> list[str]:
    return [setting.name for setting in settings]


def resolved_config(setting: ExperimentSetting, **overrides: Any) -> Config:
    return Config.from_flat(**{**setting.resolved_overrides(), **overrides})
