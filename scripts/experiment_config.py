from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.protocol import (
    AUXILIARY_SEARCH_CONTROL_SETTINGS,
    LEGACY_CONTROL_SETTINGS,
    MAIN_ABLATION_SETTINGS,
    canonical_experiment_setting,
)


@dataclass(frozen=True)
class ExperimentSetting:
    name: str
    overrides: Mapping[str, Any]

    def resolved_overrides(self) -> dict[str, Any]:
        return {"experiment_setting": self.name, **dict(self.overrides)}


COMMON = {
    "method_version": "member_aware_peer_state_v13",
    "agents": 5,
    "initialization_mode": "shared_identical",
    "vote_tie_break": "abstain",
    "responsibility_mode": "single_service_member_aware_v13",
    "proposal_memory_mode": "off",
}

SETTING_NAMES = MAIN_ABLATION_SETTINGS

ALL_EXPERIMENT_SETTINGS = [ExperimentSetting(name, COMMON) for name in SETTING_NAMES]
LEGACY_EXPERIMENT_SETTINGS = [
    ExperimentSetting(name, {**COMMON, "allow_legacy_setting": True})
    for name in LEGACY_CONTROL_SETTINGS
]
AUXILIARY_EXPERIMENT_SETTINGS = [
    ExperimentSetting(name, {**COMMON, "allow_auxiliary_setting": True})
    for name in AUXILIARY_SEARCH_CONTROL_SETTINGS
]
DEFAULT_EXPERIMENT_SETTINGS = ALL_EXPERIMENT_SETTINGS
DEFAULT_EXPERIMENT_SETTING_NAMES = list(SETTING_NAMES)


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def select_settings(
    raw: str,
    settings: Iterable[ExperimentSetting] = ALL_EXPERIMENT_SETTINGS,
    *,
    allow_legacy_setting: bool = False,
    allow_auxiliary_setting: bool = False,
) -> list[ExperimentSetting]:
    available = {setting.name: setting for setting in settings}
    if allow_legacy_setting:
        available.update(
            {setting.name: setting for setting in LEGACY_EXPERIMENT_SETTINGS}
        )
    if allow_auxiliary_setting:
        available.update(
            {setting.name: setting for setting in AUXILIARY_EXPERIMENT_SETTINGS}
        )
    requested_names = list(available) if not raw or raw == "all" else parse_csv_list(raw)
    names = [
        canonical_experiment_setting(
            name,
            allow_legacy_setting=allow_legacy_setting,
            allow_auxiliary_setting=allow_auxiliary_setting,
        )
        for name in requested_names
    ]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown experiment setting: {missing}")
    return [available[name] for name in names]


def setting_names(settings: Iterable[ExperimentSetting] = ALL_EXPERIMENT_SETTINGS) -> list[str]:
    return [setting.name for setting in settings]


def resolved_config(setting: ExperimentSetting, **overrides: Any) -> Config:
    return Config.from_flat(**{**setting.resolved_overrides(), **overrides})
