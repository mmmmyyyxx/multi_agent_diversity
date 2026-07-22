from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from multi_dataset_diverse_rl.config import Config


@dataclass(frozen=True)
class ExperimentSetting:
    name: str
    overrides: Mapping[str, Any]

    def resolved_overrides(self) -> dict[str, Any]:
        return dict(self.overrides)


COMMON = {
    "method_version": "peer_state_counterfactual_v1",
    "agents": 5,
}

ALL_EXPERIMENT_SETTINGS = [
    ExperimentSetting("shared_baseline", {**COMMON, "baseline_only": True}),
    ExperimentSetting("shared_independent_accuracy_tcs", {
        **COMMON, "independent_accuracy_only": True, "target_selector": "round_robin",
        "responsibility_assignment_enabled": False, "responsibility_conditioned_tcs": False,
        "online_responsibility_refresh": False, "responsibility_inertia_enabled": False,
    }),
    ExperimentSetting("shared_peer_state_credit_round_robin", {
        **COMMON, "target_selector": "round_robin", "responsibility_assignment_enabled": False,
        "responsibility_conditioned_tcs": False, "online_responsibility_refresh": False,
        "responsibility_inertia_enabled": False,
    }),
    ExperimentSetting("shared_peer_state_responsibility", {
        **COMMON, "target_selector": "residual_responsibility", "responsibility_assignment_enabled": True,
        "responsibility_conditioned_tcs": False, "online_responsibility_refresh": False,
        "responsibility_inertia_enabled": True,
    }),
    ExperimentSetting("shared_peer_state_full", {
        **COMMON, "target_selector": "residual_responsibility", "responsibility_assignment_enabled": True,
        "responsibility_conditioned_tcs": True, "online_responsibility_refresh": True,
        "responsibility_inertia_enabled": True,
    }),
]
DEFAULT_EXPERIMENT_SETTINGS = ALL_EXPERIMENT_SETTINGS
DEFAULT_EXPERIMENT_SETTING_NAMES = [setting.name for setting in ALL_EXPERIMENT_SETTINGS]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def select_settings(raw: str, settings: Iterable[ExperimentSetting] = ALL_EXPERIMENT_SETTINGS) -> list[ExperimentSetting]:
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
