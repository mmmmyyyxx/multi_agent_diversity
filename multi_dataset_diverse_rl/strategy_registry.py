from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Strategy(Protocol):
    name: str


@dataclass(frozen=True)
class NamedStrategy:
    name: str


@dataclass(frozen=True)
class SearchPolicyBundle:
    target_selector: NamedStrategy
    candidate_selector: NamedStrategy
    archive_policy: NamedStrategy
    joint_selector: NamedStrategy
    lineage_policy: NamedStrategy


def build_policy_bundle(config) -> SearchPolicyBundle:
    stable = str(config.method_version) == "v8_stable_qd_lineage"
    return SearchPolicyBundle(
        target_selector=NamedStrategy(str(config.target_selector_version or config.target_selector_mode)),
        candidate_selector=NamedStrategy(str(config.candidate_selection_mode)),
        archive_policy=NamedStrategy(str(config.archive_policy_version) if stable else "none"),
        joint_selector=NamedStrategy(str(config.active_team_selector_version) if stable else "none"),
        lineage_policy=NamedStrategy(str(config.lineage_policy_version) if stable else "none"),
    )
