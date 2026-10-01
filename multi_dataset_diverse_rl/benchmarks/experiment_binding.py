"""Explicit global models for current experiments, independent of CLI defaults."""
from dataclasses import asdict, dataclass, replace
from .. import versions
from ..search.schemas import SearchContractError
from .data_freeze import digest


@dataclass(frozen=True)
class ExperimentModelBinding:
    identity: str = versions.MULTIBENCH_MODEL_BINDING_VERSION
    solver: str = versions.CURRENT_EXPERIMENT_SOLVER_MODEL
    optimizer: str = versions.CURRENT_EXPERIMENT_OPTIMIZER_MODEL
    pattern: str = versions.CURRENT_EXPERIMENT_PATTERN_MODEL
    agents: int = 5
    solver_thinking: bool = False

    def require(self, benchmark: str, arm: str, seed: int) -> None:
        if (benchmark not in versions.CURRENT_RESEARCH_BENCHMARK_SUITE
                or arm not in ("A1", "A2", "A3", "A4") or type(seed) is not int
                or self != ExperimentModelBinding()):
            raise SearchContractError("CURRENT_EXPERIMENT_MODEL_BINDING_MISMATCH")

    def sha256(self) -> str:
        return digest(asdict(self))

    def bind_config(self, config, *, benchmark: str, arm: str, seed: int):
        self.require(benchmark, arm, seed)
        # This explicit boundary overrides compatibility defaults. It creates
        # no provider, prompt, authorization or real-execution composition.
        return replace(config, models=replace(config.models, agent_model=self.solver,
            optimizer_model=self.optimizer), training=replace(config.training,
            agents=self.agents, seed=seed))
