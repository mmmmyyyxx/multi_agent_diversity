"""Public orchestration facade for multi-agent prompt search."""

from .system_shared import *
from .optimization.lifecycle import LifecycleMixin
from .persistence.runtime_state import RuntimeStateMixin
from .optimization.candidate_schema import CandidateSchemaMixin
from .evaluation.solver_service import SolverServiceMixin
from .metrics.rollout_metrics import RolloutMetricsMixin
from .optimization.target_selector import TargetSelectorMixin
from .optimization.candidate_generator import CandidateGeneratorMixin
from .evaluation.candidate_evaluator import CandidateEvaluatorMixin
from .optimization.prompt_update_controller import PromptUpdateMixin
from .optimization.training_controller import TrainingControllerMixin
from .qd.joint_controller import JointControllerMixin
from .evaluation.dataset_evaluator import DatasetEvaluatorMixin
from .persistence.artifact_methods import ArtifactMethodsMixin


class TraceBeamSearchSystem(
    LifecycleMixin,
    RuntimeStateMixin,
    CandidateSchemaMixin,
    SolverServiceMixin,
    RolloutMetricsMixin,
    TargetSelectorMixin,
    CandidateGeneratorMixin,
    CandidateEvaluatorMixin,
    PromptUpdateMixin,
    TrainingControllerMixin,
    JointControllerMixin,
    DatasetEvaluatorMixin,
    ArtifactMethodsMixin,
):
    GENERIC_DISTINCT_PROCEDURE = (
        "Use a distinct decision procedure: first state which reasoning route you will use, "
        "then approach the problem through boundary checks, reverse validation, or an alternative representation. "
        "If that procedure is not useful, fall back to direct reasoning with one explicit verification step."
    )


TextualGradientRLSystem = TraceBeamSearchSystem
