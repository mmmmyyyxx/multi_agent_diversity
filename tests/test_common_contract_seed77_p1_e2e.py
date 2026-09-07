from __future__ import annotations

import asyncio

import pytest

from infrastructure.common_solver_contract_v1.evaluator import (
    CommonSolverEvaluator,
    TransportResponse,
)
from infrastructure.common_solver_contract_v1.system_adapter import CommonContractSolverAdapter
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.identity import solver_request_components
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.versions import COMMON_SOLVER_CONTRACT_V1_ID
from scripts.run_common_contract_seed77_p1_e2e import classify


QUESTION = "Question?\nOptions:\n(A) First\n(B) Second"


def test_common_contract_config_enters_identity_and_requires_adapter() -> None:
    cfg = Config.from_flat(
        solver_contract_id=COMMON_SOLVER_CONTRACT_V1_ID,
        solver_invalid_max_retries=0,
    )
    identity = solver_request_components(cfg)
    assert identity["solver_contract_id"] == COMMON_SOLVER_CONTRACT_V1_ID
    assert identity["provider_seed_policy"] == "OMITTED"
    assert identity["invalid_retry_policy"] == "SCORE_WRONG_NO_SEMANTIC_RETRY"
    with pytest.raises(ValueError, match="shared solver adapter"):
        PromptEnsembleOptimizationSystem(cfg)


def test_common_adapter_returns_strict_prompt_answer_and_accounting() -> None:
    async def transport(request: dict) -> TransportResponse:
        assert "seed" not in request
        return TransportResponse(
            text="FINAL_ANSWER: A",
            prompt_tokens=10,
            completion_tokens=2,
            finish_reason="stop",
        )

    evaluator = CommonSolverEvaluator(transport=transport)
    adapter = CommonContractSolverAdapter(evaluator)
    answer = asyncio.run(adapter.solve(QUESTION, 4, "Solve carefully."))
    assert answer.valid is True
    assert answer.answer == "A"
    assert answer.prompt_tokens == 10
    assert answer.completion_tokens == 2
    assert adapter.accounting()["successful_provider_calls"] == 1


@pytest.mark.parametrize(
    ("p0", "final", "expected"),
    [
        (
            {"mean_member_accuracy": 0.62, "vote_accuracy": 0.62, "oracle_accuracy": 0.62},
            {"mean_member_accuracy": 0.66, "vote_accuracy": 0.64, "oracle_accuracy": 0.80},
            "CONTRACT_TRANSFER_FAILURE_DOMINANT",
        ),
        (
            {"mean_member_accuracy": 0.62, "vote_accuracy": 0.62, "oracle_accuracy": 0.62},
            {"mean_member_accuracy": 0.58, "vote_accuracy": 0.60, "oracle_accuracy": 0.86},
            "COMPETENCE_REDISTRIBUTION_BIAS_SUPPORTED",
        ),
        (
            {"mean_member_accuracy": 0.62, "vote_accuracy": 0.62, "oracle_accuracy": 0.62},
            {"mean_member_accuracy": 0.64, "vote_accuracy": 0.60, "oracle_accuracy": 0.80},
            "MIXED_CONTRACT_AND_OPTIMIZATION_EFFECTS",
        ),
    ],
)
def test_classifier_is_frozen(p0: dict, final: dict, expected: str) -> None:
    assert classify(p0, final) == expected
