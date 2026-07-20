from multi_dataset_diverse_rl.evaluation.solver_service import is_transient_llm_error


def test_provider_429_and_localized_saturation_are_transient():
    error = RuntimeError(
        "Error code: 429 - {'error': {'message': '当前分组上游负载已饱和，请稍后再试'}}"
    )
    assert is_transient_llm_error(error) is True


def test_permanent_request_error_is_not_transient():
    assert is_transient_llm_error(RuntimeError("invalid model parameter")) is False
