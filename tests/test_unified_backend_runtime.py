from __future__ import annotations

import asyncio
import ast
from pathlib import Path

import jsonschema
import pytest

from multi_dataset_diverse_rl.local_optimizers.backend_registry import (
    BackendRuntimeConfig,
    Layer1BackendRegistry,
    default_backend_registry,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalOptimizationResult,
    LocalPromptCandidate,
)
from multi_dataset_diverse_rl.native_feed import (
    Layer2OptimizationRequest,
    NativeOptimizationRequest,
)
from multi_dataset_diverse_rl.team_search.schemas import (
    TeamEvidenceCase,
    TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.run_record import UnifiedRunRecord
from multi_dataset_diverse_rl.team_search.task_builder import (
    Layer2EvidenceRequestBuilder,
    NativeFeedRequestBuilder,
    task_builder_for_optimization_mode,
)


ROOT = Path(__file__).parents[1]


def config(backend: str, mode: str) -> BackendRuntimeConfig:
    return BackendRuntimeConfig(
        optimizer_backend=backend,  # type: ignore[arg-type]
        optimization_mode=mode,  # type: ignore[arg-type]
        seed=80,
        solver_model="solver-fixture",
        optimizer_model="optimizer-fixture",
        budget="zero-provider-smoke-v1",
        data_split_manifest="optimize-only-fixture-v1",
        initial_state="state-fixture-v1",
    )


def assignment() -> TeamSearchAssignment:
    rows = tuple(
        TeamEvidenceCase(
            f"{group}-{index}", f"problem {group} {index}", "A", "B",
            "sanitized feedback", group,
            (("direct_flip",) if group == "responsibility" else ()),
        )
        for group in ("responsibility", "coalition", "preservation")
        for index in range(4)
    )
    return TeamSearchAssignment(
        2, "Use semantic evidence.", rows, "repair team residuals", "resp-v1",
        primary_responsibility_lane="direct_flip", responsibility_value=8.0,
    )


def outer() -> TeamSearchRequest:
    return TeamSearchRequest(
        80, 1, "team-state", 36, "COMMON_SOLVER_CONTRACT_V1", "output-v1",
        "optimize-only-fixture-v1",
    )


class NativeFake:
    async def optimize_native(self, request: NativeOptimizationRequest) -> LocalOptimizationResult:
        assert request.optimize_universe_id == "optimize-only-fixture-v1"
        return result("native")


class Layer2Fake:
    async def optimize_layer2(self, request: Layer2OptimizationRequest) -> LocalOptimizationResult:
        assert request.packet.responsibility_examples
        return result("layer2")


def result(mode: str) -> LocalOptimizationResult:
    return LocalOptimizationResult(
        candidates=(LocalPromptCandidate(f"{mode}-candidate", "Changed prompt.", 1.0, {}, (), 1),),
        backend_name="fixture", backend_version="v1", optimizer_state=None,
        solver_calls=0, optimizer_calls=0, input_tokens=0, output_tokens=0,
        total_tokens=0, termination_reason="complete",
    )


def fixture_registry() -> Layer1BackendRegistry:
    registry = Layer1BackendRegistry()
    for backend in ("gepa", "mars"):
        registry.register(
            backend=backend, mode="native", factory=NativeFake,
            fidelity=f"{backend}-native-fixture",
        )
        registry.register(
            backend=backend, mode="layer2", factory=Layer2Fake,
            fidelity=f"{backend}-layer2-fixture",
        )
    return registry


@pytest.mark.parametrize(
    ("backend", "mode", "expected_type"),
    [
        ("gepa", "native", NativeOptimizationRequest),
        ("gepa", "layer2", Layer2OptimizationRequest),
        ("mars", "native", NativeOptimizationRequest),
        ("mars", "layer2", Layer2OptimizationRequest),
    ],
)
def test_four_modes_dispatch_from_one_registry_and_config(
    backend: str, mode: str, expected_type: type
) -> None:
    cfg = config(backend, mode)
    builder = task_builder_for_optimization_mode(mode)
    problem = builder.build(outer(), assignment())
    assert isinstance(problem, expected_type)
    runtime = fixture_registry().create(cfg)
    outcome = asyncio.run(runtime.optimize(problem))
    assert runtime.optimizer_backend == backend
    assert runtime.optimization_mode == mode
    assert outcome.candidates[0].candidate_id == f"{mode}-candidate"


def test_default_registry_contains_exact_four_current_modes() -> None:
    assert default_backend_registry().registered_modes == (
        "GEPA_LAYER2", "GEPA_NATIVE", "MARS_LAYER2", "MARS_NATIVE"
    )


def test_layer2_packet_is_backend_independent() -> None:
    request = outer()
    frozen = assignment()
    gepa_packet = Layer2EvidenceRequestBuilder().build(request, frozen).packet
    mars_packet = Layer2EvidenceRequestBuilder().build(request, frozen).packet
    assert gepa_packet.identity_payload() == mars_packet.identity_payload()
    assert gepa_packet.packet_hash == mars_packet.packet_hash


def test_native_and_layer2_builders_are_distinct_data_flow_modes() -> None:
    native = NativeFeedRequestBuilder().build(outer(), assignment())
    layer2 = Layer2EvidenceRequestBuilder().build(outer(), assignment())
    assert isinstance(native, NativeOptimizationRequest)
    assert isinstance(layer2, Layer2OptimizationRequest)
    assert not hasattr(native, "packet")
    assert layer2.packet.local_eval_examples


def test_layer2_import_boundary_excludes_backend_search_implementations() -> None:
    checked = [ROOT / "multi_dataset_diverse_rl" / "native_feed.py"]
    checked.extend((ROOT / "multi_dataset_diverse_rl" / "team_search").glob("*.py"))
    forbidden = ("gepa_native", "gepa_optimizer", "mars_native")
    findings = []
    for path in checked:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else ""
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if any(name in (module or "") for name in forbidden) or any(
                any(value in name for value in forbidden) for name in names
            ):
                findings.append(path.relative_to(ROOT).as_posix())
    assert findings == []


def test_backends_do_not_import_layer2_scheduler_or_team_admission() -> None:
    findings = []
    for name in ("gepa_native.py", "mars_native.py"):
        path = ROOT / "multi_dataset_diverse_rl" / "local_optimizers" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else ""
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if "team_search" in (module or "") or any("team_search" in item for item in names):
                findings.append(name)
    assert findings == []


def test_common_run_schema_requires_backend_details_only_as_extension() -> None:
    schema = __import__("json").loads(
        (ROOT / "infrastructure" / "unified_backend_run.schema.json").read_text(encoding="utf-8")
    )
    row = {
        "code_sha": "0" * 40,
        "config_hash": "1" * 64,
        "backend": "gepa",
        "optimization_mode": "layer2",
        "backend_fidelity": "fixture",
        "initial_state": {},
        "optimization_opportunities": [],
        "candidate_funnel": {},
        "team_evaluation_funnel": {},
        "commits": [],
        "vote_trajectory": [],
        "oracle_trajectory": [],
        "member_accuracy_trajectory": [],
        "pivotality_trajectory": [],
        "categorical_profile_hashes": [],
        "cost_accounting": {},
        "final_state": {},
        "backend_details": {"backend_specific": True},
    }
    jsonschema.validate(row, schema)
    record = UnifiedRunRecord(
        code_sha=row["code_sha"], config_hash=row["config_hash"],
        backend=row["backend"], optimization_mode=row["optimization_mode"],
        backend_fidelity=row["backend_fidelity"], initial_state={},
        optimization_opportunities=(), candidate_funnel={},
        team_evaluation_funnel={}, commits=(), vote_trajectory=(),
        oracle_trajectory=(), member_accuracy_trajectory=(),
        pivotality_trajectory=(), categorical_profile_hashes=(),
        cost_accounting={}, final_state={}, backend_details=row["backend_details"],
    )
    assert record.sanitized() == row
