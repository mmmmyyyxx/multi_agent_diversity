"""MATH V1.1 repairs and actual pinned-public-GEPA four-arm conformance."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace, asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.benchmarks.access import DataPurpose
from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
from multi_dataset_diverse_rl.benchmarks.math_execution import MATHExecutionBinding, provider_bounds
from multi_dataset_diverse_rl.search.benchmark import BenchmarkInput
from multi_dataset_diverse_rl.search.binary_composition import build_binary_orchestrator
from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample, FixedPeerPromotion, FixedPeerCommonSafe
from multi_dataset_diverse_rl.search.gepa_v2 import GEPATeamExposureOptimizer
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker, BenchmarkSolver, ReflectionProvider, PatternProvider
from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
from multi_dataset_diverse_rl.search.memory import StructuredLongTermMemoryProviderV1, OpportunityOutcome
from multi_dataset_diverse_rl.search.schemas import SearchContractError, EvaluatedCandidate, SearchCandidate
from multi_dataset_diverse_rl.shadow_gate import ShadowGateMetrics, evaluate_configured_shadow_gate, evaluate_shadow_gate
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import import_frozen_gepa

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / versions.MATH_EXECUTION_BINDING_PATH
GOOD = "Evaluate the mathematical relationships independently. Check each step and verify the conclusion using reliable general principles."
OTHER = "Consider the mathematical relationships independently. Examine the deductions carefully and verify the conclusion using general mathematical principles."


def contract():
    return json.loads(CONTRACT_PATH.read_bytes())


@pytest.mark.parametrize("field", ["canonical_manifest_sha256", "split_manifest_sha256", "benchmark_protocol_sha256",
    "initial_team_artifact_sha256", "initial_team_sha256", "pattern_prompt_sha256"])
def test_readiness_identity_poison(field):
    c = contract()
    assert not MATHExecutionBinding(ROOT, c).blockers()
    c[field] = "0" * 64
    assert MATHExecutionBinding(ROOT, c).blockers()


@pytest.mark.parametrize("field", ["models", "evaluator_pins", "membership_hashes", "provider_bounds", "budget", "memory_limits", "arms", "decoding", "access"])
def test_readiness_configuration_poison(field):
    c = contract()
    c[field] = {}
    assert MATHExecutionBinding(ROOT, c).blockers()


def test_gold_complete_optimize_and_split_metadata():
    binding = MATHExecutionBinding(ROOT, contract())
    assert not binding.blockers()
    assert len(binding.examples("optimize")) == 150
    manifest = binding.reader().manifest
    assert manifest["invalid_reference_counts_by_role"] == dict.fromkeys(("optimize", "shadow", "validation", "test"), 0)
    assert manifest["source_reference_audit"]["train"]["invalid_reference_count"] == 4
    assert manifest["source_reference_audit"]["test"]["invalid_reference_count"] == 0


@pytest.mark.parametrize("role", ["shadow", "validation", "test"])
@pytest.mark.parametrize("purpose", list(DataPurpose))
def test_heldout_reader_firewall(role, purpose, monkeypatch):
    reader = MATHExecutionBinding(ROOT, contract()).reader()
    if role == "shadow" and purpose == DataPurpose.ADAPTIVE_GATE:
        return  # Allowed private gate capability is exercised by full E2E.
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("forbidden role reached file open"))
    with pytest.raises(SearchContractError):
        reader.rows(role, purpose)


def test_reference_builder_requires_isolated_phase():
    from multi_dataset_diverse_rl.data_preparation.math_reference_split import build_reference_valid_split, data_preparation_context
    with pytest.raises(ValueError, match="DATA_PREPARATION_CONTEXT_REQUIRED"):
        build_reference_valid_split(ROOT, expected_canonical_sha256="0" * 64)
    with pytest.raises(ValueError, match="RUNTIME_IMPORT_FORBIDDEN"):
        with data_preparation_context():
            pytest.fail("runtime obtained a data-preparation capability")
    # Static production import closure may not reach the preparation module.
    from multi_dataset_diverse_rl.governance.source_identity import local_imports
    pending = list((ROOT / "multi_dataset_diverse_rl/search").glob("*.py"))
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        assert "data_preparation" not in path.parts
        pending.extend(local_imports(ROOT, path) - seen)


@pytest.mark.parametrize("count", [50, 300])
@pytest.mark.parametrize("vote,target,passed", [(0, -2, True), (-1, 1, False), (1, -3, False), (1, 0, True)])
def test_shadow_cardinality_preserves_guards(count, vote, target, passed):
    m = ShadowGateMetrics(10, 10 + vote, 10, 10 + target, count)
    assert evaluate_configured_shadow_gate(m, expected_row_count=count).passed is passed
    if count == 50:
        assert evaluate_shadow_gate(m) == evaluate_configured_shadow_gate(m, expected_row_count=count)
    else:
        with pytest.raises(ValueError):
            evaluate_shadow_gate(m)
    with pytest.raises(ValueError):
        evaluate_configured_shadow_gate(m, expected_row_count=count + 1)


@pytest.mark.parametrize("event", ["commit_broken", "team_probe", "common_safe", "shadow", "capacity", "operational"])
def test_memory_write_contract(event):
    from tests.test_unified_search_v2 import opportunity
    m = StructuredLongTermMemoryProviderV1(**contract()["memory_limits"])
    risks = {"team_probe": "TEAM_PROBE_REJECTION", "common_safe": "COMMON_SAFE_REJECTION"}
    diagnostics = {"team_newly_broken_count": 2}
    if event in risks:
        diagnostics["scientific_risk_code"] = risks[event]
    if event == "commit_broken":
        diagnostics["scientific_risk_code"] = "COMMON_SAFE_REJECTION"  # Stale rejection must not teach a committed winner.
    row = EvaluatedCandidate(SearchCandidate("candidate", GOOD), None, None, event != "capacity", False, diagnostics)
    outcome = OpportunityOutcome(opportunity(), (row,), "candidate" if event in {"commit_broken", "shadow"} else None,
        event == "commit_broken", False if event == "shadow" else True if event == "commit_broken" else None, 0,
        operational_failure=event == "operational")
    delta = m.prepare_outcome(outcome)
    m.validate_delta(delta)
    m.apply_outcome(delta)
    assert len(m.private) == int(event == "commit_broken")
    assert len(m.shared) == int(event in {"team_probe", "common_safe", "shadow"})
    assert all(e.risk_code != "NEWLY_BROKEN" for e in m.shared)
    view = m.read_for_opportunity(opportunity())
    assert bool(view.get("private")) == (event == "commit_broken")
    assert bool(view.get("shared")) == (event in {"team_probe", "common_safe", "shadow"})
    assert len(json.dumps(view, sort_keys=True)) <= 1200


def run_fake_arm(tmp_path, arm):
    c = contract()
    binding = MATHExecutionBinding(ROOT, c)
    assert not binding.blockers()
    team = json.loads((ROOT / c["initial_team_path"]).read_bytes())
    prompts = tuple(m["prompt"] for m in team["members"])
    adapter = MATHBenchmarkAdapter()
    examples = tuple(CorrectnessExample(BenchmarkInput(f"q{i}", f"Synthetic math fixture {i}: compute one plus zero.",
                       adapter.output_contract, benchmark_id="math"), "1") for i in range(6))
    shadow = tuple(CorrectnessExample(BenchmarkInput(f"gate{i}", f"Private gate synthetic fixture {i}: compute one plus zero.",
                     adapter.output_contract, benchmark_id="math"), "1") for i in range(300))
    public_invocations = []
    requests = []
    accounting = []
    contexts = []
    def fake_transport(req):
        requests.append(deepcopy(req))
        model = req["model"]
        if model == "qwen3-8b":
            shell, user = req["messages"]
            assert shell["content"] == adapter.output_contract
            prompt, problem = user["content"].split("\n\n", 1)
            assert "Optional Search Context" not in user["content"] and '"private"' not in user["content"]
            gate = problem.startswith("Private gate")
            i = int(problem.split("fixture ")[1].split(":")[0])
            if gate:
                correct = prompt in {prompts[1], prompts[2], GOOD, OTHER}
            elif prompt in prompts:
                member = prompts.index(prompt)
                correct = member in {1, 2} or member == 0 and i >= 3
            elif prompt == GOOD:
                correct = i != 2
            elif prompt == OTHER:
                correct = i >= 2
            else:
                raise AssertionError("unexpected fake proposal")
            return dict(text="Synthetic reasoning.\nFINAL_ANSWER: " + ("1" if correct else "2"), input_tokens=2, output_tokens=2)
        if len(req["messages"]) == 2:  # Real Pattern provider projection/schema parser.
            data = json.loads(req["messages"][1]["content"])
            assert all(r["source_split"] == "optimize" for r in data["evidence_rows"])
            return dict(text=json.dumps({"patterns": [dict(pattern_id="synthetic_mechanism", failure_mechanism="Missing consistency check",
                corrective_principle="Verify intermediate deductions", support_ids=data["residual_ids"], counterexample_ids=[], risk_ids=[], confidence=0.9)]}), input_tokens=2, output_tokens=2)
        assert all(adapter.output_contract not in m["content"] for m in req["messages"])
        return dict(text="```" + (GOOD if len(public_invocations) == 1 else OTHER) + "```", input_tokens=2, output_tokens=2)
    broker = RequestBroker(contract=c, transport=fake_transport, arm=arm, seed=81, ledger_writer=accounting.append)
    solver = BenchmarkSolver(adapter, broker)
    reflection = ReflectionProvider(broker)
    pattern = PatternProvider(broker, json.loads((ROOT / c["pattern_prompt_path"]).read_bytes())["prompt"]) if c["arms"][arm][0] else None
    def capture(**kwargs):
        # Dispatch to the unchanged actual pinned public GEPA implementation.
        saturation = next(cb for cb in kwargs["callbacks"] if hasattr(cb, "state") and hasattr(cb.state, "config"))
        public_invocations.append(dict(metric=kwargs["max_metric_calls"], minibatch=len(kwargs["batch_sampler"].schedule[0]),
            minibatch_kwarg=kwargs["reflection_minibatch_size"], sampler_type=type(kwargs["batch_sampler"]).__name__,
            promotion=orchestrator.evaluation.max_promoted,
            patience=[saturation.state.config.local_no_update_patience, orchestrator.stop.no_commit_patience], run_dir=kwargs["run_dir"]))
        contexts.append(kwargs["adapter"].optimization_context)
        return import_frozen_gepa().optimize(**kwargs)
    optimizer = GEPATeamExposureOptimizer(evaluator=solver, reflection_lm=reflection, accounting_reader=reflection.accounting,
                                         run_root=tmp_path, optimize_fn=capture)
    orchestrator = build_binary_orchestrator(benchmark=adapter, aggregation=EquivalencePluralityAggregation(),
        examples=examples, prompts=prompts, solver=solver, optimizer=optimizer, method=binding.method(arm), seed=81,
        shadow_loader=lambda: shadow, shadow_count=300, runtime_readiness=binding.blockers,
        pattern_provider=pattern, first_parent_epoch=False, provider_call_reader=lambda: broker.successes)
    orchestrator.state.initialize()
    initial = orchestrator.state.snapshot()
    diagnosis = orchestrator.analyzer.analyze(initial, orchestrator.history)
    result = asyncio.run(orchestrator.run(max_opportunities=3))
    assert result.transitions and result.trace[0].committed_candidate_id
    assert orchestrator.evaluation.provider.probed and orchestrator.evaluation.provider.fulled and orchestrator.gate.winner_count
    assert result.transitions[0].newly_fixed_ids == ("q0", "q1")
    assert result.transitions[0].parent_correctness and result.transitions[0].child_correctness
    assert all(r["metric"] == 36 and r["minibatch"] == 3 and r["promotion"] == 2 and r["patience"] == [3, 2] for r in public_invocations)
    assert all(r["minibatch_kwarg"] is None and r["sampler_type"] == "Layer2FrozenBatchSampler" and r["run_dir"] is None for r in public_invocations)
    assert not broker.usage["validation"] and not broker.usage["test"]
    # Gate cache capability must remain outside search's broker.
    assert not any("Private gate synthetic" in json.dumps(v) for v in broker.cache.values())
    assert not any(hasattr(orchestrator.gate, k) for k in ("examples", "profiles", "solver", "reader"))
    if c["arms"][arm][1]:
        audits = [t.memory_audit for t in result.trace]
        assert audits[0]["private_memory_count_by_member"]["0"] == 1
        assert audits[1]["read_private_count"] > 0
        assert audits[1]["shared_risk_count"] > 0
        assert audits[2]["read_shared_count"] > 0
        assert any('"private"' in ctx for ctx in contexts[1:])
        assert any('"shared"' in ctx for ctx in contexts[2:])
    if pattern:
        assert broker.usage["pattern"] > 0
        assert all('"pattern"' in ctx and '"failure_mechanism":"Missing consistency check"' in ctx for ctx in contexts)
    else:
        assert broker.usage["pattern"] == 0
    evidence = dict(arm=arm, pass_=True, budget=public_invocations[0], invocation_count=len(public_invocations),
        solver_interface_identity=c["solver_output_interface"]["identity"],
        solver_interface_sha256=c["solver_output_interface"]["sha256"],
        immutable_interface_present_once=all(r["messages"][0]["content"] == adapter.output_contract
            and json.dumps(r["messages"]).count("FINAL_ANSWER:") == 1 for r in requests if r["model"] == "qwen3-8b"),
        method_identity=orchestrator.method.identity(), target=result.trace[0].target_member,
        responsibility={str(k): asdict(v) for k, v in diagnosis.responsibility.items()},
        atomic_commits=len(result.transitions), full_count=len(orchestrator.evaluation.provider.fulled),
        probe_count=len(orchestrator.evaluation.provider.probed), gate_count=orchestrator.gate.winner_count,
        memory_audits=[t.memory_audit for t in result.trace], ledger_counts=broker.usage,
        real_solver_calls=0, real_optimizer_calls=0, real_pattern_calls=0,
        shadow_raw_search_leakage=0, validation_raw_search_leakage=0, test_raw_search_leakage=0)
    import os
    destination = os.environ.get("FORMAL_V3_EVIDENCE_CAPTURE_DIR")
    if destination:
        path = Path(destination).resolve()
        assert path.is_relative_to((ROOT / "runs").resolve())
        path.mkdir(parents=True, exist_ok=True)
        (path / f"math_v12_{arm}.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A4"])
def test_four_arm_actual_public_gepa_fake_e2e(tmp_path, arm):
    evidence = run_fake_arm(tmp_path, arm)
    assert evidence["pass_"]


def test_bound_derivation_and_empty_reference_rejection():
    b = provider_bounds(optimize_count=150, shadow_count=300)
    assert b["max_opportunities"] == 1 + 4 * (4 * 150 + 1)
    assert b["successful_provider_calls"] == b["solver_calls"] + b["reflection_calls"] + b["pattern_calls"]
    assert b["transport_attempts"] == 21 * b["successful_provider_calls"]
    for ref in ("", "  ", None):
        with pytest.raises(SearchContractError, match="REFERENCE_INVALID"):
            CorrectnessExample(None, ref)


def test_all_source_files_poison_before_provider(tmp_path):
    import shutil
    from multi_dataset_diverse_rl.governance.unified_execution import execution_identity, validate_frozen_source
    receipt = execution_identity(ROOT, contract())
    for row in receipt["execution_closure"]["files"]:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["path"], target)
    validate_frozen_source(tmp_path, receipt)
    for row in receipt["execution_closure"]["files"]:
        path = tmp_path / row["path"]
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(SearchContractError, match="SOURCE_IDENTITY_MISMATCH"):
            validate_frozen_source(tmp_path, receipt)
        path.write_bytes(original)
    validate_frozen_source(tmp_path, receipt)


@pytest.fixture
def fake_frozen_prep(tmp_path, monkeypatch):
    from multi_dataset_diverse_rl.governance import unified_execution as governed
    # Fake commit IO only for these authorization unit tests. The separate
    # source poison test uses every actual execution byte, and final freeze
    # verification checks the real Git source commit without this test seam.
    monkeypatch.setattr(governed, "verify_source_commit", lambda root, source, identity: None)
    monkeypatch.setattr(governed, "consumption_path", lambda root, scope: tmp_path / "global_consumption.json")
    manifest = governed.preexecution_manifest(ROOT, source_sha="a" * 40)
    prep = tmp_path / "prep"
    governed.prepare_canary(ROOT, manifest, destination=prep)
    return governed, prep


def test_preflight_draft_cannot_be_ready():
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest, bound_preflight
    manifest = preexecution_manifest(ROOT, source_sha=None, frozen=False)
    status = bound_preflight(ROOT, manifest)
    assert status["gate"] == "HOLD" and "PREEXECUTION_NOT_FROZEN" in status["blockers"]


def test_unauthorized_execute_never_constructs_client(fake_frozen_prep, monkeypatch, tmp_path):
    governed, prep = fake_frozen_prep
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda **kwargs: pytest.fail("unauthorized client construction"))
    with pytest.raises(SearchContractError, match="AUTHORIZATION_REQUIRED"):
        asyncio.run(governed.execute_canary(ROOT, prep, tmp_path / "run"))
    assert not (tmp_path / "run").exists() and not (prep / "authorization_consumed.json").exists()


@pytest.mark.parametrize("field,value", [("attempt_id", "rerun"), ("seed", 82), ("arm", "A2"), ("source_sha", "b" * 40),
    ("roles", ["solver", "reflection", "pattern"]), ("validation_calls", 1), ("transport_attempt_ceiling", 1)])
def test_authorization_scope_poison(fake_frozen_prep, field, value):
    governed, prep = fake_frozen_prep
    p = prep / "authorization.json"
    auth = json.loads(p.read_bytes())
    auth["explicit_user_authorized"] = True
    auth["scope"][field] = value
    p.write_bytes(json.dumps(auth).encode())
    with pytest.raises(SearchContractError, match="AUTHORIZATION_REQUIRED"):
        governed.validate_prep(ROOT, prep, require_authorized=True)


def test_consumption_survives_failed_client_construction(fake_frozen_prep, monkeypatch, tmp_path):
    governed, prep = fake_frozen_prep
    p = prep / "authorization.json"
    auth = json.loads(p.read_bytes())
    auth["explicit_user_authorized"] = True  # Synthetic permit, no real client.
    p.write_bytes(json.dumps(auth).encode())
    second = tmp_path / "independent_prep"
    governed.prepare_canary(ROOT, json.loads((prep / "prep.json").read_bytes())["manifest"], destination=second)
    (second / "authorization.json").write_bytes(json.dumps(auth).encode())
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    def fail(**kwargs):
        raise RuntimeError("synthetic construction failure")
    monkeypatch.setattr(ProviderClientFactory, "from_environment", fail)
    with pytest.raises(RuntimeError, match="synthetic construction failure"):
        asyncio.run(governed.execute_canary(ROOT, prep, tmp_path / "run"))
    assert (prep / "authorization_consumed.json").is_file()
    assert json.loads((tmp_path / "run/lifecycle.json").read_bytes())["status"] == "EXECUTION_ABORTED"
    assert json.loads((tmp_path / "run/consumed_authorization.json").read_bytes())["consumed"] is True
    assert (tmp_path / "run/raw_evidence_inventory.json").is_file()
    with pytest.raises(SearchContractError, match="AUTHORIZATION_REQUIRED"):
        asyncio.run(governed.execute_canary(ROOT, prep, tmp_path / "rerun"))
    with pytest.raises(SearchContractError, match="AUTHORIZATION_REQUIRED"):
        governed.validate_prep(ROOT, second, require_authorized=True)


def test_request_cache_accounting_and_gate_capability():
    c = contract()
    seen = []
    records = []
    def transport(request):
        seen.append(request)
        return dict(text="FINAL_ANSWER: 1", input_tokens=3, output_tokens=2)
    broker = RequestBroker(contract=c, transport=transport, arm="A1", seed=81, ledger_writer=records.append)
    kwargs = dict(role="solver", split="optimize", messages=[{"role": "user", "content": "public fixture"}])
    first = broker.complete(stage="gepa_local", **kwargs)
    second = broker.complete(stage="full", **kwargs)
    assert first["provider_called"] and not second["provider_called"]
    assert second["input_tokens"] == second["output_tokens"] == 0
    assert len(seen) == broker.successes == 1 and len(broker.cache) == 1
    gate = broker.private_capability()
    gate.complete(stage="adaptive_gate", **{**kwargs, "split": "shadow"})
    assert len(gate.cache) == 1 and len(broker.cache) == 1 and gate.cache is not broker.cache
    assert broker.usage["attempts"] == broker.usage["successes"] + broker.usage["failures"] == 2
    for split in ("validation", "test"):
        with pytest.raises(SearchContractError, match="ROLE_SPLIT_FORBIDDEN"):
            broker.complete(stage="forbidden", **{**kwargs, "split": split})
    with pytest.raises(SearchContractError, match="PATTERN_CALL_FORBIDDEN"):
        broker.complete(stage="pattern_diagnosis", **{**kwargs, "role": "pattern"})
    assert len(seen) == 2


def test_historical_frontier_stays_closed():
    import yaml
    original = subprocess.check_output(["git", "show", "fa718b9842fcfb50c5a651a59225d1435ebdd7e5:experiments/current_frontier.yaml"], cwd=ROOT)
    frontier = yaml.safe_load(original)
    assert frontier["real_execution_ready"] is False and frontier["real_api_authorized"] is False
    assert frontier["test_access"] == "sealed" and frontier["validation_access"] == "not_authorized"


def test_fake_canary_full_persistence(fake_frozen_prep, monkeypatch, tmp_path):
    from types import SimpleNamespace
    governed, prep = fake_frozen_prep
    p = prep / "authorization.json"
    auth = json.loads(p.read_bytes())
    auth["explicit_user_authorized"] = True  # Test-only synthetic permit.
    p.write_bytes(json.dumps(auth).encode())
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    observed = []
    def complete(**request):
        observed.append(request)
        if request["model"] == "qwen3-8b":
            assert request["extra_body"] == {"enable_thinking": False}
            text = "Synthetic reasoning.\nFINAL_ANSWER: " + ("1" if GOOD in request["messages"][1]["content"] else "2")
        else:
            assert request["model"] == "qwen3.7-flash"
            text = "```" + GOOD + "```"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda **kwargs: client)
    adapter = MATHBenchmarkAdapter()
    def examples(self, role):
        assert role in {"optimize", "shadow"}
        return tuple(CorrectnessExample(BenchmarkInput(f"{role}{i}", f"Synthetic {role} math fixture {i}: compute one plus zero.",
                     adapter.output_contract, benchmark_id="math"), "1") for i in range(6 if role == "optimize" else 300))
    monkeypatch.setattr(MATHExecutionBinding, "examples", examples)
    run = tmp_path / "complete"
    summary = asyncio.run(governed.execute_canary(ROOT, prep, run))
    assert observed and summary["result"]["stop_reason"] == "CANARY_PARENT_EPOCH_COMPLETE"
    assert len(summary["result"]["transitions"]) == 1
    assert summary["ledger"]["attempts"] == summary["ledger"]["successes"] and not summary["ledger"]["failures"]
    assert not summary["ledger"]["validation"] and not summary["ledger"]["test"] and not summary["ledger"]["pattern"]
    assert json.loads((run / "lifecycle.json").read_bytes())["status"] == "EXECUTION_COMPLETE"
    assert json.loads((run / "execution_summary.json").read_bytes()) == json.loads(json.dumps(summary))
    assert (run / "trajectory_private.jsonl").is_file() and (run / "provider_trace_private.jsonl").is_file()
    assert json.loads((run / "raw_evidence_inventory.json").read_bytes()) == governed.inventory(run)
    with pytest.raises(SearchContractError, match="AUTHORIZATION_REQUIRED"):
        governed.validate_prep(ROOT, prep, require_authorized=True)


@pytest.mark.parametrize("text,valid", [
    ("FINAL_ANSWER: 4", True), (r"FINAL_ANSWER: \frac{1}{2}", True),
    ("reasoning...\nFINAL_ANSWER: x=3", True), ("The answer is 4.", False),
    ("FINAL_ANSWER:", False), ("FINAL_ANSWER: 4\nFINAL_ANSWER: 4", False),
    ("FINAL_ANSWER: 4\nFINAL_ANSWER: 5", False), ("FINAL_ANSWER: 4\nMore prose", False)])
def test_v12_strict_format_regression(text, valid):
    item = BenchmarkInput("synthetic", "Synthetic public arithmetic fixture.", "stale item contract", benchmark_id="math")
    assert MATHBenchmarkAdapter().parse_member_output(text, item).valid is valid


def test_v12_request_capture_initial_and_evolved():
    c = contract()
    team = json.loads((ROOT / c["initial_team_path"]).read_bytes())
    prompts = [m["prompt"] for m in team["members"]] + [GOOD, OTHER,
        "Identify the mathematical relationships. Deduce the result and check the reasoning carefully."]
    captured, ledger = [], []
    def transport(request):
        captured.append(deepcopy(request))
        return dict(text="reasoning\nFINAL_ANSWER: 4", input_tokens=2, output_tokens=2)
    broker = RequestBroker(contract=c, transport=transport, arm="A1", seed=81, ledger_writer=ledger.append)
    solver = BenchmarkSolver(MATHBenchmarkAdapter(), broker)
    # A per-item contract cannot remove or replace the authoritative interface.
    item = BenchmarkInput("synthetic", "Unique synthetic problem slot: compute two plus two.", "stale per-item contract", benchmark_id="math")
    for index, prompt in enumerate(prompts):
        assert solver.solve(prompt, item, stage="initial" if index < 5 else "full", split="optimize")
        request = captured[-1]
        assert request["messages"] == [dict(role="system", content=solver.benchmark.output_contract),
                                        dict(role="user", content=prompt + "\n\n" + item.problem)]
        text = json.dumps(request["messages"])
        assert text.count("FINAL_ANSWER:") == text.count(prompt) == text.count(item.problem) == 1
    assert len(captured) == 8
    contracts = [r for r in ledger if r["kind"] == "SOLVER_REQUEST_CONTRACT"]
    assert len(contracts) == 8 and len({r["effective_solver_request_contract_hash"] for r in contracts}) == 8
    assert [r["mutable_prompt_sha256"] for r in contracts[:5]] == [m["prompt_sha256"] for m in team["members"]]
    from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import validate_mutable_decision_procedure
    with pytest.raises(ValueError, match="mutable_prompt_contract_violation"):
        validate_mutable_decision_procedure(solver.benchmark.output_contract)


def test_v12_old_cache_and_cross_attempt_isolation():
    import hashlib
    facts = json.loads((ROOT / "tests/fixtures/math_canary_attempt1_format_facts.json").read_bytes())
    c = contract()
    calls = []
    def transport(request):
        calls.append(request)
        return dict(text="FINAL_ANSWER: 4", input_tokens=2, output_tokens=2)
    broker = RequestBroker(contract=c, transport=transport, arm="A1", seed=81)
    broker.cache[facts["provider_request_sha256"]] = dict(text="historical-invalid", input_tokens=1, output_tokens=1)
    messages = [dict(role="system", content=MATHBenchmarkAdapter.output_contract), dict(role="user", content="synthetic cache fixture")]
    kwargs = dict(role="solver", split="optimize", stage="initial", messages=messages)
    assert broker.complete(**kwargs)["provider_called"]
    assert not broker.complete(**kwargs)["provider_called"]
    for field, value in [("cache_namespace", "fresh-other-attempt"), ("solver_output_interface", {**c["solver_output_interface"], "identity": "OLD_INTERFACE"})]:
        changed = deepcopy(c)
        changed[field] = value
        other = RequestBroker(contract=changed, transport=transport, arm="A1", seed=81)
        other.cache.update(broker.cache)
        assert other.complete(**kwargs)["provider_called"]
    assert len(calls) == 3


def test_v12_attempt1_sanitized_format_negative_control():
    facts = json.loads((ROOT / "tests/fixtures/math_canary_attempt1_format_facts.json").read_bytes())
    assert facts["root_cause"] == "R2" and facts["request_marker_occurrences"] == 1
    assert facts["response_marker_lines"] == 0 and facts["parser_valid"] is False
    # Public fixture contains format facts only. Actual private response replay
    # is an independently hashed audit receipt, never copied into the tests.
    raw = "\n".join("Synthetic marker-free reasoning." for _ in range(facts["response_nonempty_lines"]))
    item = BenchmarkInput("synthetic", "Synthetic fixture.", "stale item contract", benchmark_id="math")
    assert not MATHBenchmarkAdapter().parse_member_output(raw, item).valid


def test_v12_invalid_fake_execution_fails_closed(fake_frozen_prep, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    governed, prep = fake_frozen_prep
    p = prep / "authorization.json"
    auth = json.loads(p.read_bytes())
    assert auth["scope"]["attempt_id"] == "math_unified_v2_A1_seed81_canary_attempt2"
    auth["explicit_user_authorized"] = True  # Isolated synthetic permit only.
    p.write_bytes(json.dumps(auth).encode())
    captured = []
    def complete(**request):
        captured.append(request)
        assert request["messages"][0]["content"] == MATHBenchmarkAdapter.output_contract
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="The answer is 4."))],
                               usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setattr(ProviderClientFactory, "from_environment", lambda **kwargs: client)
    monkeypatch.setattr(MATHExecutionBinding, "examples", lambda self, role: (
        CorrectnessExample(BenchmarkInput("synthetic", "Synthetic public arithmetic.", "stale item contract", benchmark_id="math"), "4"),))
    run = tmp_path / "invalid"
    with pytest.raises(SearchContractError, match="SOLVER_INVALID_RESPONSE_NO_REGENERATION"):
        asyncio.run(governed.execute_canary(ROOT, prep, run))
    assert len(captured) == 1
    lifecycle = json.loads((run / "lifecycle.json").read_bytes())
    assert lifecycle["status"] == "EXECUTION_ABORTED" and lifecycle["provider_usage"]["solver"] == 1
    assert not lifecycle["provider_usage"]["reflection"] and not lifecycle["provider_usage"]["pattern"]
    assert not (run / "execution_summary.json").exists()
    assert not (run / "final_team_private.json").exists()
    assert (prep / "authorization_consumed.json").exists()
    with pytest.raises(SearchContractError, match="AUTHORIZATION_REQUIRED"):
        governed.validate_prep(ROOT, prep, require_authorized=True)


@pytest.mark.parametrize("field", ["solver_output_interface", "cache_namespace", "canary_attempt_id"])
def test_v12_interface_binding_poison(field):
    c = contract()
    c[field] = None
    assert MATHExecutionBinding(ROOT, c).blockers()
