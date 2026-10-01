"""Deterministic data-only split, firewall, provenance and global-model contracts."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from multi_dataset_diverse_rl.benchmarks.access import DataPurpose
from multi_dataset_diverse_rl.benchmarks.data_freeze import freeze_local, file_hash, wrap_rows, canonical
from multi_dataset_diverse_rl.benchmarks.experiment_binding import ExperimentModelBinding
from multi_dataset_diverse_rl.benchmarks.experiment_splits import (
    COUNTS, ROLES, quotas, select_groups, overlap_audit, freeze_experiment_split,
    verify_experiment_split, ExperimentSplitReader, optimize_to_legacy_search)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from tests.test_benchmark_data_freeze_v1 import rows


@pytest.fixture
def frozen(tmp_path):
    root, dest = tmp_path / "canonical", tmp_path / "experiment"
    freeze_local(root, "ifbench", {"train": rows(1000, "ifbench"), "test": rows(294, "ifbench", "test")}, {"files": []})
    sha = file_hash(root / "manifests/ifbench.json")
    manifest = freeze_experiment_split(root, dest, "ifbench", expected_canonical_sha256=sha)
    return root, dest, manifest


def test_freeze_repeated_bytes_and_canonical_preservation(frozen):
    root, dest, manifest = frozen
    before = {p.name: p.read_bytes() for p in (root / "manifests").iterdir()}
    ids = (dest / manifest["membership_file"]).read_bytes()
    repeated = freeze_experiment_split(root, dest, "ifbench", expected_canonical_sha256=manifest["canonical_dataset_manifest_sha256"])
    assert canonical(repeated) == canonical(manifest)
    assert (dest / manifest["membership_file"]).read_bytes() == ids
    assert before == {p.name: p.read_bytes() for p in (root / "manifests").iterdir()}
    assert manifest["counts"] == COUNTS["ifbench"]
    assert all(p["unchanged"] for p in manifest["ifbench_existing_membership_parity"].values())
    assert verify_experiment_split(root, dest, "ifbench", expected_manifest_sha256=file_hash(dest / "ifbench.json"))["verified"]


def test_source_manifest_mismatch_fails_closed(frozen):
    root, dest, _ = frozen
    with pytest.raises(ValueError, match="CANONICAL_MANIFEST_IDENTITY_MISMATCH"):
        freeze_experiment_split(root, dest, "ifbench", expected_canonical_sha256="0" * 64)


def test_existing_membership_mismatch_preserved(frozen):
    root, dest, manifest = frozen
    path = dest / "ifbench.ids.jsonl"
    path.write_bytes(b"corrupted immutable artifact\n")
    with pytest.raises(ValueError, match="EXISTING_FROZEN_MEMBERSHIP_MISMATCH"):
        freeze_experiment_split(root, dest, "ifbench", expected_canonical_sha256=manifest["canonical_dataset_manifest_sha256"])
    assert path.read_bytes() == b"corrupted immutable artifact\n"


def test_source_bytes_poison_pre_read(frozen):
    root, dest, _ = frozen
    reader = ExperimentSplitReader(root, dest, "ifbench", expected_manifest_sha256=file_hash(dest / "ifbench.json"))
    path = root / "raw/ifbench/train.jsonl"
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(SearchContractError, match="CANONICAL_SOURCE_HASH_MISMATCH"):
        reader.rows("optimize", DataPurpose.PATTERN)


@pytest.mark.parametrize("purpose", list(DataPurpose))
@pytest.mark.parametrize("role", ("validation", "test"))
def test_heldout_access_rejected_before_file_io(frozen, monkeypatch, purpose, role):
    root, dest, _ = frozen
    reader = ExperimentSplitReader(root, dest, "ifbench", expected_manifest_sha256=file_hash(dest / "ifbench.json"))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("role denial must precede I/O"))
    assert reader.metadata(role)["count"] in (294, 300)
    with pytest.raises(SearchContractError, match="HELDOUT_SEARCH_ACCESS_FORBIDDEN"):
        reader.rows(role, purpose)


@pytest.mark.parametrize("purpose", [p for p in DataPurpose if p != DataPurpose.ADAPTIVE_GATE])
def test_shadow_search_pattern_memory_inaccessible(frozen, purpose):
    root, dest, _ = frozen
    reader = ExperimentSplitReader(root, dest, "ifbench", expected_manifest_sha256=file_hash(dest / "ifbench.json"))
    with pytest.raises(SearchContractError, match="SHADOW_CONTENT_SEARCH_VISIBILITY_FORBIDDEN"):
        reader.rows("shadow", purpose)


def test_optimize_roles_and_gate_are_separate(frozen):
    root, dest, _ = frozen
    reader = ExperimentSplitReader(root, dest, "ifbench", expected_manifest_sha256=file_hash(dest / "ifbench.json"))
    pattern = reader.rows("optimize", DataPurpose.PATTERN)
    memory = reader.rows("optimize", DataPurpose.MEMORY)
    shadow = reader.rows("shadow", DataPurpose.ADAPTIVE_GATE)
    assert pattern == memory and len(pattern) == 150 and len(shadow) == 300
    assert not {r["stable_example_id"] for r in pattern} & {r["stable_example_id"] for r in shadow}
    with pytest.raises(SearchContractError):
        reader.rows("optimize", DataPurpose.ADAPTIVE_GATE)
    assert optimize_to_legacy_search("optimize") == "search"
    with pytest.raises(SearchContractError):
        optimize_to_legacy_search("validation")


def math_sources():
    subjects = ["Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus"]
    return {split: wrap_rows("math", split, [{"id": i, "problem": f"synthetic {split} {i}",
        "solution": r"\boxed{1}", "type": subjects[i % 7], "level": "Level 1"} for i in range(n)])
        for split, n in (("train", 7500), ("test", 5000))}


@pytest.mark.parametrize("benchmark", ("math", "hotpotqa", "ifbench"))
def test_row_reorder_keeps_selected_ids(benchmark):
    if benchmark == "math":
        sources, legacy = math_sources(), []
    elif benchmark == "hotpotqa":
        sources, legacy = {"train": rows(1000), "validation": rows(400, split="validation")}, []
    else:
        sources = {"train": rows(1000, "ifbench"), "test": rows(294, "ifbench", "test")}
        legacy = [{"stable_example_id": r["stable_example_id"], "project_split": role}
                  for role, subset in (("search", sources["train"][300:450]), ("shadow", sources["train"][:300]), ("test", sources["test"])) for r in subset]
    a = select_groups(benchmark, sources, legacy)
    b = select_groups(benchmark, {k: list(reversed(v)) for k, v in sources.items()}, legacy)
    assert {r: [x["stable_example_id"] for x in a[r]] for r in ROLES} == {r: [x["stable_example_id"] for x in b[r]] for r in ROLES}
    assert {r: len(a[r]) for r in ROLES} == COUNTS[benchmark]
    assert overlap_audit(a)["status"] == "DISJOINT"


def test_largest_remainder_tie_and_math_count_gate():
    assert quotas({"b": 1, "a": 1}, 1) == {"b": 0, "a": 1}
    with pytest.raises(ValueError, match="STOP_MATH_SOURCE_COUNT_MISMATCH"):
        select_groups("math", {"train": [], "test": []})


def test_overlap_on_input_hash_is_blocking():
    sources = {"train": rows(1000), "validation": rows(400, split="validation")}
    groups = select_groups("hotpotqa", sources)
    groups["test"][0]["input_sha256"] = groups["validation"][0]["input_sha256"]
    assert overlap_audit(groups)["status"] == "STOP_SPLIT_OVERLAP"


@pytest.mark.parametrize("benchmark", ("math", "ifbench", "hotpotqa"))
@pytest.mark.parametrize("arm", ("A1", "A2", "A3", "A4"))
def test_models_bound_across_every_benchmark_arm_seed(benchmark, arm):
    binding = ExperimentModelBinding()
    for seed in (81, 82, 83):
        bound = binding.bind_config(Config(), benchmark=benchmark, arm=arm, seed=seed)
        assert bound.models.agent_model == "qwen3-8b"
        assert bound.models.optimizer_model == binding.pattern == "qwen3.7-flash"
        assert bound.training.agents == 5 and bound.training.seed == seed
        with pytest.raises(SearchContractError, match="MODEL_BINDING_MISMATCH"):
            replace(binding, solver="qwen3.7-flash-2026-07-15").require(benchmark, arm, seed)


def test_current_suite_and_bbh_history_only():
    from multi_dataset_diverse_rl.governance.registries import load_yaml
    root = Path(__file__).resolve().parents[1]
    frontier = load_yaml(root / "experiments/current_frontier.yaml")
    assert frontier["current_benchmark_suite"] == ["math", "ifbench", "hotpotqa"]
    assert frontier["historical_benchmark_only"] is True
    assert not frontier["real_api_authorized"]
    if frontier["real_execution_ready"] == "true_for_canary_only":
        # Versioned preexecution readiness supersedes the migration-time HOLD;
        # independent authorization and held-out locks remain closed.
        from multi_dataset_diverse_rl.governance.unified_execution import bound_preflight
        assert frontier["validation_access"] == "not_authorized" and frontier["test_access"] == "sealed"
        manifest = load_yaml(root / frontier["canary_manifest"])
        assert manifest["lifecycle"]["status"] == "PREEXECUTION_FROZEN"
        assert not bound_preflight(root, manifest)["blockers"]
    else:
        assert frontier["real_execution_ready"] is False


def test_public_data_guard_denies_provider_and_hf_inference_before_transport(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "tests/public_data_network"),
           "PUBLIC_DATA_NETWORK_LOG": str(tmp_path / "controls.jsonl")}
    source = """import urllib.request, socket, sitecustomize
assert socket.getaddrinfo.__module__ == 'sitecustomize'
for url in ('https://api.openai.com/v1/responses', 'https://huggingface.co/api/inference'):
    try:
        urllib.request.urlopen(url, timeout=1)
        raise AssertionError('unapproved transport admitted')
    except PermissionError:
        pass
print('DATA_ONLY_PROVIDER_NEGATIVE_CONTROL_PASS')
"""
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "DATA_ONLY_PROVIDER_NEGATIVE_CONTROL_PASS" in result.stdout
    events = [json.loads(line) for line in (tmp_path / "controls.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "blocked_non_data_transport" for e in events) == 2
    assert not any(e["kind"] == "allowed_http_attempt" for e in events)
