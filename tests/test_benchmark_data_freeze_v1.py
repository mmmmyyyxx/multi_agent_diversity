"""Synthetic split/guard contracts and real-data identity checks; no LLMs."""
from dataclasses import replace
import json
from pathlib import Path
import random
import socket

import pytest

from multi_dataset_diverse_rl.benchmarks.access import DataPurpose, FrozenSplitReader, SplitAccessPolicy, AdaptiveGateFeedback
from multi_dataset_diverse_rl.benchmarks.data_freeze import (
    BENCHMARK_IDS, SOURCE_PINS, build_manifest, canonical, freeze_local,
    file_hash, gepa_split, hover_three_hop, isolation_audit, reference_answer,
    select_splits, verify_only, wrap_rows,
)
from multi_dataset_diverse_rl.benchmarks.registry import BENCHMARKS
from multi_dataset_diverse_rl.benchmarks.scorer_contracts import PUPAJudgeOutputs, pupa_contract_score
from multi_dataset_diverse_rl.search.schemas import EvidenceItem, EvidenceView, SearchContractError

ROOT = Path(__file__).resolve().parents[1] / "data/benchmark_suite_v1"


def rows(n, benchmark="hotpotqa", split="train"):
    field = {"hotpotqa": "question", "hover": "claim", "ifbench": "prompt", "pupa": "user_query", "math": "problem"}[benchmark]
    raw = [{field: f"synthetic {split} {i}", "id": i} for i in range(n)]
    for i, row in enumerate(raw):
        if benchmark == "hotpotqa":
            row.update(answer="synthetic answer", context=[], supporting_facts=[])
        elif benchmark == "ifbench":
            row.update(instruction_id_list=["synthetic"], kwargs=[{}], key=i)
        elif benchmark == "pupa":
            row.update(pii_units="synthetic unit", target_response="synthetic response")
    if benchmark == "math":
        for row in raw:
            row.update(solution=r"Answer: \boxed{\frac{1}{2}}", type="Algebra", level="Level 1")
    return wrap_rows(benchmark, split, raw)


def test_gepa_generic_split_exact_algorithm():
    data = rows(1000)
    result = gepa_split(data)
    assert result["search"] == random.Random(1).sample(data[800:], 150)
    assert result["shadow"] == random.Random(1).sample(data[400:800], 300)
    assert result["test"] == random.Random(1).sample(data[:400], 300)
    small = gepa_split(data[:10])
    assert small["search"] == data[8:10] and small["test"] == data[:4]


def test_hotpotqa_split_counts():
    assert {g: len(r) for g, r in select_splits("hotpotqa", {"train": rows(2000)}).items()} == {"search": 150, "shadow": 300, "validation": 0, "test": 300}


def hover_rows():
    return wrap_rows("hover", "train", [{"uid": str(i), "claim": f"synthetic {i}", "label": 0, "num_hops": 2 + i % 3, "hpqa_id": str(i), "supporting_facts": [{"key": str(k), "value": 0} for k in range(2 + i % 3)]} for i in range(30)])


def test_hover_three_hop_filter():
    data = hover_rows()
    assert {r["source_index"] for r in hover_three_hop(data)} == set(range(1, 30, 3))
    # Rows can contain repeated sentences within the same document.
    data[1]["content"]["supporting_facts"].append({"key": "0", "value": 4})
    assert data[1] in hover_three_hop(data)


def test_hover_seed0_shuffle():
    data = hover_rows()
    expected = data[1::3]
    random.Random(0).shuffle(expected)
    assert hover_three_hop(data) == expected


def test_ifbench_exact_slice_and_sample():
    train, test = rows(600, "ifbench"), rows(294, "ifbench", "test")
    result = select_splits("ifbench", {"train": train, "test": test})
    assert result["search"] == random.Random(1).sample(train[300:600], 150)
    assert result["shadow"] == train[:300] and result["test"] == test


def test_pupa_exact_sequential_split():
    data = rows(443, "pupa")
    result = select_splits("pupa", {"train": data})
    assert result == {"search": data[:111], "shadow": data[111:222], "validation": [], "test": data[222:]}


def test_math_exact_or_project_status(tmp_path):
    sources = {"train": rows(7500, "math"), "test": rows(5000, "math", "test")}
    result, _ = build_manifest(tmp_path, "math", sources, {})
    assert result["agentgrad_status"] == "AGENTGRAD_MATH_EXACT_SPLIT_UNRESOLVED"
    assert result["selection_rule"] == "PROJECT_MATH_V1_PROPOSED"
    assert result["split_exactness"] == "PROJECT_PROPOSED"
    train = sources["train"][:]
    random.Random(1).shuffle(train)
    selected = select_splits("math", sources)
    assert selected["search"] == train[:150] and selected["shadow"] == train[150:450]
    assert reference_answer(r"\boxed{1} then \boxed{\frac{2}{3}}") == r"\frac{2}{3}"
    assert reference_answer("no boxed answer") is None


def test_split_pairwise_disjoint():
    result = gepa_split(rows(2000))
    assert isolation_audit(result)["status"] == "DISJOINT"


def test_content_hash_overlap():
    result = gepa_split(rows(2000))
    result["test"][0] = {**result["test"][0], "content_sha256": result["search"][0]["content_sha256"]}
    audit = isolation_audit(result)
    assert audit["status"] == "SOURCE_DUPLICATION_DETECTED" and audit["FORMAL_READY"] == "NO"
    result["test"][1] = {**result["test"][1], "input_sha256": result["shadow"][0]["input_sha256"]}
    assert isolation_audit(result)["status"] == "SOURCE_DUPLICATION_DETECTED"


def fixture_freeze(tmp_path):
    source = {"train": rows(443, "pupa")}
    metadata = {"files": [], "datasets_version": "3.6.0"}
    return source, metadata, freeze_local(tmp_path, "pupa", source, metadata)


def test_manifest_reproducible(tmp_path):
    source, metadata, first = fixture_freeze(tmp_path)
    assert freeze_local(tmp_path, "pupa", source, metadata) == first
    source["train"][0]["content"]["user_query"] = "changed synthetic data"
    with pytest.raises(ValueError, match="FROZEN_ARTIFACT_MISMATCH"):
        freeze_local(tmp_path, "pupa", source, metadata)


def test_verify_only_no_network(tmp_path, monkeypatch):
    fixture_freeze(tmp_path)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network attempt"))
    assert verify_only(tmp_path, "pupa")["network_attempts"] == 0
    path = tmp_path / "materialized/pupa/search.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="MATERIALIZED_SPLIT_MISMATCH"):
        verify_only(tmp_path, "pupa")


@pytest.mark.parametrize("purpose", list(DataPurpose))
def test_test_split_search_access_forbidden(tmp_path, purpose):
    fixture_freeze(tmp_path)
    reader = FrozenSplitReader(tmp_path, "pupa", expected_manifest_sha256=file_hash(tmp_path / "manifests/pupa.json"))
    with pytest.raises(SearchContractError, match="HELDOUT_SEARCH_ACCESS_FORBIDDEN"):
        reader.rows("test", purpose)
    reader.policy.require("test", purpose, manifest_only=True)


@pytest.mark.parametrize("purpose", [p for p in DataPurpose if p != DataPurpose.ADAPTIVE_GATE])
def test_shadow_content_not_search_visible(tmp_path, purpose):
    fixture_freeze(tmp_path)
    reader = FrozenSplitReader(tmp_path, "pupa", expected_manifest_sha256=file_hash(tmp_path / "manifests/pupa.json"))
    with pytest.raises(SearchContractError, match="SHADOW_CONTENT_SEARCH_VISIBILITY_FORBIDDEN"):
        reader.rows("shadow", purpose)
    assert len(reader.rows("shadow", DataPurpose.ADAPTIVE_GATE)) == 111
    assert AdaptiveGateFeedback(.5, False).passed is False
    with pytest.raises(SearchContractError, match="evidence must come from Optimize"):
        EvidenceView((EvidenceItem("fake", "shadow", frozenset({"RESPONSIBILITY"})),), (), (), "search", "shadow")


def test_pupa_reports_have_no_raw_pii(tmp_path):
    raw = {"user_query": "Synthetic Secret Jane 555-0100", "pii_units": "Secret Jane||555-0100", "target_response": "Synthetic confidential response", "id": "PII secret identifier"}
    sources = {"train": wrap_rows("pupa", "train", [raw] * 443)}
    manifest, membership = build_manifest(tmp_path, "pupa", sources, {})
    safe = canonical(manifest) + membership
    for secret in ("Secret Jane", "555-0100", "Synthetic confidential response", "PII secret identifier"):
        assert secret.encode() not in safe


@pytest.mark.parametrize("flag", ["split_frozen", "provenance_frozen", "output_contract_frozen", "dataset_adapter_ready", "evaluator_ready", "aggregation_policy_frozen", "responsibility_policy_frozen"])
def test_unified_search_ready_matches_blockers(flag):
    flags = {key: True for key in ("split_frozen", "provenance_frozen", "output_contract_frozen", "dataset_adapter_ready", "evaluator_ready", "aggregation_policy_frozen", "responsibility_policy_frozen")}
    spec = replace(BENCHMARKS["hotpotqa"], **flags)
    assert spec.unified_search_ready == (not spec.blockers())
    spec = replace(spec, **{flag: False})
    assert not spec.unified_search_ready and spec.blockers()


def test_pupa_fake_judge_contract():
    assert pupa_contract_score(PUPAJudgeOutputs(True, False, 1), "unit1||unit2") == .75
    assert pupa_contract_score(PUPAJudgeOutputs(False, False, 0), "unit1") == 1
    assert pupa_contract_score(PUPAJudgeOutputs(False, True, 0), "unit1") == .5
    assert pupa_contract_score(PUPAJudgeOutputs(True, False, 0), "unit1", trace=True) is True


def test_reader_requires_frozen_manifest_identity(tmp_path):
    fixture_freeze(tmp_path)
    with pytest.raises(SearchContractError, match="BENCHMARK_MANIFEST_IDENTITY_MISMATCH"):
        FrozenSplitReader(tmp_path, "pupa", expected_manifest_sha256="0" * 64)


def test_ifbench_registry_receipt_matches_manifest():
    from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
    path = ROOT / "manifests/ifbench.json"
    manifest = json.loads(path.read_text())
    assert BENCHMARKS["ifbench"].split_identity == file_hash(path)
    assert BENCHMARKS["ifbench"].dataset_identity == digest(manifest["source"])


def test_public_download_cache_requires_receipt_and_rejects_provider_hosts(tmp_path):
    from multi_dataset_diverse_rl.benchmarks.data_sources import PublicDataDownloader
    download = PublicDataDownloader(tmp_path)
    with pytest.raises(ValueError, match="PUBLIC_DATA_SOURCE_NOT_ALLOWED"):
        download.fetch("https://api.openai.com/v1/responses", "not-data.json")
    with pytest.raises(ValueError, match="DOWNLOAD_FILENAME_INVALID"):
        download.fetch("https://huggingface.co/example", "../outside.json")
    (tmp_path / "fixture.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="CACHED_DOWNLOAD_RECEIPT_MISSING"):
        download.fetch("https://huggingface.co/example", "fixture.json")


@pytest.mark.parametrize("benchmark", BENCHMARK_IDS)
def test_frozen_real_data_identity(benchmark):
    path = ROOT / "manifests" / (benchmark + ".json")
    if not path.exists():
        pytest.skip("USER_DEFERRED_DOWNLOAD: no frozen real-data manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"search": 111 if benchmark == "pupa" else 150, "shadow": 111 if benchmark == "pupa" else 300,
                "validation": 0, "test": {"ifbench": 294, "pupa": 221}.get(benchmark, 300)}
    assert manifest["counts"] == expected and manifest["source"]["revision"] == SOURCE_PINS[benchmark]["revision"]
    members = [json.loads(line) for line in (ROOT / "manifests" / manifest["membership_file"]).read_text(encoding="utf-8").splitlines()]
    assert len(members) == sum(expected.values())
    assert all(len(row["content_sha256"]) == 64 and row["source_index"] >= 0 for row in members)
