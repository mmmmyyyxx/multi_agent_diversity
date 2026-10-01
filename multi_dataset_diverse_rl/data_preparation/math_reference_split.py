"""Isolated reference-validity derivation. No provider or optimizer imports."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import sys

from ..versions import MATH_EXPERIMENT_SPLIT_VERSION, MATH_REFERENCE_VALIDITY_VERSION, MATH_REFERENCE_EXTRACTOR_VERSION
from ..benchmarks.data_freeze import canonical, digest, file_hash, immutable_write, jsonl_bytes, membership, read_jsonl, reference_answer
from ..benchmarks.experiment_splits import ACCESS, COUNTS, ROLES, SEED, distribution, overlap_audit, stratified_math_groups

_PHASE = ContextVar("math_reference_preparation", default=None)


@contextmanager
def data_preparation_context():
    forbidden = ("multi_dataset_diverse_rl.local_optimizers", "multi_dataset_diverse_rl.provider_factory",
                 "multi_dataset_diverse_rl.llm_client", "multi_dataset_diverse_rl.search.gepa",
                 "multi_dataset_diverse_rl.search.patterns", "multi_dataset_diverse_rl.search.memory")
    if any(name.startswith(forbidden) for name in sys.modules):
        raise ValueError("DATA_PREPARATION_RUNTIME_IMPORT_FORBIDDEN")
    token = _PHASE.set("DATA_PREPARATION_CONTEXT")
    try:
        yield
    finally:
        _PHASE.reset(token)


def build_reference_valid_split(canonical_root: Path, *, expected_canonical_sha256: str):
    if _PHASE.get() != "DATA_PREPARATION_CONTEXT":
        raise ValueError("DATA_PREPARATION_CONTEXT_REQUIRED")
    path = canonical_root / "manifests/math.json"
    if file_hash(path) != expected_canonical_sha256:
        raise ValueError("CANONICAL_MANIFEST_IDENTITY_MISMATCH")
    legacy = json.loads(path.read_bytes())
    eligible, audits = {}, {}
    for source in legacy["source"]["canonical_sources"]:
        raw_path = canonical_root / "raw/math" / (source["name"] + ".jsonl")
        if file_hash(raw_path) != source["canonical_sha256"]:
            raise ValueError("CANONICAL_SOURCE_HASH_MISMATCH")
        rows = read_jsonl(raw_path)
        if len(rows) != source["row_count"]:
            raise ValueError("CANONICAL_SOURCE_COUNT_MISMATCH")
        selected, invalid = [], []
        for row in rows:
            extracted = reference_answer(row["content"]["solution"])
            if extracted != row["reference_final_answer"]:
                raise ValueError("FROZEN_REFERENCE_DERIVATION_MISMATCH")
            (selected if extracted is not None and extracted.strip() else invalid).append(row)
        eligible[source["name"]] = selected
        audits[source["name"]] = {
            "total":len(rows), "eligible_count":len(selected), "invalid_reference_count":len(invalid),
            "invalid_by_subject":dict(sorted(Counter(r["content"]["type"] for r in invalid).items())),
            "invalid_by_level":dict(sorted(Counter(r["content"]["level"] for r in invalid).items())),
            "eligible_universe_hash":digest([r["stable_example_id"] for r in selected]),
            "excluded_identity_set_hash":digest(sorted(r["stable_example_id"] for r in invalid)),
            "canonical_source_sha256":source["canonical_sha256"]}
    if {s:audits[s]["total"] for s in audits} != {"train":7500,"test":5000}:
        raise ValueError("STOP_MATH_SOURCE_COUNT_MISMATCH")
    groups = stratified_math_groups(eligible, COUNTS["math"], seed=SEED)
    overlap = overlap_audit(groups)
    if overlap["status"] != "DISJOINT":
        raise ValueError("STOP_SPLIT_OVERLAP")
    members = [membership(row,role) for role in ROLES for row in groups[role]]
    ids = jsonl_bytes(members)
    hashes = {role:digest([r["stable_example_id"] for r in groups[role]]) for role in ROLES}
    manifest = {
        "benchmark_id":"math", "status":"FROZEN", "protocol":MATH_EXPERIMENT_SPLIT_VERSION,
        "canonical_dataset_manifest_sha256":expected_canonical_sha256,
        "source_revision":legacy["source"]["revision"], "canonical_sources":legacy["source"]["canonical_sources"],
        "reference_extractor":MATH_REFERENCE_EXTRACTOR_VERSION, "reference_validity_policy":MATH_REFERENCE_VALIDITY_VERSION,
        "reference_resolution_case":"B", "source_reference_audit":audits,
        "split_seed":SEED, "split_algorithm":"eligible subject proportions; unchanged largest remainder, lexicographic subject ties, SHA256(str(seed)+stable_id), sequential disjoint role quotas",
        "counts":COUNTS["math"], "hashes":hashes, "membership_file":"math.ids.jsonl",
        "membership_sha256":digest_bytes(ids),
        "role_membership_sha256":{role:digest_bytes(jsonl_bytes([r for r in members if r["project_split"]==role])) for role in ROLES},
        "pairwise_overlap_audit":overlap, "access_policy":ACCESS,
        "invalid_reference_counts_by_role":{role:sum(not str(r.get("reference_final_answer") or "").strip() for r in groups[role]) for role in ROLES},
        "distributions":{role:distribution("math",rows) for role,rows in groups.items()},
        "frozen_before_real_solver_and_efficacy":True, "canonical_dataset_changed":False}
    return manifest, ids


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def freeze_reference_valid_split(canonical_root, destination, *, expected_canonical_sha256):
    manifest, ids = build_reference_valid_split(canonical_root, expected_canonical_sha256=expected_canonical_sha256)
    targets = {destination/"math.json":canonical(manifest)+b"\n", destination/"math.ids.jsonl":ids}
    for path,value in targets.items():
        if path.exists() and path.read_bytes() != value:
            raise ValueError("EXISTING_FROZEN_MEMBERSHIP_MISMATCH")
    for path,value in targets.items():
        immutable_write(path,value)
    return manifest
