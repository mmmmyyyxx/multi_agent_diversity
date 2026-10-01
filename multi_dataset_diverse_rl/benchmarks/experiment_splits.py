"""Immutable experiment memberships derived from canonical source rows.

No raw rows are copied. Canonical freeze V1 and its memberships remain intact.
Runtime readers enforce roles before opening source rows; verification and
split construction are offline data-governance actions, not search actions.
"""
from collections import Counter
from dataclasses import replace
from itertools import combinations
import hashlib
import json
from pathlib import Path

from ..versions import (BENCHMARK_EXPERIMENT_SPLIT_VERSION, CURRENT_RESEARCH_BENCHMARK_SUITE,
                        MATH_EXPERIMENT_SPLIT_VERSION)
from ..search.schemas import SearchContractError
from .access import DataPurpose
from .data_freeze import (SOURCE_PINS, canonical, digest, file_hash, immutable_write,
                          jsonl_bytes, membership, read_jsonl, verify_only)

PROTOCOL = BENCHMARK_EXPERIMENT_SPLIT_VERSION
ROLES = ("optimize", "shadow", "validation", "test")
SEED = 20261001
COUNTS = {b: dict(zip(ROLES, (150, 300, 300, 294 if b == "ifbench" else 300)))
          for b in CURRENT_RESEARCH_BENCHMARK_SUITE}
ACCESS = {"optimize": ["responsibility", "target", "pattern", "memory", "gepa", "team_probe", "full"],
          "shadow": "ADAPTIVE_GATE_ONLY_SCORE_PASS_FAIL", "validation": "POST_FREEZE_READ_ONLY_NEVER_ADAPTIVE",
          "test": "SEALED_REQUIRES_SEPARATE_AUTHORIZATION"}


def hash_order(rows, prefix):
    return sorted(rows, key=lambda r: (hashlib.sha256((str(prefix) + r["stable_example_id"]).encode()).hexdigest(), r["stable_example_id"]))


def subject(row):
    value = row["content"]["type"].lower().replace(" ", "_").replace("&", "and")
    if value not in SOURCE_PINS["math"]["subject_configs"]:
        raise ValueError("MATH_SUBJECT_UNKNOWN")
    return value


def quotas(population, count):
    """Integer largest remainder against original source proportions."""
    total = sum(population.values())
    if total < count:
        raise ValueError("EXPERIMENT_SOURCE_TOO_SMALL")
    result = {k: count * n // total for k, n in population.items()}
    ordered = sorted(population, key=lambda k: (-(count * population[k] % total), k))
    for key in ordered[:count - sum(result.values())]:
        result[key] += 1
    return result


def select_groups(benchmark, sources, legacy_members=(), *, seed=SEED):
    if benchmark not in COUNTS or seed != SEED:
        raise ValueError("EXPERIMENT_SPLIT_POLICY_MISMATCH")
    counts = COUNTS[benchmark]
    if benchmark == "math":
        if {s: len(sources[s]) for s in ("train", "test")} != {"train": 7500, "test": 5000}:
            raise ValueError("STOP_MATH_SOURCE_COUNT_MISMATCH")
        return stratified_math_groups(sources, counts, seed=seed)
    if benchmark == "hotpotqa":
        train, test = hash_order(sources["train"], seed), hash_order(sources["validation"], seed)
        return {"optimize": train[:150], "shadow": train[150:450], "validation": train[450:750], "test": test[:300]}
    lookup = {r["stable_example_id"]: r for rows in sources.values() for r in rows}
    groups = {role: [lookup[r["stable_example_id"]] for r in legacy_members
                    if r["project_split"] == ("search" if role == "optimize" else role)]
              for role in ("optimize", "shadow", "test")}
    used = {r["stable_example_id"] for role in ("optimize", "shadow") for r in groups[role]}
    groups["validation"] = hash_order([r for r in sources["train"] if r["stable_example_id"] not in used], "ifbench_validation_v1")[:300]
    return {role: groups[role] for role in ROLES}


def stratified_math_groups(sources, counts, *, seed=SEED):
    """Unchanged proportional algorithm; caller supplies its versioned universe."""
    if seed != SEED or counts != COUNTS["math"]:
        raise ValueError("EXPERIMENT_SPLIT_POLICY_MISMATCH")
    groups = {}
    for source, roles in (("train", ROLES[:3]), ("test", ("test",))):
        pools = {s: hash_order([r for r in sources[source] if subject(r) == s], seed)
                 for s in SOURCE_PINS["math"]["subject_configs"]}
        if any(not pool for pool in pools.values()):
            raise ValueError("MATH_SUBJECT_MISSING")
        original = {s: len(pool) for s, pool in pools.items()}
        offsets = Counter()
        for role in roles:
            allocation = quotas(original, counts[role])
            groups[role] = []
            for s in sorted(pools):
                part = pools[s][offsets[s]:offsets[s] + allocation[s]]
                if len(part) != allocation[s]:
                    raise ValueError("MATH_SUBJECT_CAPACITY_EXHAUSTED")
                groups[role].extend(part)
                offsets[s] += allocation[s]
    return groups


def overlap_audit(groups):
    keys = ("stable_example_id", "content_sha256", "input_sha256")
    pairs = [{"roles": [a, b], "overlap_counts": {k: len({r[k] for r in groups[a]} & {r[k] for r in groups[b]}) for k in keys}}
             for a, b in combinations(ROLES, 2)]
    within = {role: {k: len(rows) - len({r[k] for r in rows}) for k in keys} for role, rows in groups.items()}
    ok = not any(n for pair in pairs for n in pair["overlap_counts"].values()) and not any(n for counts in within.values() for n in counts.values())
    return {"status": "DISJOINT" if ok else "STOP_SPLIT_OVERLAP", "pairwise": pairs, "within_role_duplicate_counts": within}


def distribution(benchmark, rows):
    if benchmark == "math":
        values = {"subject": [subject(r) for r in rows], "level": [r["content"]["level"] for r in rows]}
    elif benchmark == "hotpotqa":
        values = {k: [r["content"].get(k, "UNAVAILABLE") for r in rows] for k in ("type", "level")}
        values["supporting_fact_count"] = [len(r["content"]["supporting_facts"]["title"]) if isinstance(r["content"]["supporting_facts"], dict) else len(r["content"]["supporting_facts"]) for r in rows]
        values["context_paragraph_count"] = [len(r["content"]["context"]["title"]) if isinstance(r["content"]["context"], dict) else len(r["content"]["context"]) for r in rows]
    else:
        values = {"instruction_count": [len(r["content"]["instruction_id_list"]) for r in rows],
                  "instruction_type": [i for r in rows for i in r["content"]["instruction_id_list"]],
                  "checker_family": [i.split(":")[0] for r in rows for i in r["content"]["instruction_id_list"]]}
    return {k: dict(sorted(Counter(map(str, v)).items())) for k, v in values.items()}


def load_canonical(root, benchmark, expected_sha):
    path = root / "manifests" / (benchmark + ".json")
    if not expected_sha or file_hash(path) != expected_sha:
        raise ValueError("CANONICAL_MANIFEST_IDENTITY_MISMATCH")
    verify_only(root, benchmark)
    manifest = json.loads(path.read_bytes())
    sources = {i["name"]: read_jsonl(root / "raw" / benchmark / (i["name"] + ".jsonl"))
               for i in manifest["source"]["canonical_sources"]}
    return manifest, sources


def build_experiment_manifest(canonical_root, benchmark, *, expected_canonical_sha256, seed=SEED):
    legacy, sources = load_canonical(canonical_root, benchmark, expected_canonical_sha256)
    legacy_members = read_jsonl(canonical_root / "manifests" / legacy["membership_file"])
    groups = select_groups(benchmark, sources, legacy_members, seed=seed)
    if {role: len(rows) for role, rows in groups.items()} != COUNTS[benchmark]:
        raise ValueError("EXPERIMENT_SPLIT_COUNT_MISMATCH")
    audit = overlap_audit(groups)
    if audit["status"] != "DISJOINT":
        raise ValueError("STOP_SPLIT_OVERLAP")
    members = []
    for role in ROLES:
        for row in groups[role]:
            entry = membership(row, role)
            if benchmark == "hotpotqa":
                entry.update(question_sha256=row["input_sha256"], answer_content_sha256=digest(row["content"]["answer"]),
                             supporting_facts_sha256=digest(row["content"]["supporting_facts"]))
            members.append(entry)
    identity_hashes = {role: digest([r["stable_example_id"] for r in groups[role]]) for role in ROLES}
    parity = {}
    if benchmark == "ifbench":
        for role, old in (("optimize", "search"), ("shadow", "shadow"), ("test", "test")):
            if identity_hashes[role] != legacy["hashes"][old]:
                raise ValueError("EXISTING_FROZEN_MEMBERSHIP_MISMATCH")
            parity[role] = {"unchanged": True, "legacy_hash": legacy["hashes"][old]}
    ids = jsonl_bytes(members)
    result = {"benchmark_id": benchmark, "protocol": PROTOCOL, "status": "FROZEN",
        "canonical_dataset_manifest_sha256": expected_canonical_sha256, "source_revision": legacy["source"]["revision"],
        "canonical_sources": legacy["source"]["canonical_sources"], "split_seed": seed,
        "split_algorithm": {"math": "original subject proportions; integer largest remainder; lexicographic subject ties; per-subject SHA256(str(seed)+stable_id); sequential disjoint role quotas",
            "hotpotqa": "SHA256(str(seed)+stable_id); train offsets 0:150/150:450/450:750; labeled validation first 300",
            "ifbench": "inherit search/shadow/test exactly; unused train SHA256(ifbench_validation_v1+stable_id) first 300"}[benchmark],
        "counts": COUNTS[benchmark], "hashes": identity_hashes,
        "membership_file": benchmark + ".ids.jsonl", "membership_sha256": hashlib.sha256(ids).hexdigest(),
        "role_membership_sha256": {role: hashlib.sha256(jsonl_bytes([r for r in members if r["project_split"] == role])).hexdigest() for role in ROLES},
        "pairwise_overlap_audit": audit, "access_policy": ACCESS,
        "project_test_source": "official_validation" if benchmark == "hotpotqa" else "official_test",
        "exactness": "PROJECT_PROPOSED", "ifbench_existing_membership_parity": parity,
        "legacy_membership_sha256": legacy["membership_sha256"],
        "distributions": {role: distribution(benchmark, rows) for role, rows in groups.items()}}
    result.update({role + "_membership_hash": identity_hashes[role] for role in ROLES})
    return result, ids


def freeze_experiment_split(canonical_root, destination, benchmark, *, expected_canonical_sha256):
    manifest, ids = build_experiment_manifest(canonical_root, benchmark, expected_canonical_sha256=expected_canonical_sha256)
    targets = {destination / (benchmark + ".json"): canonical(manifest) + b"\n", destination / manifest["membership_file"]: ids}
    for path, value in targets.items():
        if path.exists() and path.read_bytes() != value:
            raise ValueError("EXISTING_FROZEN_MEMBERSHIP_MISMATCH")
    for path, value in targets.items():
        immutable_write(path, value)
    return manifest


def verify_experiment_split(canonical_root, destination, benchmark, *, expected_manifest_sha256):
    path = destination / (benchmark + ".json")
    if file_hash(path) != expected_manifest_sha256:
        raise ValueError("EXPERIMENT_MANIFEST_IDENTITY_MISMATCH")
    observed = json.loads(path.read_bytes())
    expected, ids = build_experiment_manifest(canonical_root, benchmark,
        expected_canonical_sha256=observed["canonical_dataset_manifest_sha256"])
    if observed != expected or (destination / observed["membership_file"]).read_bytes() != ids:
        raise ValueError("EXPERIMENT_MEMBERSHIP_MISMATCH")
    return {"verified": True, "counts": expected["counts"], "overlap": expected["pairwise_overlap_audit"]["status"]}


class ExperimentSplitReader:
    """Role API exposes only selected canonical rows, with no held-out fallback."""
    def __init__(self, canonical_root: Path, destination: Path, benchmark: str, *, expected_manifest_sha256: str,
                 expected_protocol: str = PROTOCOL):
        if benchmark not in COUNTS:
            raise SearchContractError("BENCHMARK_UNKNOWN")
        self.root, self.benchmark = canonical_root, benchmark
        path = destination / (benchmark + ".json")
        if not expected_manifest_sha256 or file_hash(path) != expected_manifest_sha256:
            raise SearchContractError("EXPERIMENT_MANIFEST_IDENTITY_MISMATCH")
        self.manifest = json.loads(path.read_bytes())
        from ..versions import MATH_EXPERIMENT_SPLIT_VERSION, MATH_REFERENCE_VALIDITY_VERSION, MATH_REFERENCE_EXTRACTOR_VERSION
        if expected_protocol not in {PROTOCOL, MATH_EXPERIMENT_SPLIT_VERSION} or (expected_protocol != PROTOCOL and benchmark != "math"):
            raise SearchContractError("EXPERIMENT_SPLIT_POLICY_MISMATCH")
        if (self.manifest["benchmark_id"] != benchmark or self.manifest["protocol"] != expected_protocol
                or self.manifest["status"] != "FROZEN" or self.manifest["access_policy"] != ACCESS):
            raise SearchContractError("EXPERIMENT_SPLIT_POLICY_MISMATCH")
        if expected_protocol == MATH_EXPERIMENT_SPLIT_VERSION and (
                self.manifest.get("reference_validity_policy") != MATH_REFERENCE_VALIDITY_VERSION or
                self.manifest.get("reference_extractor") != MATH_REFERENCE_EXTRACTOR_VERSION or
                self.manifest.get("invalid_reference_counts_by_role") != dict.fromkeys(ROLES, 0)):
            raise SearchContractError("MATH_REFERENCE_POLICY_MISMATCH")
        canonical_path = canonical_root / "manifests" / (benchmark + ".json")
        if file_hash(canonical_path) != self.manifest["canonical_dataset_manifest_sha256"]:
            raise SearchContractError("CANONICAL_MANIFEST_IDENTITY_MISMATCH")
        member_path = destination / (benchmark + ".ids.jsonl")
        if self.manifest["membership_file"] != member_path.name or file_hash(member_path) != self.manifest["membership_sha256"]:
            raise SearchContractError("EXPERIMENT_MEMBERSHIP_MISMATCH")
        self.members = read_jsonl(member_path)

    def metadata(self, role):
        if role not in ROLES:
            raise SearchContractError("EXPERIMENT_SPLIT_ROLE_INVALID")
        return {"count": self.manifest["counts"][role], "membership_hash": self.manifest["hashes"][role]}

    def rows(self, role, purpose):
        if role not in ROLES or not isinstance(purpose, DataPurpose):
            raise SearchContractError("EXPERIMENT_SPLIT_ROLE_INVALID")
        if role in ("validation", "test"):
            raise SearchContractError("HELDOUT_SEARCH_ACCESS_FORBIDDEN")
        if (role == "shadow") != (purpose == DataPurpose.ADAPTIVE_GATE):
            raise SearchContractError("SHADOW_CONTENT_SEARCH_VISIBILITY_FORBIDDEN")
        selected = [r for r in self.members if r["project_split"] == role]
        result = {}
        for source_split in {r["source_split"] for r in selected}:
            identities = [i for i in self.manifest["canonical_sources"] if i["name"] == source_split]
            if len(identities) != 1 or source_split not in ("train", "test", "validation"):
                raise SearchContractError("CANONICAL_SOURCE_IDENTITY_MISMATCH")
            path = self.root / "raw" / self.benchmark / (source_split + ".jsonl")
            if file_hash(path) != identities[0]["canonical_sha256"]:
                raise SearchContractError("CANONICAL_SOURCE_HASH_MISMATCH")
            wanted = {r["source_index"]: r for r in selected if r["source_split"] == source_split}
            # Nonselected rows are never parsed or exposed to search/Pattern/Memory.
            with path.open("rb") as stream:
                for index, line in enumerate(stream):
                    if index not in wanted:
                        continue
                    row, expected = json.loads(line), wanted[index]
                    if any(row[k] != expected[k] for k in ("stable_example_id", "content_sha256", "input_sha256")):
                        raise SearchContractError("EXPERIMENT_SOURCE_ROW_MISMATCH")
                    if self.manifest["protocol"] == MATH_EXPERIMENT_SPLIT_VERSION and not str(row.get("reference_final_answer") or "").strip():
                        raise SearchContractError("REFERENCE_INVALID_NOT_SOLVER_WRONG")
                    result[row["stable_example_id"]] = row
        if len(result) != len(selected) or len(selected) != self.manifest["counts"][role]:
            raise SearchContractError("EXPERIMENT_SPLIT_COUNT_MISMATCH")
        return tuple(result[r["stable_example_id"]] for r in selected)


def optimize_to_legacy_search(role):
    if role != "optimize":
        raise SearchContractError("LEGACY_SPLIT_ADAPTER_OPTIMIZE_ONLY")
    return "search"


def bind_experiment_benchmark(canonical_root, destination, benchmark, *, expected_manifest_sha256):
    """Explicit offline data binding; no implicit mutation of replay registry."""
    verify_experiment_split(canonical_root, destination, benchmark,
                            expected_manifest_sha256=expected_manifest_sha256)
    canonical_manifest = json.loads((canonical_root / "manifests" / (benchmark + ".json")).read_bytes())
    from .registry import benchmark_spec
    return replace(benchmark_spec(benchmark), provenance_frozen=True, split_frozen=True,
                   dataset_identity=digest(canonical_manifest["source"]), split_identity=expected_manifest_sha256)
