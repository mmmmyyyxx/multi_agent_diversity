"""Reproducible local benchmark data, immutable memberships and offline audit.

This module imports neither providers nor optimizer code. Download construction
is kept in data_sources; verification uses only the standard library.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
from typing import Any

BENCHMARK_IDS = ("hotpotqa", "hover", "ifbench", "pupa", "math")
GROUPS = ("search", "shadow", "validation", "test")
PROTOCOL = "benchmark_data_freeze_v1"
GEPA_REVISION = "cbefbc1aa0f43dd39874ec4bf42211365dbda42e"
SOURCE_PINS = {
    "hotpotqa": {"repository": "hotpotqa/hotpot_qa", "revision": "1908d6afbbead072334abe2965f91bd2709910ab", "config": "fullwiki"},
    "hover": {"repository": "hover-nlp/hover", "revision": "c0e43052759879b3461642ca6c0dd26658f47691", "config": "default",
              "data_repository": "hover-nlp/hover", "data_revision": "39b84697f196308f398a251a7aea9b82ae0f0562"},
    "ifbench": {"repository": "gepa-ai/gepa-artifact", "revision": GEPA_REVISION, "config": "vendored_IFBench"},
    "pupa": {"repository": "Columbia-NLP/PUPA", "revision": "9981b49b6ced0033988a224b6712895ebf119294", "config": "pupa_new", "auxiliary_config": "pupa_tnb"},
    "math": {"repository": "EleutherAI/hendrycks_math", "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3", "config": "canonical_seven_subjects",
             "subject_configs": ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]},
}
SOURCE_FIELDS = {
    "hotpotqa": {"question", "answer", "context", "supporting_facts"},
    "hover": {"uid", "claim", "supporting_facts", "label", "num_hops", "hpqa_id"},
    "ifbench": {"prompt", "instruction_id_list", "kwargs", "key"},
    "pupa": {"user_query", "pii_units", "target_response"},
    "math": {"problem", "solution", "type", "level"},
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical(row) + b"\n" for row in rows)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def immutable_write(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("FROZEN_ARTIFACT_MISMATCH: " + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)


def gepa_trim(pool: list, size: int) -> list:
    return pool[:] if size >= len(pool) else random.Random(1).sample(pool, size)


def gepa_split(rows: list[dict]) -> dict[str, list[dict]]:
    n = len(rows)
    return {"search": gepa_trim(rows[int(.8 * n):], 150),
            "shadow": gepa_trim(rows[int(.4 * n):int(.8 * n)], 300),
            "validation": [], "test": gepa_trim(rows[:int(.4 * n)], 300)}


def hover_three_hop(rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if len({fact["key"] for fact in row["content"]["supporting_facts"]}) == 3]
    random.Random(0).shuffle(selected)
    return selected


def select_splits(benchmark: str, sources: dict[str, list[dict]]) -> dict[str, list[dict]]:
    if benchmark in {"hotpotqa", "hover"}:
        rows = sources["train"]
        return gepa_split(hover_three_hop(rows) if benchmark == "hover" else rows)
    if benchmark == "ifbench":
        rows = sources["train"]
        return {"search": gepa_trim(rows[300:600], 150), "shadow": rows[:300], "validation": [], "test": sources["test"][:]}
    if benchmark == "pupa":
        rows = sources["train"]
        return {"search": rows[:111], "shadow": rows[111:222], "validation": [], "test": rows[222:443]}
    if benchmark == "math":
        train = sources["train"][:]
        test = sources["test"][:]
        random.Random(1).shuffle(train)
        random.Random(1).shuffle(test)
        return {"search": train[:150], "shadow": train[150:450], "validation": [], "test": test[:300]}
    raise ValueError("BENCHMARK_UNKNOWN")


def reference_answer(solution: str) -> str | None:
    """Last balanced boxed/fbox payload. Extraction only, never equivalence."""
    start = max(solution.rfind("\\boxed{"), solution.rfind("\\fbox{"))
    if start < 0:
        return None
    left = solution.index("{", start)
    depth = 1
    for pos in range(left + 1, len(solution)):
        if solution[pos] == "{" and (pos == 0 or solution[pos - 1] != "\\"):
            depth += 1
        elif solution[pos] == "}" and (pos == 0 or solution[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return solution[left + 1:pos]
    return None


def wrap_rows(benchmark: str, split: str, rows: list[dict], official_ids: list[str] | None = None) -> list[dict]:
    pin = SOURCE_PINS[benchmark]
    wrapped = []
    for index, row in enumerate(rows):
        if not SOURCE_FIELDS[benchmark].issubset(row):
            raise ValueError("SOURCE_SCHEMA_MISMATCH: " + benchmark)
        sha = digest(row)
        official = (official_ids[index] if official_ids else
                    next((str(row[k]) for k in ("uid", "id", "_id", "key") if row.get(k) is not None), None))
        if benchmark == "pupa":
            official = hashlib.sha256(official.encode()).hexdigest() if official is not None else None
        stable = f"{benchmark}:{split}:{official}" if official is not None else f"{benchmark}:{split}:{index}:{sha}"
        text_key = {"hotpotqa": "question", "hover": "claim", "ifbench": "prompt", "pupa": "user_query", "math": "problem"}[benchmark]
        if not isinstance(row.get(text_key), str):
            raise ValueError("SOURCE_SCHEMA_MISMATCH: " + benchmark)
        item = {"benchmark_id": benchmark, "source_dataset": pin["repository"], "source_revision": pin["revision"],
                "source_split": split, "source_index": index, "stable_example_id": stable,
                "content_sha256": sha, "input_sha256": hashlib.sha256(row[text_key].encode()).hexdigest(), "content": row}
        if benchmark == "math":
            item["reference_final_answer"] = reference_answer(row["solution"])
        wrapped.append(item)
    return wrapped


def membership(row: dict, group: str) -> dict:
    safe = {key: row[key] for key in ("stable_example_id", "source_split", "source_index", "content_sha256", "input_sha256")}
    safe["project_split"] = group
    if row["benchmark_id"] == "hover":
        safe["supporting_facts_sha256"] = digest(row["content"]["supporting_facts"])
    if row["benchmark_id"] == "pupa":
        safe["pii_count"] = len(set(str(row["content"].get("pii_units", "")).split("||")))
    return safe


def isolation_audit(groups: dict[str, list[dict]]) -> dict:
    checks = []
    for a, b in combinations(GROUPS, 2):
        counts = {key: len({r[key] for r in groups[a]} & {r[key] for r in groups[b]})
                  for key in ("stable_example_id", "content_sha256", "input_sha256")}
        checks.append({"groups": [a, b], "overlap_counts": counts})
    within = {group: {key: len(rows) - len({r[key] for r in rows})
                      for key in ("stable_example_id", "content_sha256", "input_sha256")}
              for group, rows in groups.items()}
    duplicate = any(n for check in checks for n in check["overlap_counts"].values()) or any(n for counts in within.values() for n in counts.values())
    return {"pairwise": checks, "within_split_duplicate_counts": within,
            "status": "SOURCE_DUPLICATION_DETECTED" if duplicate else "DISJOINT",
            "FORMAL_READY": "NO" if duplicate else "DATA_ISOLATION_ONLY"}


def distribution(rows: list[dict]) -> dict:
    return {key: dict(sorted(Counter(row["content"][key] for row in rows).items())) for key in ("type", "level")}


def build_manifest(root: Path, benchmark: str, sources: dict[str, list[dict]], metadata: dict) -> tuple[dict, bytes]:
    groups = select_splits(benchmark, sources)
    ids = [membership(row, group) for group in GROUPS for row in groups[group]]
    exactness = "EXACT_SOURCE_AND_INSTANCE_ALIGNED" if benchmark == "ifbench" else "SOURCE_LOGIC_REPRODUCED"
    # GEPA's unpinned load_dataset revision cannot prove original PUPA bytes.
    if benchmark == "math":
        exactness = "PROJECT_PROPOSED"
    algorithms = {
        "hotpotqa": "source train order; test[:int(.4*N)]; shadow[int(.4*N):int(.8*N)]; search[int(.8*N):]; independently Random(1).sample pools",
        "hover": "unique supporting-fact keys == 3; Random(0).shuffle; GEPA .4/.8 pools; independently Random(1).sample",
        "ifbench": "shadow=train[:300]; search=Random(1).sample(train[300:600],150); test=entire vendored test",
        "pupa": "pupa_new train original order; search[:111]; shadow[111:222]; test[222:443]; no shuffle",
        "math": "PROJECT_MATH_V1_PROPOSED; independent Random(1).shuffle train/test indexes; search=train[:150]; shadow=train[150:450]; test=test[:300]",
    }
    raw_identity = [{"name": name, "row_count": len(rows), "canonical_sha256": hashlib.sha256(jsonl_bytes(rows)).hexdigest()}
                    for name, rows in sources.items()]
    manifest = {"schema_version": 1, "benchmark_id": benchmark, "protocol": PROTOCOL,
                "source": {**SOURCE_PINS[benchmark], **metadata, "canonical_sources": raw_identity},
                "selection": {"source_protocol": "PROJECT" if benchmark == "math" else "GEPA", "gepa_artifact_revision": GEPA_REVISION,
                              "algorithm": algorithms[benchmark], "seed": 1, "pre_shuffle_seed": 0 if benchmark == "hover" else None},
                "counts": {g: len(groups[g]) for g in GROUPS},
                "hashes": {g: digest([row["stable_example_id"] for row in groups[g]]) for g in GROUPS},
                "membership_file": benchmark + ".ids.jsonl", "membership_sha256": hashlib.sha256(jsonl_bytes(ids)).hexdigest(),
                "materialized": {g: {"path": f"materialized/{benchmark}/{g}.jsonl", "sha256": hashlib.sha256(jsonl_bytes(groups[g])).hexdigest(),
                                     "bytes": len(jsonl_bytes(groups[g]))} for g in GROUPS},
                "split_exactness": exactness, "status": "FROZEN", "isolation": isolation_audit(groups),
                "access_policy": {"search_read": ["search"], "adaptive_evaluator_read": ["shadow"],
                                  "independent_validation": "EMPTY", "test": "SEALED", "shadow_search_visibility": "SCORE_PASS_FAIL_ONLY"}}
    if benchmark == "hover":
        manifest["filtered_three_hop_count"] = len(hover_three_hop(sources["train"]))
    if benchmark == "math":
        manifest["agentgrad_status"] = "AGENTGRAD_MATH_EXACT_SPLIT_UNRESOLVED"
        manifest["selection_rule"] = "PROJECT_MATH_V1_PROPOSED"
        manifest["distributions"] = {**{f"source_{s}": distribution(rows) for s, rows in sources.items()},
                                     **{g: distribution(rows) for g, rows in groups.items() if rows}}
        manifest["reference_extraction"] = {s: {"missing": sum(r["reference_final_answer"] is None for r in rows), "total": len(rows)} for s, rows in sources.items()}
    return manifest, jsonl_bytes(ids)


def freeze_local(root: Path, benchmark: str, sources: dict[str, list[dict]], metadata: dict) -> dict:
    manifest, ids = build_manifest(root, benchmark, sources, metadata)
    targets = {root / "manifests" / (benchmark + ".json"): canonical(manifest) + b"\n",
               root / "manifests" / (benchmark + ".ids.jsonl"): ids}
    for name, rows in sources.items():
        targets[root / "raw" / benchmark / (name + ".jsonl")] = jsonl_bytes(rows)
    for group, rows in select_splits(benchmark, sources).items():
        targets[root / "materialized" / benchmark / (group + ".jsonl")] = jsonl_bytes(rows)
    # Compare all existing files before writing any output.
    for path, data in targets.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError("FROZEN_ARTIFACT_MISMATCH: " + path.name)
    for path, data in targets.items():
        immutable_write(path, data)
    return manifest


def verify_only(root: Path, benchmark: str) -> dict:
    """No downloads, datasets import, cache access or network fallbacks."""
    path = root / "manifests" / (benchmark + ".json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN" or manifest["protocol"] != PROTOCOL or manifest["benchmark_id"] != benchmark:
        raise ValueError("MANIFEST_IDENTITY_MISMATCH")
    for key, value in SOURCE_PINS[benchmark].items():
        if manifest["source"][key] != value:
            raise ValueError("SOURCE_REVISION_MISMATCH")
    sources = {}
    for item in manifest["source"]["canonical_sources"]:
        raw_path = root / "raw" / benchmark / (item["name"] + ".jsonl")
        if file_hash(raw_path) != item["canonical_sha256"]:
            raise ValueError("SOURCE_HASH_MISMATCH")
        rows = read_jsonl(raw_path)
        if len(rows) != item["row_count"]:
            raise ValueError("SOURCE_COUNT_MISMATCH")
        for i, row in enumerate(rows):
            if (row["benchmark_id"] != benchmark or row["source_index"] != i or row["source_split"] != item["name"]
                    or row["source_dataset"] != SOURCE_PINS[benchmark]["repository"]
                    or not SOURCE_FIELDS[benchmark].issubset(row["content"])
                    or row["source_revision"] != SOURCE_PINS[benchmark]["revision"] or digest(row["content"]) != row["content_sha256"]):
                raise ValueError("SOURCE_SCHEMA_MISMATCH")
        sources[item["name"]] = rows
    for item in manifest["source"]["files"]:
        if file_hash(root / "raw" / benchmark / item["local_name"]) != item["sha256"]:
            raise ValueError("RAW_BYTES_MISMATCH")
    metadata = {k: v for k, v in manifest["source"].items() if k not in SOURCE_PINS[benchmark] and k != "canonical_sources"}
    expected, ids = build_manifest(root, benchmark, sources, metadata)
    if expected != manifest or (root / "manifests" / manifest["membership_file"]).read_bytes() != ids:
        raise ValueError("MANIFEST_MEMBERSHIP_MISMATCH")
    groups = select_splits(benchmark, sources)
    for group, item in manifest["materialized"].items():
        materialized = root / item["path"]
        if file_hash(materialized) != item["sha256"] or materialized.read_bytes() != jsonl_bytes(groups[group]):
            raise ValueError("MATERIALIZED_SPLIT_MISMATCH")
    return {"benchmark_id": benchmark, "verified": True, "counts": manifest["counts"], "isolation": manifest["isolation"]["status"], "network_attempts": 0, "provider_attempts": 0}


def freeze_suite(root: Path) -> dict:
    manifests = {b: json.loads((root / "manifests" / (b + ".json")).read_text(encoding="utf-8")) for b in BENCHMARK_IDS}
    suite = {"protocol": PROTOCOL, "benchmark_manifest_sha256": {b: file_hash(root / "manifests" / (b + ".json")) for b in BENCHMARK_IDS},
             "data_frozen": True, "real_execution_ready": False, "independent_validation": "EMPTY",
             "test_policy": "SEALED", "readiness": {b: {"DATA_READY": True, "SPLIT_READY": True,
                 "FORMAL_READY": False, "UNIFIED_SEARCH_READY": False, "exactness": m["split_exactness"]} for b, m in manifests.items()}}
    immutable_write(root / "manifests" / "suite.json", canonical(suite) + b"\n")
    return suite
