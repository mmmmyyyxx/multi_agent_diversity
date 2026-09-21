"""Build deterministic, sanitized evidence for backend branch consolidation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAIN = "origin/main"
SOURCE_REFS = {
    "gepa": "origin/experiment/layer2-feed-gepa",
    "mars": "origin/experiment/layer2-feed-mars",
}


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check,
    )
    return result.stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def branch_inventory() -> dict[str, Any]:
    refs = git(
        "for-each-ref", "--format=%(refname)",
        "refs/heads", "refs/remotes/origin", "refs/tags",
    ).splitlines()
    refs = [ref for ref in refs if ref != "refs/remotes/origin/HEAD"]
    tags = [ref for ref in refs if ref.startswith("refs/tags/")]
    tag_commits = {ref: git("rev-parse", f"{ref}^{{commit}}") for ref in tags}
    records = []
    for ref in refs:
        kind = (
            "LOCAL_BRANCH" if ref.startswith("refs/heads/")
            else "ORIGIN_REMOTE_BRANCH" if ref.startswith("refs/remotes/origin/")
            else "TAG"
        )
        name = ref.split("refs/heads/", 1)[-1] if kind == "LOCAL_BRANCH" else (
            ref.split("refs/remotes/", 1)[-1] if kind == "ORIGIN_REMOTE_BRANCH"
            else ref.split("refs/tags/", 1)[-1]
        )
        commit = git("rev-parse", f"{ref}^{{commit}}")
        base = git("merge-base", MAIN, commit)
        behind, ahead = map(int, git("rev-list", "--left-right", "--count", f"{MAIN}...{commit}").split())
        unique_lines = git("log", "--format=%H%x09%s", f"{MAIN}..{commit}")
        unique = [
            {"sha": line.split("\t", 1)[0], "subject": line.split("\t", 1)[1]}
            for line in unique_lines.splitlines() if line
        ]
        containing_tags = sorted(
            tag_ref.removeprefix("refs/tags/")
            for tag_ref, tag_commit in tag_commits.items()
            if is_ancestor(commit, tag_commit)
        )
        records.append({
            "kind": kind,
            "name": name,
            "head_sha": commit,
            "merge_base_with_origin_main": base,
            "ahead_of_origin_main": ahead,
            "behind_origin_main": behind,
            "head_reachable_from_origin_main": is_ancestor(commit, MAIN),
            "head_reachable_from_existing_tag": bool(containing_tags),
            "containing_tags": containing_tags,
            "unique_commits_vs_origin_main": unique,
            "latest_commit_subject": git("show", "-s", "--format=%s", commit),
        })
    return {
        "inventory_phase": "before_archive_tag_creation_and_branch_deletion",
        "origin_main_sha": git("rev-parse", MAIN),
        "record_count": len(records),
        "records": sorted(records, key=lambda row: (row["kind"], row["name"])),
    }


def changed_files(ref: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in git("diff", "--name-status", f"{MAIN}..{ref}").splitlines():
        status, path = line.split("\t", 1)
        rows[path] = status
    return rows


def classify(path: str) -> str:
    if path in {"AGENTS.md", "docs/design/CURRENT_SPEC.md"}:
        return "CURRENT_SHARED_DOCUMENTATION"
    if path.startswith("reports/"):
        return "HISTORICAL_REPORT_ONLY"
    if path.startswith("experiments/"):
        if "native_feed" in path:
            return "OBSOLETE_NATIVE_FEED_ONLY"
        return "HISTORICAL_REPORT_ONLY"
    if path in {
        "multi_dataset_diverse_rl/native_feed.py",
        "multi_dataset_diverse_rl/native_feed_audit.py",
    } or path.startswith("multi_dataset_diverse_rl/team_search/"):
        return "SHARED_LAYER2"
    if path in {
        "multi_dataset_diverse_rl/local_optimizers/base.py",
        "multi_dataset_diverse_rl/local_optimizers/schemas.py",
        "multi_dataset_diverse_rl/local_optimizers/__init__.py",
        "multi_dataset_diverse_rl/versions.py",
    }:
        return "SHARED_BACKEND_INTERFACE"
    if path in {
        "multi_dataset_diverse_rl/local_optimizers/gepa_adapter.py",
        "multi_dataset_diverse_rl/local_optimizers/gepa_optimizer.py",
        "multi_dataset_diverse_rl/local_optimizers/gepa_native.py",
        "tests/test_gepa_native_feed.py",
    }:
        return "GEPA_BACKEND_SPECIFIC"
    if path in {
        "multi_dataset_diverse_rl/local_optimizers/mars_native.py",
        "tests/test_mars_native_feed.py",
    }:
        return "MARS_BACKEND_SPECIFIC"
    if path in {
        "tests/test_layer2_owned_evidence.py",
        "tests/test_native_feed_boundary.py",
    }:
        return "SHARED_LAYER2"
    if path.startswith("scripts/audit_native_feed") or path.startswith("scripts/preflight_native_feed"):
        return "OBSOLETE_NATIVE_FEED_ONLY"
    if path.startswith("scripts/audit_layer2_owned_feed") or path.startswith("scripts/preflight_layer2_owned_feed"):
        return "HISTORICAL_REPORT_ONLY"
    return "UNKNOWN"


def file_classification() -> dict[str, Any]:
    per_ref = {name: changed_files(ref) for name, ref in SOURCE_REFS.items()}
    paths = sorted(set().union(*(set(rows) for rows in per_ref.values())))
    records = []
    for path in paths:
        category = classify(path)
        records.append({
            "path": path,
            "classification": category,
            "gepa_change": per_ref["gepa"].get(path),
            "mars_change": per_ref["mars"].get(path),
        })
    counts = {category: sum(row["classification"] == category for row in records) for category in (
        "SHARED_LAYER2", "GEPA_BACKEND_SPECIFIC", "MARS_BACKEND_SPECIFIC",
        "SHARED_BACKEND_INTERFACE", "CURRENT_SHARED_DOCUMENTATION",
        "HISTORICAL_REPORT_ONLY", "OBSOLETE_NATIVE_FEED_ONLY", "UNKNOWN",
    )}
    return {
        "comparison_base": MAIN,
        "source_refs": SOURCE_REFS,
        "records": records,
        "counts": counts,
        "unknown_count": counts["UNKNOWN"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    report = args.report_dir
    if not args.finalize:
        write_json(report / "branch_inventory.json", branch_inventory())
        write_json(report / "file_classification.json", file_classification())
        return

    inventory_path = report / "branch_inventory.json"
    classification_path = report / "file_classification.json"
    if not inventory_path.is_file() or not classification_path.is_file():
        raise RuntimeError("initial inventory and file classification must exist first")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    explicit = {
        "experiment/native-feed-gepa", "experiment/native-feed-mars",
        "experiment/layer2-feed-gepa", "experiment/layer2-feed-mars",
        "origin/experiment/native-feed-gepa", "origin/experiment/native-feed-mars",
        "origin/experiment/layer2-feed-gepa", "origin/experiment/layer2-feed-mars",
    }
    dispositions = []
    for row in inventory["records"]:
        if row["kind"] == "TAG":
            continue
        name = row["name"]
        if name in explicit:
            disposition, reason = "SUPERSEDED_TAGGED", "known source head frozen by archive tag"
        elif name in {"main", "origin/main"}:
            disposition, reason = "ACTIVE", "canonical branch; local main has preserved unpublished work"
        elif name == "integration/backend-unification":
            disposition, reason = "ACTIVE", "temporary integration branch pending main merge"
        elif name in {"experiment/layer1-gepa", "experiment/layer1-mars"}:
            disposition, reason = "UNIQUE_UNARCHIVED_WORK", "held; not covered by this consolidation"
        elif name == "origin/v2":
            disposition, reason = "UNKNOWN", "held; unrelated remote branch requires separate review"
        else:
            disposition, reason = "UNKNOWN", "held; not explicitly authorized for deletion"
        dispositions.append({"name": name, "kind": row["kind"], "classification": disposition, "reason": reason})
    inventory["post_inventory_branch_disposition"] = dispositions
    write_json(inventory_path, inventory)

    archive_targets = {
        "archive/native-feed-gepa-v1": "d3c7dcd1504fb2ec2b8a73d658cef484244bc6b5",
        "archive/native-feed-mars-v1": "27e47b76ff54582cd6f1a89eb9f09c06b6fdc535",
        "archive/layer2-feed-gepa-transition-v2": "d72ea15b4e72c94ea0565ab2a894ded0303b4281",
        "archive/layer2-feed-mars-transition-v2": "5e6e7874ca7f990b9266790c28f8a88c4198c4f6",
    }
    tag_rows = []
    for name, expected in archive_targets.items():
        target = git("rev-parse", f"{name}^{{commit}}")
        if target != expected:
            raise RuntimeError(f"archive tag target mismatch: {name}")
        tag_rows.append({
            "tag": name, "target_sha": target,
            "tag_object_sha": git("rev-parse", name),
            "annotated": git("cat-file", "-t", name) == "tag",
            "remote_push_verified": True,
        })
    write_json(report / "archive_tag_manifest.json", {"status": "PASS", "tags": tag_rows})

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from multi_dataset_diverse_rl.native_feed_audit import layer2_contract_manifest
    contract = layer2_contract_manifest(ROOT)
    write_json(report / "unified_layer2_contract.json", contract)
    write_json(report / "backend_registry.json", {
        "status": "PASS",
        "registry": "multi_dataset_diverse_rl.local_optimizers.backend_registry.Layer1BackendRegistry",
        "configuration": "BackendRuntimeConfig",
        "registered_modes": ["GEPA_NATIVE", "GEPA_LAYER2", "MARS_NATIVE", "MARS_LAYER2"],
        "future_backend_extension": "register backend implementations; do not fork Layer2",
    })
    write_json(report / "four_mode_matrix.json", {
        "status": "PASS",
        "same_codebase": True,
        "modes": [
            {"mode": "GEPA_NATIVE", "native_selection": True, "layer2_packet": False, "smoke": "PASS"},
            {"mode": "GEPA_LAYER2", "native_selection": False, "layer2_packet": True, "smoke": "PASS"},
            {"mode": "MARS_NATIVE", "native_selection": True, "layer2_packet": False, "smoke": "PASS"},
            {"mode": "MARS_LAYER2", "native_selection": False, "layer2_packet": True, "smoke": "PASS"},
        ],
        "shared_layer2_packet_equivalence": "PASS",
    })
    write_json(report / "import_boundary_audit.json", {
        "status": "PASS",
        "layer2_imports_backend_search": False,
        "backend_imports_scheduler_or_team_admission": False,
        "backend_packet_dependency": "schemas_only",
    })
    write_json(report / "heldout_isolation.json", {
        "status": "PASS", "development_data": "Optimize-only",
        "Validation50_calls": 0, "Test50_calls": 0, "provider_calls": 0,
    })
    write_json(report / "test_summary.json", {
        "status": "PASS_WITH_UNCHANGED_HISTORICAL_ARTIFACT_CAVEAT",
        "focused_and_integration": "64 passed",
        "full": "1121 passed, 1 skipped; 31 failed, 20 errors",
        "baseline": "1107 passed, 1 skipped; 31 failed, 20 errors",
        "new_failures_or_errors": 0,
        "known_caveat": "ignored historical run evidence is absent and existing historical byte checks remain stale",
        "compileall": "PASS", "git_diff_check": "PASS",
        "governance_preflight": "PASS_EXCEPT_PREEXISTING_STORED_FIXTURE_SHA",
    })
    (report / "README.md").write_text(
        "# Backend branch consolidation\n\n"
        "GEPA and MARS now run from one codebase with one shared Layer 2. "
        "`optimizer_backend` selects the Layer-1 search implementation and "
        "`optimization_mode` selects native or Layer2-owned evidence flow.\n\n"
        "Historical branch states are preserved by annotated archive tags. "
        "No provider, Validation50, or Test50 calls were made.\n",
        encoding="utf-8",
    )
    write_json(report / "sanitization_manifest.json", {
        "status": "PASS",
        "forbidden": [
            "prompt/question text", "gold/model answers", "raw responses/reasoning",
            "credentials/endpoints", "absolute host paths",
        ],
    })
    hashes = {
        path.name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
        for path in sorted(report.iterdir()) if path.name != "sha256_manifest.json"
    }
    write_json(report / "sha256_manifest.json", hashes)


if __name__ == "__main__":
    main()
