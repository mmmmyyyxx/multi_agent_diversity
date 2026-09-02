"""Validation helpers for experiment, lineage, invariant, and failure registries."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .manifest import load_manifest, validate_manifest


def load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_experiment_registry(
    workspace: str | Path,
    registry: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    root = Path(workspace)
    errors: list[str] = []
    entries = registry.get("experiments", [])
    ids = [row.get("experiment_id") for row in entries]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate experiment IDs: {duplicates}")
    manifests: dict[str, dict[str, Any]] = {}
    known = set(ids)
    for row in entries:
        experiment_id = row.get("experiment_id")
        manifest_path = root / str(row.get("manifest", ""))
        report_path = row.get("report")
        if not manifest_path.is_file():
            errors.append(f"{experiment_id}: missing manifest {row.get('manifest')}")
            continue
        try:
            manifest = load_manifest(manifest_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"{experiment_id}: invalid manifest: {exc}")
            continue
        manifests[str(experiment_id)] = manifest
        if manifest.get("experiment_id") != experiment_id:
            errors.append(f"{experiment_id}: manifest ID mismatch")
        for error in validate_manifest(manifest, schema):
            errors.append(f"{experiment_id}: {error}")
        parent = row.get("parent")
        if parent not in {None, "UNKNOWN"} and parent not in known:
            errors.append(f"{experiment_id}: unknown parent {parent}")
        if report_path and not (root / report_path).exists():
            errors.append(f"{experiment_id}: missing report {report_path}")
    return errors, manifests


def validate_lineage(
    lineage: Mapping[str, Any], known_experiments: Iterable[str]
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    known = set(known_experiments)
    edges = lineage.get("edges", [])
    seen: set[tuple[str, str, str]] = set()
    graph: dict[str, set[str]] = defaultdict(set)
    indegree = {node: 0 for node in known}
    allowed = {"derived_experiment", "audit_of", "followup_of", "supersedes"}
    for row in edges:
        source, target, relation = row.get("from"), row.get("to"), row.get("relation")
        key = (source, target, relation)
        if key in seen:
            errors.append(f"duplicate lineage edge: {key}")
        seen.add(key)
        if source not in known:
            errors.append(f"unknown lineage parent: {source}")
        if target not in known:
            errors.append(f"unknown lineage child: {target}")
        if relation not in allowed:
            errors.append(f"unknown lineage relation: {relation}")
        if source in known and target in known and target not in graph[source]:
            graph[source].add(target)
            indegree[target] += 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in sorted(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(known):
        errors.append("lineage cycle detected")
    return errors, order


def render_lineage_mermaid(lineage: Mapping[str, Any]) -> str:
    lines = ["```mermaid", "graph TD"]
    for row in lineage.get("edges", []):
        source = row["from"]
        target = row["to"]
        relation = row["relation"]
        lines.append(f'  {source}["{source}"] -->|{relation}| {target}["{target}"]')
    lines.append("```")
    return "\n".join(lines)


def validate_invariants(
    workspace: str | Path,
    registry: Mapping[str, Any],
    known_failure_ids: Iterable[str],
) -> list[str]:
    root = Path(workspace)
    errors: list[str] = []
    rows = registry.get("invariants", [])
    ids = [row.get("id") for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate invariant IDs: {duplicates}")
    known_failures = set(known_failure_ids)
    for row in rows:
        invariant_id = row.get("id")
        if row.get("status") != "ACTIVE":
            continue
        authority = row.get("authority")
        if not authority or not (root / authority).is_file():
            errors.append(f"{invariant_id}: unknown specification authority {authority}")
        refs = row.get("implementation_refs", [])
        tests = row.get("tests", [])
        if not refs:
            errors.append(f"{invariant_id}: ACTIVE invariant has no implementation ref")
        for path in refs:
            if not (root / path).exists():
                errors.append(f"{invariant_id}: missing implementation ref {path}")
        if row.get("critical") and not tests:
            errors.append(f"{invariant_id}: critical invariant has no test")
        for path in tests:
            if not (root / path.split("::", 1)[0]).is_file():
                errors.append(f"{invariant_id}: missing test {path}")
        for failure_id in row.get("failure_refs", []):
            if failure_id not in known_failures:
                errors.append(f"{invariant_id}: unknown failure ID {failure_id}")
    return errors


def validate_failure_registry(
    workspace: str | Path, registry: Mapping[str, Any]
) -> list[str]:
    root = Path(workspace)
    errors: list[str] = []
    rows = registry.get("failures", [])
    ids = [row.get("failure_id") for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate failure IDs: {duplicates}")
    allowed_statuses = {
        "OPEN", "DIAGNOSED", "MITIGATED", "RESOLVED", "NOT_PRIMARY", "NOT_SUPPORTED"
    }
    allowed_evidence = {"observed", "causally_supported", "hypothesized"}
    required = {
        "failure_id", "title", "status", "first_observed", "affected_components",
        "symptom", "evidence", "root_cause_status", "forbidden_inference",
        "regression_tests", "mitigation", "evidence_level",
    }
    for row in rows:
        failure_id = row.get("failure_id")
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"{failure_id}: missing fields {missing}")
        if row.get("status") not in allowed_statuses:
            errors.append(f"{failure_id}: invalid status")
        if row.get("evidence_level") not in allowed_evidence:
            errors.append(f"{failure_id}: invalid evidence_level")
        for evidence in row.get("evidence", []):
            if not (root / evidence).exists():
                errors.append(f"{failure_id}: missing evidence {evidence}")
        for test in row.get("regression_tests", []):
            if not (root / test.split("::", 1)[0]).is_file():
                errors.append(f"{failure_id}: missing regression test {test}")
    return errors
