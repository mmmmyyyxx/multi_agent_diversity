# Codex Experiment Workflow

## Phase 1 — Read

Read `AGENTS.md`, `docs/design/CURRENT_SPEC.md`, relevant failure-registry
entries, and every parent experiment manifest.

## Phase 2 — Impact analysis

Before editing, state the affected invariant IDs, files, sole design delta,
frozen behavior, and failure IDs at risk.

## Phase 3 — Implement

Implement only the preregistered delta. Preserve historical artifacts. Freeze
the manifest and its preregistration hash before any API call.

## Phase 4 — Conformance

Run governance preflight, invariant checks, manifest/lineage checks, focused
regressions, full tests, compileall, deterministic replay, sanitization, and
`git diff --check`.

## Phase 5 — API

Enter only after explicit user authorization and a fail-closed manifest/phase/
role/budget/hash authorization check. Log sanitized API and split access events.

## Phase 6 — Freeze and evaluate

Advance through the manifest lifecycle without skipping states. Freeze
selection before validation and the final state before test. Protocol changes
require an amendment and, when needed, a fresh run rather than artifact edits.
