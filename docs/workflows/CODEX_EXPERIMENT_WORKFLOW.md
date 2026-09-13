# Codex Experiment Workflow

## Ownership rule

```text
Sol designs, codes, freezes, audits, interprets, and publishes.
Luna executes an already-frozen handoff, monitors, and reports facts.
```

Sol is the sole scientific and tracked-code owner. Luna is a scoped execution
subagent and must not span design, implementation, freeze, integrity audit, or
scientific interpretation. If explicitly requested Luna is unavailable, report
`LUNA_DISPATCH_UNAVAILABLE`; never silently substitute another Sol.

## Phase 1 — Read / Diagnose [Sol]

Read `AGENTS.md`, `docs/design/CURRENT_SPEC.md`, relevant failure-registry
entries, and every parent experiment manifest.

## Phase 2 — Design / Impact analysis [Sol]

Before editing, state the affected invariant IDs, files, sole design delta,
frozen behavior, and failure IDs at risk.

## Phase 3 — Implement [Sol]

Implement only the preregistered delta. Preserve historical artifacts. Freeze
the manifest and its preregistration hash before any API call.

## Phase 4 — Conformance [Sol; Luna may run bounded commands]

Run governance preflight, invariant checks, manifest/lineage checks, focused
regressions, full tests, compileall, deterministic replay, sanitization, and
`git diff --check`. Sol owns diagnoses and fixes. Luna may execute only commands
specified by Sol and returns factual results without editing tracked files.

## Phase 5 — Freeze / Handoff [Sol]

Create a concrete handoff conforming to `EXPERIMENT_HANDOFF.md`. Sol alone may
set `READY_TO_RUN=true`, after verifying the source commit, clean tracked
worktree, protocol/preregistration/manifest hashes, split identity, model and
budget, exact command, output roots, Validation/Test policy, and explicit API
authorization.

## Phase 6 — Execute frozen experiment [Luna]

Explicitly dispatch `gpt-5.6-luna` when available. Luna independently verifies
the handoff, then executes exactly the frozen command and frozen stop rules.
It monitors operational facts only. A mismatch or unexpected condition fails
closed and returns preserved evidence to Sol; Luna does not patch or resume.

## Phase 7 — Integrity audit [Sol]

Sol audits protocol, artifacts, ledger arithmetic, split access, source hashes,
budgets, retries, checkpoints, and runner termination. Execution completion is
not scientific validity.

## Phase 8 — Scientific analysis [Sol]

Sol performs funnel analysis, causal interpretation, scientific classification,
and report integration. Luna's factual summary is evidence, not a conclusion.

## Phase 9 — Publish artifacts/code [Sol, explicit authorization only]

Advance through the manifest lifecycle without skipping states. Freeze
selection before validation and the final state before test. Protocol changes
require an amendment and, when needed, a fresh run rather than artifact edits.
Only Sol commits. Sol pushes only when the user explicitly authorizes it.
