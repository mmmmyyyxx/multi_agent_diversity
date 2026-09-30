# AGENTS.md

Before making changes:
1. Read AGENTS.md.
2. Read docs/design/CURRENT_SPEC.md.
3. Read experiments/current_frontier.yaml.
4. Read the relevant benchmark contract.
5. Read the parent manifest.
6. Check docs/failures/registry.yaml.

## 1. Purpose

This is the stable engineering and safety contract. Keep dynamic commits,
results, readiness, blockers, seeds, host paths, credentials, endpoints and
model availability in manifests, registries or evidence, never here.

## 2. Source-of-truth hierarchy

Use the narrowest authority for each fact:

| Authority | Scope |
|---|---|
| AGENTS.md | Stable engineering, agent workflow and safety |
| docs/design/CURRENT_SPEC.md | Active scientific method |
| multi_dataset_diverse_rl/versions.py | Runtime identities and constants |
| multi_dataset_diverse_rl/benchmarks/ | Benchmark scientific and data contracts |
| experiments/registry.yaml | Experiment metadata |
| experiments/lineage.yaml | Experiment lineage |
| Frozen experiment manifest | One experiment's preregistration |
| reports/ | Immutable evidence |

Reports are evidence, never design authority. Historical claims of current,
active or canonical refer to the report's creation context.

## 3. Current project direction

The sole active research architecture is Unified Team Prompt Search.
`UnifiedSearchOrchestrator` owns the production graph. GEPA is the current
replaceable SearchEngine; aggregation is a benchmark-selected policy.
Pattern and Memory default to null and require a versioned experiment to
activate. Structured optimizer history is separate from LLM memory.
New experiments use `scripts/run_experiment.py`, the sole current composition
entrypoint. Domain logic belongs in production modules, not scripts.
Shared compatibility physics and historical replay have explicit classifications
in docs/REPOSITORY_MAP.md; they cannot determine new experiment identity.

## 4. Agent workflow

## Agent execution model

The standard research workflow has one scientific and tracked-code owner:

```text
GPT-5.6 Sol
    designs -> implements -> freezes -> audits -> interprets -> publishes

GPT-5.6 Luna (scoped subagent)
    verifies handoff -> executes frozen command -> monitors -> reports facts
```

Runtime model availability must be checked at dispatch time. When Luna is
requested for a runner or monitor task, explicitly request `gpt-5.6-luna`. Do
not spawn another Sol copy or silently substitute another model. If the runtime
cannot provide model-selectable Luna delegation, fail closed with
`LUNA_DISPATCH_UNAVAILABLE`; Sol may ask the user or deliberately execute the
task itself only in a later explicitly authorized step.

### Sol ownership

Sol owns every action requiring scientific, architectural, or publication
judgment, including:

- research questions, causal interpretation, method and ablation design;
- architecture, implementation, bug diagnosis, and tracked source edits;
- model, seed, split, budget, threshold, cache, retry, and access-policy choices;
- preregistration, protocol definition or amendment, manifest and hash freeze;
- scientific postmortems, result classifiers, paper conclusions, and report
  integration;
- git commits and, only when explicitly authorized, git pushes.

Sol is the only default coding agent. Sol must independently inspect Luna's
execution evidence and may not adopt a subagent's interpretation without the
required integrity and scientific audits.

### Luna execution scope

Luna may perform only bounded work that Sol has already specified and frozen:

- execute the exact frozen runner or verification command;
- run specified tests, preflights, `compileall`, or deterministic audits;
- monitor process health, stdout/stderr, checkpoints, ledgers, API/token budget,
  expected artifacts, frozen early-stop counters, and Validation/Test access;
- write ordinary runtime artifacts produced by the frozen command and an
  explicitly authorized factual execution summary.

Luna must not independently change code, prompts, scheduler or GEPA settings,
models, seeds, splits, thresholds, budgets, stopping, cache/retry semantics,
Validation/Test policy, preregistration, or protocol. Luna must not select a new
experiment, decide scientific causality or efficacy, edit tracked algorithm or
protocol files, commit, push, reset, clean, or stash. If any such action appears
necessary, Luna stops, preserves current evidence, and returns the anomaly to
Sol. Luna reports what happened; Sol decides what it means.

### Single writer and frozen handoff

There is one tracked-code writer at a time: Sol. Parallel Luna tasks are
normally read-only or execution-only and may not patch overlapping source.

No Luna experiment execution is allowed without a concrete handoff conforming
to `docs/workflows/EXPERIMENT_HANDOFF.md`. Only Sol may set
`READY_TO_RUN=true`, after protocol and manifest freeze, source/worktree and
hash checks, split governance, budget checks, and explicit API authorization.
Before execution Luna independently verifies the frozen commit, hashes, files,
command, model/seed/budget, worktree, access policy, and authorization. Any
mismatch fails closed; Luna does not improvise.

Luna executes only preregistered stop conditions and never stops or adapts based
on observed efficacy. Provider, credential, process, checkpoint, ledger, hash,
budget, Test-access, or protocol failures require fail-closed termination,
preserved evidence, and a factual return to Sol. Luna's completion record may
state `EXECUTION_COMPLETE` or `EXECUTION_ABORTED`; only Sol may classify the
scientific run as `VALID`, `INVALID`, `HOLD`, `NOT_EVALUABLE`, `SUPPORTED`, or
`NOT_SUPPORTED` after integrity, ledger, split-access, funnel, and scientific
audits.

## 5. Scientific-change rules

Change normative specification, runtime identity, implementation and meaningful
tests together when changing scientific behavior. Every behavior-affecting
configuration must have a read point and enter run identity. Keep diagnosis,
opportunity scheduling, search, candidate evaluation and transition separate.
Do not loosen parsers or acceptance guards to hide failures. Preserve the
benchmark's frozen team size, aggregation, prompt interface and split contract.

## 6. Experiment governance

Register experiments and parents before freeze. Use manifest schema v2 for new
Unified experiments; preserve historical schemas and permanent manifest paths.
Freeze source, scientific/governance/benchmark/data hashes, models, seed,
concurrency, budgets, access, stopping, cache and provider policy. Protocol
changes require an explicit amendment and fresh identity. Fail closed on
identity, ledger, authorization, split, persistence or conformance mismatch.
Audit validity separately from efficacy. Unfavorable valid results never justify
a rerun. Never reuse incompatible checkpoints or authorization.

## 7. Data / Validation / Test governance

Only frozen search and adaptive-gate memberships may affect optimization.
Post-freeze Validation and sealed Test have distinct roles and require their own
frozen access policy and authorization. Never feed held-out outcomes into search,
selection, memory or method design during execution. In paired final evaluation,
identical request identities share one provider realization/cache identity;
byte-identical final teams produce byte-identical paired evidence.

## 8. API authorization

Real APIs require explicit user authorization tied to the exact frozen attempt,
roles, phase, source and budgets. Architecture and code tasks use fake providers
and an offline network guard. Do not combine zero-API work with real API tests.
Readiness, templates, registry entries and cleanup never grant authorization.
Authorization scopes cannot be reused for retries, reruns or another cell.

## 9. Git / branch / commit rules

Preserve user changes and unrelated untracked files. The only long-lived branch
is main; freeze experiments by source SHA and hashes, not permanent branches.
Only Sol commits. Push only on explicit user request. For authorized direct
publishing, verify branch, remote, intended files and sanitization, stage only
task files and run `git push origin main`; no side branch or pull request.
Ignored runs, raw data, credentials, checkpoints and caches stay untracked.
Publish only sanitized evidence: hashes, counters, categories and metrics, never
prompts, questions, gold/model answers, raw responses, endpoints or host paths.

## 10. Historical replay policy

Frozen historical contracts live in docs/archive/specs/ and original source
commits. Keep original constants, manifests, reports and reproduction paths.
Do not silently promote historical runtime into the active method. Unknown,
unique evidence and replay-required files are never cleanup deletion targets.
Archive changes need provenance and an invariant index; deletion requires an
inventory, dry-run plan and tombstone. Historical reports remain immutable.

## 11. Required preflight checklist

Run compileall, the explicitly classified current pytest suite, governance and
manifest preflight, required deterministic smokes, sanitization and
`git diff --check`. Report omitted historical private-artifact tests explicitly;
never label a partial suite as all historical replay passing. Before real runs,
verify frozen source/startup identity, split isolation, authorization, lifecycle,
ledger, resource ceilings and exact command via the governed handoff.
