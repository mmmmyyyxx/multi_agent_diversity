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
| multi_dataset_diverse_rl/current_contract.py | Closed current identity imports |
| multi_dataset_diverse_rl/versions.py | Preserved current and historical identities |
| multi_dataset_diverse_rl/benchmarks/ | Benchmark scientific and data contracts |
| experiments/registry.yaml | Experiment metadata |
| experiments/lineage.yaml | Experiment lineage |
| Frozen experiment manifest | One experiment's preregistration |
| reports/ | Immutable evidence |

Reports are evidence, never design authority. Historical claims of current,
active or canonical refer to the report's creation context.

## 3. Current project direction

The sole active research architecture is Unified Team Prompt Search.
Member state is one Role/Strategy/Answer object; User content is the raw problem.
All three blocks are editable. No immutable CoT or output-style instruction exists.
`UnifiedSearchOrchestrator` owns the production graph. The current composition
binds Gradient Pattern, Rolling Risk Memory and bounded Layer1 search through
one complete current policy bundle. Aggregation remains benchmark-selected.
Structured optimizer history is separate from LLM memory.
New experiments use `scripts/run_experiment.py`, the sole current composition
entrypoint and must compose through `build_current_team_prompt_search`.
Do not import legacy classes into current runtime. Retired method implementations are recovered only from their frozen Git source;
current production contains no compatibility execution modes.
Historical manifest parsing does not grant current execution eligibility.
Domain logic belongs in production modules, not scripts.
Local search acceptance cannot silently gate outer team candidate admission.
Pattern and Memory require explicit non-null policy identities and provider/limit
bindings; optimizer-only context must never enter Solver or aggregator inputs.
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
An explicit push request does not expand verification scope. Apply section 11
to the complete task diff and reuse applicable completed checks.
Ignored runs, raw data, credentials, checkpoints and caches stay untracked.
Publish only sanitized evidence: hashes, counters, categories and metrics, never
prompts, questions, gold/model answers, raw responses, endpoints or host paths.

## 10. Historical replay policy

Frozen historical contracts live in docs/archive/specs/ and original source
commits. Keep original constants, manifests and scientific reports. Retired runnable
code and redundant implementation snapshots may be removed after an inventory,
dry-run plan and tombstone pin their Git recovery source.
Do not silently promote historical runtime into the active method. Unknown,
unique evidence and replay-required files are never cleanup deletion targets.
Archive changes need provenance and an invariant index; deletion requires an
inventory, dry-run plan and tombstone. Historical reports remain immutable.

## 11. Verification scope and required preflight

Choose checks from the complete task diff and intended action before launching
them. State the change category and relevant checks briefly. Use the highest
applicable category for mixed changes; publishing reports after code changes
does not turn the task into a report-only change. File extension alone is not
a classification: normative scientific documentation and behavior-affecting
configuration belong to scientific changes.

| Change category | Required verification |
|---|---|
| Reports, result artifacts, ordinary documentation or workflow instructions only | Check intended files, links, relevant report arithmetic/consistency, sanitization, artifact hashes and `git diff --check`. Use focused documentation guards when relevant. No production `compileall`, full current pytest suite or whole historical governance/manifest sweep solely to publish or push. |
| Publication metadata only: report registration, parent links and derived indexes | Validate the changed records, references and lineage edges, plus affected generated views. Reuse unchanged manifest/source checks; do not revalidate every historical experiment. Changes to readiness, authorization, access or executable settings are not publication metadata. |
| Code, tools or tests without scientific behavior changes | Compile affected Python modules; run relevant tests, deterministic smokes and affected governance checks under the offline network guard. Broaden to the full current suite only for a concrete cross-module risk or insufficient coverage, and explain that reason before starting. |
| Scientific behavior, runnable protocol/configuration, or real experiment preparation/execution | Run production `compileall`, the explicitly classified current pytest suite, governance and manifest preflight, required deterministic smokes, sanitization and `git diff --check`. Verify the exact experiment freeze and handoff before real calls. |

Scientific/runtime changes include models, prompts, evidence interfaces,
sampling, evaluation, aggregation, admission, scheduling, memory, cache/retry,
concurrency, budgets, stopping and split/access/authorization policies. They
retain sections 5-8 and all frozen identity requirements. The full conformance
checklist in an experiment workflow applies to preparing or executing an
experiment; it is not a requirement for every documentation or report push.
Changes to this file that only clarify verification workflow use the focused
documentation category; changes to scientific or safety requirements do not.

Reuse a completed check when its relevant source, inputs and configuration
have not changed. Record the covered scope and reused evidence; do not claim
the check was newly run. Do not repeat or broaden checks merely because the
next step is commit/push. New changes, failures or unresolved risks may justify
additional checks; explain the affected scope rather than restarting all
preflights by default.

Record unrelated known baseline failures without claiming they pass or
silently fixing them during report publication. Failures affecting the changed
artifacts or required experiment/safety contracts must be resolved or reported
as blockers. Report omitted historical/private-artifact tests explicitly;
never label a focused or partial suite as all historical replay passing.

Before real runs, always verify frozen source/startup identity, split isolation,
exact single-use authorization, lifecycle, ledger, resource ceilings and exact
command via the governed handoff. Reusing offline check evidence never reuses
an authorization, incompatible checkpoint/cache, or consumed execution scope.
