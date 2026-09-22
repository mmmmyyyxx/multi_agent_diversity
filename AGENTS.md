# AGENTS.md

This file is the stable project memory and normative engineering contract.
Read it before `method.md`, `README.md`, implementation code, tests, or run
artifacts. Runtime identifiers must always be verified in
`multi_dataset_diverse_rl/versions.py`.

Do not record current commits, one-off results, temporary migration status, API
keys, endpoints, or local paths here.

## Source of Truth hierarchy

Use the narrowest authority for the fact being checked:

```text
AGENTS.md
    stable engineering workflow and safety contract

docs/design/CURRENT_SPEC.md
    normative current algorithm implementation specification

multi_dataset_diverse_rl/versions.py
    runtime identity and frozen numeric/protocol constants

experiment manifest
    experiment-specific preregistered and observed facts

reports/
    evidence only; never normative design
```

Conflict resolution is explicit: runtime constants come from `versions.py`,
algorithm semantics from `CURRENT_SPEC.md`, experiment-specific facts from the
manifest, and engineering workflow from this file. Historical reports are
immutable evidence and cannot silently promote an experimental arm into the
canonical runtime.

## Current active research architecture

The active research direction is a backend-neutral two-layer architecture:

```text
Layer 2 — Team-Level Responsibility/Search Controller
    responsibility and primary-residual assignment
    target scheduling and persistent realizability
    team-level candidate evaluation
    Common-Safe and Shadow checks
    competitive selection and atomic write-back
                         |
                         | LocalPromptOptimizer interface
                         v
Layer 1 — Local Prompt Optimizer
    current grounded backend: official frozen GEPA
    replaceable future backends: SEPO, ESPO, or another conforming optimizer
```

Layer 1 is an interchangeable prompt-optimization backend. It must not own
member selection, responsibility attribution, persistent realizability,
team-level acceptance, or write-back. Layer 2 is the team-level research layer
and must not depend on GEPA, Teacher, Critic, Student, or any other backend's
internal implementation. Replacing GEPA must not require redesigning Layer 2.

This architectural direction does not silently promote an experimental runtime
to canonical status. `versions.py` still identifies the canonical historical
runtime as `member_aware_peer_state_v15`, checkpoint v25. Opt-in two-layer and
Layer-2 scheduler identities must remain explicit in their manifests until a
separate, versioned promotion changes `CURRENT_SPEC.md`, `versions.py`, code,
and tests together.

### Two-layer ownership boundary

Layer 2 owns member selection, responsibility attribution, exact responsibility,
latest-transition focus/anchor, and local-evaluation evidence selection, the
ordered evidence schedule, persistent realizability, team-level empirical evaluation,
Common-Safe and Shadow checks, cross-member competition, and atomic write-back.
Layer 1 retains optimizer-specific prompt-search mechanics, but treatment
backends may consume only the immutable Layer-2 evidence packet. They may not
fetch, select, replace, expand, reorder for selection, or fall back to examples
outside that packet.

Layer 1 owns how the assigned local problem is searched: internal sampling,
proposal generation, candidate population and selection, and optimizer-specific
state. It must not choose the target member, redefine responsibility, use
persistent team realizability or plurality outcomes for target allocation, or
commit prompts to the team.

```text
Layer 2 defines WHO to optimize, WHAT responsibility to pursue, and WHICH
evidence Layer 1 must optimize from.
Layer 1 retains HOW to mutate, reason over, compare and accept prompt candidates.
```

The experimental treatment decomposes Layer-2 evidence as
`E_L2 = E_team union E_transition`: current team residuals provide
responsibility evidence, while only the latest accepted parent-to-child edit
provides focus (newly broken) and anchor (newly fixed) evidence. Root prompts
have empty focus and anchor sets. Local evaluation is frozen separately. This
borrows transition-evidence semantics associated with SEPO; it does not import
SEPO structural edits, architects, breadcrumb search, Lexicase selection,
archive admission, or lineage-parent selection into Layer 2.

### Unified backend runtime

Optimizer and controller choices are runtime configuration, never scientific
facts inferred from a branch name. The current executable matrix is:

```text
optimizer_backend=gepa  optimization_mode=native
optimizer_backend=gepa  optimization_mode=layer2
optimizer_backend=mars  optimization_mode=native
optimizer_backend=mars  optimization_mode=layer2
```

All four modes come from one commit and one shared Layer-2 implementation.
Native mode preserves the backend's own Optimize-only data flow. Layer2 mode
requires the backend to consume the immutable `ResponsibilityEvidencePacket`
without fetching or selecting other examples. Formal manifests record both
fields and the backend fidelity identity. Backend-specific telemetry is nested
under `backend_details` in the common run record.

The only long-lived development branch is `main`. Backend work may use a
temporary feature branch, but formal experiments are frozen by commit SHA,
configuration/data/protocol hashes, and immutable preregistration/result tags.
Do not create permanent experiment branches for an optimizer or controller.

Replacing GEPA with SEPO/ESPO must not require an algorithmic change to Layer 2.

### Production architecture rules

New experiments use `multi_dataset_diverse_rl.experiment.run_experiment` and
the single `scripts/run_experiment.py` composition entrypoint. Do not add a new
experiment-named production runner. Historical runners may remain for exact
reproduction or translate legacy arguments into the production contract, but
production modules must never import them.

Do not put scientific logic in scripts. Layer 1 owns local search only. Layer 2
owns target/responsibility/evidence, team admission and write-back, and must not
import a concrete optimizer backend. GEPA and MARS implement the same typed
`LocalOptimizerBackend` contract. Production Layer 1 receives an explicit
`LocalOptimizationRequest` and `RuntimeContext`, never the historical
monolithic `Config` attribute bag.

Backend, native/Layer2 scope and fixed-budget/saturation regime are orthogonal
`ExperimentSpec` fields. Shared stopping belongs in `saturation.py`. Provider
construction uses `ProviderClientFactory`; identity, authorization and
lifecycle use the centralized governance layer. A new experiment should
normally require only a manifest/configuration, not a runner, controller,
backend adapter or stopping implementation.

### Saturation-mode stopping

Saturation mode disables ordinary scientific hard budgets. Native backends use
backend-native complete optimization units. Layer-2 modes distinguish local
accepted updates from successful team commits, and team-level no-update
patience is reset only by a successful atomic team commit. A Vote-neutral
Common-Safe commit is still a team update. Emergency ceilings are mandatory
operational safeguards and never imply scientific convergence.

The native optimizer data flow remains a separate control. The Layer-2
treatment intentionally replaces native example selection while preserving the
optimizer's prompt-search core. GEPA and MARS native budget units need not match
and must not be used for direct cross-backend superiority claims.

### Local optimizer fidelity policy

The current official GEPA backend is **Level B: API-compatible adaptation**.
Allowed adaptations use supported optimizer seams: adapters/interfaces, system
text-component representation, local datasets/problem domain, metrics and
evaluator, reflection templates, provider/model configuration, resource budget,
callbacks, and logging. Editing or forking the optimizer search core,
reimplementing its population/Pareto/parent selection, replacing its local
acceptance, or making Layer 2 control its internal search is forbidden. A
derived optimizer doing any of those requires a separate identity and must not
claim Level-B fidelity.

Official optimizer code fidelity and optimization-problem equivalence are
different concepts. Ours + GEPA uses an API-defined local problem while keeping
the official GEPA search engine unmodified; never describe it as an exact native
GEPA reproduction. This distinction applies equally to future SEPO/ESPO
backends.

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

Sections 1-9 below are retained as a **HISTORICAL CANONICAL/REPLAY MIGRATION
MIRROR** of the v15 method-specific contract while tooling moves to
`CURRENT_SPEC.md`. They are not the sole active research architecture. Do not
delete or independently edit their semantics; an intentional canonical
algorithm change must update the normative specification, runtime identity,
implementation, and tests together.

## 1. Historical v15 canonical runtime mission

This repository studies joint optimization of a five-prompt LLM ensemble. The
five equal-weight member outputs are aggregated by plurality voting; a top-count
tie abstains and is incorrect. Model weights remain frozen.

The canonical v15 replay/runtime identity is:

```text
Repairability-Adjusted Dual-Target Prompt-Team Optimization

method_version = member_aware_peer_state_v15
checkpoint_version = 25
```

The historical v15 paper method has two core modules:

```text
1. Repairability-Adjusted Member-Aware Dual-Target Search
2. Responsibility-Conditioned Evolution
```

Candidate write-back uses the Common-Safe Team Update. It is the foundational
safety rule, not a third independent research contribution.

Module one is added atomically: counterfactual responsibility, unique routing,
one active lane, scalar repairability-adjusted selection, state-local failure
discount, Top-2 independent branches, and competitive max-one commit are not
separate default ablations. Joint voting state, Teacher-Critic-Student, Stage
A/B, caching,
retry, checkpointing, and audits are implementation or reliability mechanisms
rather than additional research contributions.

## 2. Historical v15 formal objective and fixed-peer safety

For member `i` on the fixed optimization probe:

```text
c_i^0 = initial correct count
c_i   = current or candidate correct count
g_i   = c_i - c_i^0
g_min = min_i g_i
g_sum = sum_i g_i
V_count = plurality-vote correct count
O(Theta) = (V_count, g_min, g_sum)
```

Formal search uses integer counts. Normalized accuracy is reporting-only. The
method does not optimize prompt distance, trace distance, generic disagreement,
or a standalone diversity reward.

Every common-policy candidate replaces exactly one target while holding four
peers fixed. It must satisfy:

```text
candidate target correct >= incumbent target correct
candidate vote correct >= incumbent vote correct
target correct or vote correct strictly improves
terminal-invalid count does not increase
```

This permits vote-only progress when target gain is zero. Under fixed peers the
first three guards imply strict Pareto improvement in `O(Theta)`; this is a
derived invariant, not another rejection reason.

All optimized v15 main settings use this common policy. Responsibility or lane
utility cannot redefine progress. The v14 RCRU policy remains only for explicit
legacy replay and offline analysis.

## 3. Historical v15 joint voting diagnosis

For every probe example:

```text
G = valid gold-vote count
H = largest valid wrong-answer cluster
M = G - H
vote correct iff M > 0
```

Full-team `TeamVoteState` and leave-one-out `PeerVoteContext` are distinct
typed concepts. Coverage, conversion, dominant-wrong, unique/pivotal, and soft
utility remain diagnostic or proposal-context evidence.

## 4. Historical v15 Member-Aware Responsibility

### 4.1 Residual eligibility

Only vote-wrong examples create team residuals. Only members currently wrong on
a residual are considered. Holding peers fixed, compute:

```text
DeltaV_i,x = counterfactual vote-correct gain
DeltaM_i,x = counterfactual plurality-margin gain
E_x = lexicographic argmax_i (DeltaV_i,x, DeltaM_i,x)
```

Direct vote flips rank first, then larger margin gain, while exact ties are
retained. Member gain, wait, search history, failure counts, Proposal Memory,
coverage labels, and routing load may not affect `E_x`.

### 4.2 Unique service and active lanes

Legal portfolios may overlap:

```text
R_i = {x : i in E_x}
```

Every serviceable residual is routed to exactly one `q_x in E_x`, producing
pairwise-disjoint service portfolios `P_i`. Routing uses anchor compatibility,
lane load, total load, and stable seeded rank. v15 has no active freeze filter:
all legally eligible members may receive service.

Every residual has one lane:

```text
coverage
direct_flip
margin_support
```

Each member has one active lane and active slice:

```text
A_i = {x in P_i : lane(x) = active_lane_i}
```

Only an accepted update sets or switches the winner's specialization anchor.
Rejected, operationally failed, and competition-losing branches do not change
prompts or anchors.

### 4.3 Repairability-adjusted target score

Only members with non-empty legal portfolio, service portfolio, and active
slice are actionable. For each actionable member:

```text
D_i = count of x in A_i with DeltaV_i,x = 1

S_i_support =
    sum max(0, DeltaM_i,x)
    over x in A_i where DeltaV_i,x = 0

d_i = max(0, g_max - g_i - 5)
w_i = updates_since_selected_i
```

Direct-flip margin must not also enter `S_i_support`.

Normalize `D`, `S_support`, `d`, and `w` by their per-state maxima over the
actionable set. A zero maximum yields a zero normalized dimension.

```text
B_i = 0.5 Dhat_i + 0.3 Shat_i + 0.2 dhat_i
rho_i = 1 / (1 + branch_failure_count_i)
A_i_score = (B_i + 0.05 what_i) * rho_i
```

The weights are frozen in `versions.py` and must not be tuned on formal test
results. Members are totally ordered by:

```text
(
  -expected_update_value,
  -opportunity_value,
  -normalized_direct_fix,
  -normalized_support_margin,
  -normalized_uplift_deficit,
  -normalized_wait,
  seeded_rank,
  agent_id,
)
```

The v15 canonical path must not construct or query a target Pareto frontier.
Old Pareto code may remain only for explicit legacy-v12 readers and replay
tests.

All reduced-matrix member-aware settings S1-S2 select Top-2 distinct actionable
members. If only one is actionable, dual search degrades to one branch. If none
is actionable:

```text
early_stop_reason = no_actionable_responsibility
```

### 4.4 State-local repairability

The runtime tracks:

```text
branch_failure_count_by_agent
branch_attempt_count_by_agent
branch_feasible_count_by_agent
repairability_state_team_hash
repairability_reset_count
```

A normal completed branch increments attempts. A branch with at least one
passed candidate increments feasible count and never failure count, even when
it loses cross-branch competition. A normal completed branch with no passed
candidate increments failure count.

Transport failure, provider truncation, Teacher/Critic schema failure, terminal
Student schema failure, Solver infrastructure failure, and persistence failure
do not increment the branch counters.

Only an accepted prompt-team transition with a changed team hash resets all
three per-agent counters. Rejection, routing refresh, epoch change, audit
refresh, and checkpoint save do not reset them.

v15 has no active freeze/unfreeze mechanism, freeze threshold, frozen portfolio
signature, service block, or `all_actionable_members_frozen` stop.

## 5. Historical v15 dual-target competitive search

S1-S2 use:

```text
target_branch_count = 2
candidates_per_target_branch = 2
total_generated_candidates_per_update = 4
```

S0 uses one branch with two candidates. Static Reference uses zero branches,
zero candidates, zero planned updates, and zero TCS calls. Every generated
valid candidate enters its branch's Stage B.

Both selected target branches start from the same parent team hash, team-state
version, peer-state cache, responsibility refresh, routing snapshot, and active
profiles. They independently build target-specific context, generate
candidates, and run complete Stage A/B. Neither branch may observe a prompt,
anchor, or team-state mutation produced by the other.

Each branch yields zero or one branch winner. After both branches finish:

- no branch winner means no team update;
- one branch winner is committed;
- two branch winners are compared by the versioned cross-branch key;
- at most one member prompt is committed per update.

Common-policy cross-branch key:

```text
(
  team_vote_gain,
  minimum_member_gain_delta,
  total_member_gain_delta,
  soft_vote_utility_delta,
  -vote_loss_count,
  -total_edit_token_count,
  -target_selection_rank,
  prompt_hash,
)
```

Cross-member comparison must not use target absolute correct count, raw active
lane utility, or raw portfolio size.

Commit order is:

```text
choose global branch winner
atomically commit one prompt/profile
update only the winner's anchor
compute successor team hash
reset state-local repairability
refresh eligibility, routing, and active lanes
write checkpoint and audit
```

Any failure during atomic refresh restores prompt, profile, anchor, counters,
caches, versions, and audit lengths.

## 6. Historical v15 Responsibility-Conditioned Evolution

The historical v15 division of labor is:

```text
Program: numerical and typed diagnosis
Teacher: one bounded repair hypothesis
Critic: hard semantic blockers only
Student: replacement prompt realization
Rollout: empirical candidate value
```

The compact conditioned context contains:

```text
parent prompt
one program-selected lane and repair goal
one dominant lane/error-role pattern
at most two same-lane repair cases
at most one independent preservation case
compact previous empirical status
```

It exposes no member identity, score, failure count, gain, wait, routing load,
seeded rank, raw model output, or all rejection reasons. Student sees only the
parent prompt, approved plan, immutable task output contract, and requested
candidate count.

Student invalid recovery remains bounded to two cycles and at most eight calls.
Operational and schema failures are audit-only and expose no empirical rollout
feedback to the next Teacher.

Proposal Memory remains optional and defaults to `off`. It may not alter
eligibility, routing, active lanes, target scores, branch selection, or
acceptance.

## 7. Historical v15 reduced two-module matrix

Canonical main settings are:

```text
Static shared_static_reference                    not in module vector
S0     shared_generic_evolution                   00
S1     shared_member_aware_dual_target            10
S2     shared_responsibility_conditioned_dual_target 11
```

Adjacent settings add exactly:

```text
Static -> S0 generic prompt evolution
S0 -> S1 repairability-adjusted member-aware dual-target search
S1 -> S2 responsibility-conditioned evolution
```

S2 is the full method. Common-Safe Team Update is shared by S0-S2 and is not a
module-vector dimension.

The old v14 RCRU setting, seven-setting v13 semantics, and legacy v12 settings
are rejected by default and are available only through explicit legacy
identities with `--allow_legacy_setting 1`.

Auxiliary compute controls are not main settings:

```text
aux_dual_target_budget_matched_2x1
aux_single_target_compute_matched_1x4
```

They require `--allow_auxiliary_setting 1`.

## 8. Historical v15 solver, evaluation, and isolation

Every Solver response must contain exactly one strict:

```text
FINAL_ANSWER: <answer>
```

The program appends the immutable task output interface after the mutable
decision procedure. Mutable prompts containing recognizable output-interface
markers or fixed answer payloads are rejected rather than rewritten. Keep
`solver_max_tokens=1800`; strict parser failures remain visible.

No validation rollout or checkpoint selection is used. The final active team is
selected automatically. Test data is evaluated at most once, only after
training completes, and never affects diagnosis, responsibility, branch
selection, candidate acceptance, or final-state selection.

No-API code tasks must not be combined with API tests unless the user explicitly
authorizes API calls in that task.

## 9. Historical v15 persistence and artifacts

Checkpoint v25 stores branch failure/attempt/feasible counts, repairability team
hash and reset count, selected targets, target scores, branch lifecycle,
routing, active lanes, and anchors. A v23 or earlier checkpoint fails with:

```text
checkpoint_version_mismatch
```

There is no automatic migration and legacy freeze state must not enter a v15
runtime.

Required v15 sanitized artifacts include:

```text
repairability_adjusted_target_scores.jsonl
dual_target_branch_decisions.jsonl
dual_target_commit_decisions.jsonl
repairability_failure_events.jsonl
repairability_reset_events.jsonl
```

These artifacts may contain hashes, counters, lanes, normalized values, and
decision keys. They must not contain prompt text, question text, gold answers,
model answers, raw LLM output, credentials, endpoints, SQLite contents,
checkpoints, or absolute local paths.

## 10. Engineering guardrails

- Preserve exactly five agents, equal-weight plurality, and tie-as-abstain.
- Keep eligibility, scheduling, branch evaluation, and candidate acceptance
  separate.
- Keep domain calculations in their domain modules; do not move everything
  into `system.py`.
- Every behavior-affecting config field must have a real read point and enter
  run identity.
- Do not loosen parsers or acceptance guards to hide failures.
- Preserve user changes in a dirty worktree.
- Do not modify historical run directories or reports during code-only tasks.
- Do not call real APIs unless the user explicitly authorizes them.
- Do not reuse incompatible checkpoints.
- In a paired final evaluation, identical solver request identities must share
  one provider realization/cache identity across arms. Byte-identical final
  teams must produce byte-identical paired evaluation evidence.
- Run `compileall`, `pytest`, preflight, required deterministic smokes, a
  sanitization scan, and `git diff --check` before handoff.

## 11. Direct main publishing

Do not infer permission to publish. When the user explicitly requests a direct
push to `main` or says `git push main`, verify the intended files, current
branch, remote, tracked worktree, and sanitization status; stage only the
task-related files, create one intentional commit when needed, and run:

```text
git push origin main
```

For this explicitly requested workflow, do not create a side branch or pull
request. Runtime artifacts under ignored `runs*/` must remain untracked; copy
only sanitized, analysis-ready evidence into a purpose-specific `reports/`
directory. Never publish SQLite caches, checkpoints, prompts, questions, gold
answers, model answers, raw responses, credentials, endpoints, or absolute
local paths. Report the pushed commit and the exact published file scope.
