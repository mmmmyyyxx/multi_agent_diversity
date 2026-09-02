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

Sections 1-9 below are retained as a **MIGRATION MIRROR** of the method-specific
contract while tooling moves to `CURRENT_SPEC.md`. Do not delete or independently
edit their semantics; any intentional algorithm change must update the normative
specification, runtime identity, implementation, and tests together.

## 1. Project mission

This repository studies joint optimization of a five-prompt LLM ensemble. The
five equal-weight member outputs are aggregated by plurality voting; a top-count
tie abstains and is incorrect. Model weights remain frozen.

The current method is:

```text
Repairability-Adjusted Dual-Target Prompt-Team Optimization

method_version = member_aware_peer_state_v15
checkpoint_version = 25
```

The paper method has two core modules:

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

## 2. Formal objective and fixed-peer safety

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

## 3. Joint voting diagnosis

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

## 4. Member-Aware Responsibility

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

## 5. Dual-target competitive search

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

## 6. Responsibility-Conditioned Evolution

The stable division of labor is:

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

## 7. Reduced two-module matrix

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

## 8. Solver, evaluation, and isolation

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

## 9. Persistence and artifacts

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
