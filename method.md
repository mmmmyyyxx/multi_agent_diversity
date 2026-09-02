# Method

This file is the human-readable method exposition. The normative, mechanically
referenced implementation specification is `docs/design/CURRENT_SPEC.md`.

## 1. Scope

The current runtime implements:

```text
Repairability-Adjusted Dual-Target Prompt-Team Optimization
method_version = member_aware_peer_state_v15
checkpoint_version = 25
```

The runtime also supports frozen experimental Module2 context variants C2 and
C3 beside the byte-compatible C0/v15 path. These variants change generation
context only and are not yet promoted to the canonical method.

The team contains five prompts and uses equal-weight plurality with
tie-as-abstain. Model weights remain fixed. Every evaluated candidate changes
one target prompt while holding four peers fixed.

The final paper method has two core modules:

1. Repairability-Adjusted Member-Aware Dual-Target Search
2. Responsibility-Conditioned Evolution

Candidate write-back uses the Common-Safe Team Update. This is the foundational
safety policy, not a third research module.

## 2. Voting state and objective

For each example:

```text
G = valid gold votes
H = largest valid wrong cluster
M = G - H
vote correct iff M > 0
```

On the fixed optimization probe:

```text
g_i = c_i - c_i^0
O(Theta) = (V_count, min_i g_i, sum_i g_i)
```

The common candidate safety rule requires target non-regression, vote
non-regression, strict target-or-vote progress, and terminal-invalid
non-regression. Fixed-peer replacement makes strict improvement in `O(Theta)` a
derived invariant. Lane-only or responsibility-only progress is insufficient.

## 3. Residual eligibility and service

Only vote-wrong examples create residuals. For each currently wrong member:

```text
DeltaV_i,x = vote-correct change after counterfactual gold repair
DeltaM_i,x = plurality-margin change after counterfactual gold repair
```

The legal eligibility set retains all lexicographic maximizers:

```text
E_x = argmax_i (DeltaV_i,x, DeltaM_i,x)
```

Every serviceable residual is routed to one eligible member. The resulting
service portfolios are disjoint. Routing prefers anchor match, then no anchor,
then a different anchor, followed by lower lane load, lower total load, and
stable seeded rank.

Each residual has one lane: `coverage`, `direct_flip`, or `margin_support`.
Each member exposes one non-empty active lane and slice `A_i`. Only accepted
updates set or switch the committed member's specialization anchor. v15 has no
freeze, cooldown, blacklist, or persistent realizability state.

## 4. Repairability-adjusted target selection

For actionable member `i`:

```text
D_i = #{x in A_i : DeltaV_i,x = 1}

S_i_support =
  sum_{x in A_i, DeltaV_i,x = 0} max(0, DeltaM_i,x)

d_i = max(0, g_max - g_i - 5)
w_i = updates_since_selected_i
```

Normalize each dimension by its maximum over the current actionable set. When
a maximum is zero, every normalized value in that dimension is zero.

```text
B_i = 0.5 Dhat_i + 0.3 Shat_i + 0.2 dhat_i
rho_i = 1 / (1 + f_i)
score_i = (B_i + 0.05 what_i) rho_i
```

The v15 W1 change places wait inside the state-local repairability discount.
`f_i` is the number of normally completed, zero-feasible branch searches for
member `i` under the current prompt-team hash. No cross-state reputation or
realizability estimate is used.

Order actionable members by:

```text
(
  -score_i,
  -B_i,
  -Dhat_i,
  -Shat_i,
  -dhat_i,
  -what_i,
  seeded_rank_i,
  agent_id,
)
```

S1-S2 select Top-2 distinct members. Top-2 degrades to Top-1 when necessary.
An empty actionable set stops with `no_actionable_responsibility`. No target
Pareto frontier is used in the v15 path.

## 5. State-local repairability

A normal branch completion increments `branch_attempt_count`. It increments
`branch_feasible_count` when any candidate passes branch-local acceptance;
otherwise it increments `branch_failure_count`.

Operational failures do not update these counters. A feasible branch that loses
cross-branch competition is still feasible and receives no failure penalty.

Only a committed prompt transition with a changed team hash resets all
failure/attempt/feasible counters. Rejection, responsibility refresh, epoch
change, checkpointing, and audit refresh do not reset them.

## 6. Independent target branches

Candidate budget contracts are:

```text
Static: 0 branches x 0 candidates = 0
S0:     1 branch x 2 candidates = 2
S1-S2: 2 branches x 2 candidates = 4
```

Both S1-S2 branches start from the same parent team hash, profiles, team-state
version, peer cache, responsibility refresh, and routing snapshot. They build
independent target contexts and run complete Stage A/B without committing.
Every valid generated candidate enters its own branch's Stage B.

Each branch returns at most one winner. If two winners exist, every v15 main
setting uses the common cross-branch key:

```text
(vote gain,
 minimum-member-gain delta,
 total-member-gain delta,
 soft-utility delta,
 -vote losses,
 -edit tokens,
 -target selection rank,
 prompt hash)
```

The program atomically commits at most one winner, updates only its anchor,
computes the successor team hash, resets repairability counters, and refreshes
responsibility. The losing branch cannot change prompts, profiles, anchors, or
team state and is not counted as a failure when it had a feasible candidate.

## 7. Responsibility-conditioned evolution and update

Teacher receives bounded programmatic diagnosis and produces one repair plan.
Critic checks only hard semantic blockers. Student sees the parent prompt,
approved plan, immutable task output contract, and requested candidate count.
Rollouts determine value.

S2 compact context contains one selected lane, one dominant pattern, at most
two repair cases, and at most one preservation case. Scores, member IDs,
failure counts, waits, and routing loads are not exposed.

All optimized v15 main settings use:

```text
candidate_acceptance_policy = fixed_peer_monotone_target_or_vote
candidate_ranking_policy = common_monotone_safe
stage_a_policy = matched_all_generated
```

The v14 experimental RCRU implementation remains available only through
explicit legacy opt-in for historical replay and offline analysis. It is not a
v15 main module and its artifacts are not required for v15 runs.

## 8. Ablation matrix

| Setting | Optimization | Module 1 | Module 2 |
|---|---:|---:|---:|
| Static Reference | No | -- | -- |
| S0 Generic Prompt Evolution | Yes | No | No |
| S1 Member-Aware Dual-Target Search | Yes | Yes | No |
| S2 Responsibility-Conditioned Evolution (Full) | Yes | Yes | Yes |

Canonical names are:

```text
Static shared_static_reference
S0     shared_generic_evolution                   00
S1     shared_member_aware_dual_target            10
S2     shared_responsibility_conditioned_dual_target 11
```

S0 uses `1 x 2` search while S1/S2 use `2 x 2`; therefore S0-to-S1 estimates
the effect of Module 1 as implemented, not a compute-matched causal effect.
Explicit auxiliary compute controls remain outside the main matrix.

## 9. Historical RCRU decision

v14 included an experimental robust contribution update. Seed46 offline
diagnosis found lane-only progress misalignment, common-safe variants that made
responsibility ranking nearly inactive, and zero responsibility-attributable
scalar activation across 32 updates. v15 therefore removes RCRU from the final
main method without deleting its historical runtime or replay helpers.

## 10. Evaluation lifecycle and persistence

The active lifecycle performs no validation rollout or checkpoint selection.
The final active state is selected automatically. Test executes once after
training and never participates in optimization.

Checkpoint v25 persists state-local repairability counters, selected target
IDs, target-score history, branch decisions, routing, active lanes, and anchors.
Checkpoint v23 and earlier are rejected with `checkpoint_version_mismatch`;
there is no semantic migration.

Sanitized v15 artifacts contain only hashes, counters, lanes, normalized
values, and decision keys. They exclude prompt/question/answer text, raw model
output, credentials, endpoints, cache contents, checkpoints, and absolute
paths. RCRU-specific artifacts are required only for explicit legacy RCRU runs.
