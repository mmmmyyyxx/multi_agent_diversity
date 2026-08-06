# Method

## 1. Scope

The current runtime implements:

```text
Repairability-Adjusted Dual-Target Prompt-Team Optimization
method_version = member_aware_peer_state_v13
checkpoint_version = 22
```

The team contains five prompts and uses equal-weight plurality with
tie-as-abstain. Model weights remain fixed. Every evaluated candidate changes
one target prompt while holding four peers fixed.

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
derived invariant.

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
updates set or switch the committed member's specialization anchor.

v13 service routing never filters a member through freeze state.

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
score_i = B_i rho_i + 0.05 what_i
```

`f_i` is the number of normally completed, zero-feasible branch searches for
member `i` under the current prompt-team hash.

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

S1-S3 select Top-2 distinct members. Top-2 degrades to Top-1 when necessary.
An empty actionable set stops with
`no_actionable_responsibility`.

No target Pareto frontier is used in the v13 path.

## 5. State-local repairability

A normal branch completion increments `branch_attempt_count`. It increments
`branch_feasible_count` when any candidate passes branch-local acceptance;
otherwise it increments `branch_failure_count`.

Operational failures do not update these counters. A feasible branch that loses
cross-branch competition is still feasible and receives no failure penalty.

Only a committed prompt transition with a changed team hash resets all
failure/attempt/feasible counters. Rejection, responsibility refresh, epoch
change, checkpointing, and audit refresh do not reset them.

v13 has no freeze/unfreeze state machine.

## 6. Independent target branches

Candidate budget contracts are:

```text
Static: 0 branches x 0 candidates = 0
S0:     1 branch x 2 candidates = 2
S1-S3: 2 branches x 2 candidates = 4
```

Both S1-S3 branches start from the same parent team hash, profiles, team-state
version, peer cache, responsibility refresh, and routing snapshot. They build
independent target contexts and run complete Stage A/B without committing.
Every valid generated candidate enters its own branch's Stage B.

Each branch returns at most one winner. If two winners exist, common settings
use:

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

S3 uses the corresponding RCRU key with normalized lane utility, coalition
contribution, bootstrap LCB, positive/negative support, and edit size.

Absolute target correct count, raw lane utility, and raw portfolio size are
not cross-member comparison dimensions.

The program atomically commits at most one winner, updates only its anchor,
computes the successor team hash, resets repairability counters, and refreshes
responsibility. The losing branch cannot change prompts, profiles, anchors, or
team state.

## 7. Proposal and candidate decisions

Teacher receives bounded programmatic diagnosis and produces one repair plan.
Critic checks only hard semantic blockers. Student sees the parent prompt,
approved plan, immutable task output contract, and requested candidate count.
Rollouts determine value.

S2/S3 compact context contains one selected lane, one dominant pattern, at
most two repair cases, and at most one preservation case. Scores, member IDs,
failure counts, waits, and routing loads are not exposed.

S0-S2 use the common monotone target-or-vote branch policy. S3 retains RCRU
branch-local evaluation and changes no responsibility or dual-search rule.

## 8. Ablation matrix

```text
Static shared_static_reference                    outside module vector
S0     shared_generic_evolution                   000
S1     shared_member_aware_dual_target            100
S2     shared_responsibility_conditioned_dual_target 110
S3     shared_full_dual_target_rcru                111
```

Static performs no optimization, planned updates, target selection, or TCS
calls. The two auxiliary budget controls and the old seven-setting semantics
are explicit, require opt-in, and do not belong to the main matrix.

## 9. Evaluation lifecycle and persistence

The active lifecycle performs no validation rollout or checkpoint selection.
The final active state is selected automatically. Test executes once after
training and never participates in optimization.

Checkpoint v22 persists the state-local repairability counters, selected target
IDs, target-score history, branch decisions, routing, active lanes, and anchors.
Checkpoint v21/v12 is rejected with `checkpoint_version_mismatch`; there is no
migration.

Sanitized v13 artifacts contain only hashes, counters, lanes, normalized
values, and decision keys. They exclude prompt/question/answer text, raw model
output, credentials, endpoints, cache contents, checkpoints, and absolute
paths.
