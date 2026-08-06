# Method

## 1. Scope

The current method is **Member-Aware Prompt-Team Optimization**:

```text
method_version = member_aware_peer_state_v11
```

It jointly searches a team of five prompts. Solver, optimizer, and evaluator
model weights remain frozen. Five equal-weight outputs are aggregated by
plurality vote, and a top-count tie abstains.

The paper method has exactly three modules:

```text
1. Member-Aware Responsibility
2. Responsibility-Conditioned Evolution
3. Monotone Target-or-Vote Team Update
```

The complete data flow is:

```text
Current Prompt Team
        ↓
Joint Team Diagnosis
(G, H, M and member gains)
        ↓
Member-Aware Responsibility
(counterfactual eligibility + compact target scheduling)
        ↓
Responsibility-Conditioned Evolution
(member-specific residual context and prompt candidates)
        ↓
Monotone Target-or-Vote Update
        ↓
Updated Prompt Team
```

Three decisions remain separate:

```text
who is eligible to repair a residual
!=
which eligible member is updated now
!=
whether an empirically evaluated candidate is committed
```

Teacher-Critic-Student, Stage A/B, fixed-peer rollout, repairability freeze, retry, cache,
checkpointing, and audit are bounded implementation or reliability mechanisms.
They are not additional paper modules.

## 2. Solver And Vote Contract

Each member must return exactly one valid:

```text
FINAL_ANSWER: <answer>
```

The optimized prompt is only the mutable decision procedure. Every Solver
request appends an immutable task-specific output interface after it:

```text
Follow the decision procedure below.

Decision procedure:
<mutable candidate prompt>

Mandatory output interface:
This interface is immutable and overrides any conflicting instruction above.
<strict task-specific FINAL_ANSWER contract>
```

The mutable procedure is validated before initialization, checkpoint restore,
candidate rollout, accepted-state commit, and Solver request construction. It
must not contain a recognizable `FINAL_ANSWER` marker, copy or describe the
immutable interface, or add final-response formatting instructions. Student
responses are filtered candidate by candidate; contamination is never stripped
or rewritten. A protocol-only exhausted search is recorded as
`proposal_protocol_failure` and does not advance repairability-freeze state.

Strict-parser invalid output receives request-local identical retries. The first
valid result is used; an exhausted sequence becomes one terminal-invalid
observation. Formal guards use terminal-invalid count. Transport retry remains
separate.

For each example:

```text
G = number of valid gold votes
H = size of the largest valid wrong-answer cluster
M = G - H
```

Under tie-as-abstain:

```text
vote correct iff M > 0
```

Full-team `TeamVoteState` and each member's leave-one-out `PeerVoteContext`
diagnose coverage failure, conversion failure, dominant wrong clusters, and
unique or pivotal correct behavior. This diagnosis supports Module 1 and Module
2; it is not a separate contribution.

## 3. Formal Member And Team Objectives

All formal objectives use integer correct counts on one fixed optimization
probe.

Let `c_i^0` be member `i`'s initial correct count and `c_i` its current or
candidate count:

```text
g_i   = c_i - c_i^0
g_min = min_i g_i
g_sum = sum_i g_i
```

The team objective is:

```text
O(Theta) = (V_count, g_min, g_sum)
```

`V_count` is the number of examples correctly answered by plurality vote.
Vector `a` Pareto-dominates `b` when every component is no lower and at least
one is strictly higher.

Under fixed-peer single-target replacement, `g_sum` changes by exactly the
target member's correct-count change. Target non-regression therefore implies
that neither `g_sum` nor `g_min` can regress. Combined with vote
non-regression and strict target-or-vote progress, this implies strict Pareto
improvement in `O(Theta)`. The objective is retained for evaluation,
trajectories, and invariant auditing; it is not an additional candidate
rejection guard.

`g_min` is candidate-discriminative only when the selected target is the
unique weakest member. Thus the member-first selector is a conditional
member-first preference, not a general three-objective Pareto optimizer. The
all-member fairness mechanism with broad operational effect is the uplift
deficit `d_i` in target scheduling, not an independent `g_min` acceptance
guard.

Normalized accuracy and soft vote utility are diagnostics. The method does not
replace the integer objective with a weighted scalar and does not directly
optimize prompt distance, trace distance, generic disagreement, or another
standalone diversity reward.

## 4. Module 1: Member-Aware Responsibility

### 4.1 Counterfactual eligibility

Only vote-wrong examples create team residuals. For a vote-wrong example `x`,
only members currently wrong on `x` are considered.

Holding the other four members fixed, replace member `i`'s answer by gold and
compute:

```text
DeltaV_i,x = counterfactual vote-correct gain
DeltaM_i,x = counterfactual plurality-margin gain
```

The eligibility key is:

```text
eligibility_key(i, x) = (DeltaV_i,x, DeltaM_i,x)
```

It is maximized lexicographically:

```text
1. direct vote flip first
2. larger margin gain second
3. retain all exact ties
```

Thus:

```text
E_x = lexicographic argmax over wrong members
      of (DeltaV_i,x, DeltaM_i,x)
```

The same residual may legitimately appear in several member portfolios.
Eligibility is state-local repair legitimacy, not permanent ownership.

Member gain, uplift deficit, member wait, accepted-update history,
candidate-search history, Proposal Memory, coverage label, conversion label,
dominant-wrong label, soft utility, and portfolio load cannot alter `E_x`.

Coverage, conversion, dominant-wrong, unique/pivotal, and soft-utility fields
remain diagnostic, proposal-context, and artifact evidence only.

### 4.2 Repair lanes and unique service routing

The legal portfolio remains:

```text
R_i = {x : i in E_x}
```

Legal portfolios may overlap and are used for eligibility audit, complete
coverage statistics, and repairability-freeze signatures. The program assigns
each residual exactly one mutually exclusive lane:

```text
coverage       if G_x = 0
direct_flip    if G_x > 0 and DeltaV_x = 1
margin_support otherwise
```

Dominant-wrong is a diagnostic label and preservation is separate protection
evidence; neither is another repair lane.

For every serviceable residual, deterministic routing chooses exactly one
`q_x in E_x`. If any eligible member is unfrozen, frozen members are excluded;
if all are frozen, the residual is retained in legal audit but marked blocked.
Routing orders candidates by:

```text
1. matching specialization anchor
2. no anchor
3. a different anchor
4. lower current load in the residual lane
5. lower total service load
6. stable seeded rank
```

The resulting service portfolios are disjoint:

```text
P_i = {x : q_x = i}
P_i intersect P_j = empty, i != j
```

### 4.3 Specialization anchor and active slice

Each S4/S5 member has an optional specialization anchor. Only an accepted
update sets or switches it to the active lane used by that update. Rejection,
schema failure, transport failure, and candidate regression leave it unchanged.
If the anchored lane remains in `P_i`, it is retained. Otherwise the program
selects one non-empty lane lexicographically by `(D_i^lane, S_i^lane,
N_i^lane, laneRank)`, with `direct_flip > coverage > margin_support` only as
the final exact-tie breaker.

Define the single active slice:

```text
A_i = {x in P_i : lane(x) = active_lane_i}
```

The target scheduler and the S5 proposal context use `A_i`, never the mixed
service portfolio. Unfreezing clears the old anchor so the member can choose a
new direction.

### 4.4 Uplift deficit

Let:

```text
g_max = max_j g_j
d_i   = max(0, g_max - g_i - 5)
```

`d_i` affects only member-level scheduling. It never changes per-residual
eligibility.

### 4.5 One member-level Pareto

For each unfrozen member with a non-empty service portfolio and active slice:

```text
D_i = number of direct flips in A_i
S_i = sum of DeltaM over A_i
T_i = (D_i, S_i, d_i)
```

The scheduler computes exactly one member-level Pareto frontier. Within its
first frontier, selection uses:

```text
1. larger updates_since_selected
2. stable seeded rank
```

Equivalently:

```text
A_t = {i : P_i and A_i are non-empty and frozen_i = false}
T_i = (D_i, S_i, d_i), i in A_t
F_t = ParetoFront({T_i : i in A_t})
i_t = lexicographic argmax over i in F_t of
      (updates_since_selected_i, -seeded_rank_i)
```

There is no second responsibility frontier, joint frontier, hidden scalar, or
substantive reordering by `D_i`, `S_i`, or `d_i` after the frontier is formed.

`d_i` is the sole weak-member protection term. A weakest member is protected
only when its portfolio is non-empty, it is not frozen, and its vector remains
on the first frontier. Equal deficit does not prevent strict domination in
`D_i` and `S_i`. There is no waiting-time override, deficit-service lane, or
generic compensation mechanism.

### 4.6 State-Conditioned Repairability Freeze

The active responsibility set is:

```text
A_t = {i : P_i and A_i are non-empty and frozen_i = false}
```

Two consecutive complete failures under the same responsibility portfolio
state freeze a member. The sanitized portfolio signature hashes sorted
`(question_hash, DeltaV, DeltaM)` tuples plus `D_i`, `S_i`, and residual count.
A complete failure requires a formally selected target, exhausted normal
Teacher/Critic/Student and candidate evaluation budget, no infrastructure,
protocol, or parser failure, and no accepted candidate. Each update contributes
at most one failure. An accepted update resets the target's failure streak.

Frozen members remain in residual eligibility, legal portfolios, anchors, and
audit records, but cannot receive new service routing or enter target selection. They
return only when both conditions hold:

```text
at least two accepted updates by other members
and
Jaccard(frozen residual hashes, current residual hashes) < 0.8
    or current D_i != frozen D_i
```

Minor `S_i` changes alone do not unfreeze a member. Rejected updates do not
refresh routing. Accepted transitions atomically refresh eligibility, freeze
state, routing, service portfolios, and active lanes. If no service portfolio
is actionable, selection reports
`no_actionable_repairability` and optimization stops with
`early_stop_reason = all_actionable_members_frozen`.

Freeze is a search-budget safeguard inside target scheduling, not a fourth
research module. `updates_since_selected` remains only a first-frontier
tie-break and never revives a frozen or dominated member.

## 5. Module 2: Responsibility-Conditioned Evolution

S5 receives only the target's program-selected active lane. Its model-facing
context contains:

```text
parent prompt
one repair lane and its fixed repair goal
one dominant (lane, target-error-role) pattern
at most two repair examples from that lane
at most one independent preservation example
one compact previous-outcome status and main rejection
```

The context never exposes member identity, gain, uplift deficit, responsibility
scores, vote/peer-state numbers, routing loads, freeze/anchor state, or the
complete rejection list. The program computes eligibility, chooses and routes
the residual, selects the lane and pattern, and compares scheduler vectors;
the LLM performs none of those decisions.

The role division is:

```text
Program:
    compute all numerical and typed diagnostic evidence

Teacher:
    propose one bounded repair hypothesis

Critic:
    check hard semantic blockers

Student:
    realize the approved plan as replacement prompts

Rollouts:
    determine empirical value
```

Programmatic aggregation uses the complete fixed probe. S5 supplies exactly
one pattern, no more than two repair cases, and no more than one preservation
case. Its serialized context is capped at
`min(config.tcs_context_max_chars, 6000)`: preservation is removed first, then
the second repair case; parent prompt, question, gold answer, and repair goal
are never truncated. S4 uses the same routing and scheduler but retains the
generic peer-state context. S1-S3 create no service routing, anchors, active
lanes, or freeze state.

Teacher returns exactly:

```json
{"failure_pattern":"...", "repair_rule":"...", "preservation_rule":"..."}
```

Critic checks only:

```text
evidence_mismatch
actionable_specificity
shortcut_or_copying
preservation_or_output_risk
```

and returns:

```json
{"failed_checks":[], "risk_case_ids":[], "feedback":""}
```

Student sees only the parent prompt, approved repair plan, immutable output
contract, and requested candidate count. The contract is context only: Student
must return reasoning/decision procedures without quoting, imitating, or
describing the interface, adding a fixed answer, or adding final-response
formatting. It returns:

```json
{"candidate_prompts":["complete replacement prompt"]}
```

The existing revision and recovery protocol remains unchanged: a zero-valid
Student response receives structured feedback and up to three retries; after
four invalid calls, at most one fresh Teacher-Critic regeneration starts one
final four-call Student cycle. A partially valid response stops recovery and
only valid candidates enter Stage A.

`PreviousUpdateOutcome` distinguishes operational pipeline execution from
empirical rollout feedback. Transport, truncation, and schema failures never
masquerade as candidate evidence.

## 6. Module 3: Monotone Target-or-Vote Team Update

Candidate evaluation replaces exactly one target prompt and holds the other
four prompts and profiles fixed. It computes target, team vote, all-member,
terminal-invalid, residual, coverage, conversion, and protection diagnostics.

### 6.1 Stage A

Member-aware settings shortlist through three channels:

```text
team_vote
worst_member
mean_member
```

Channel ranks are merged deterministically into the fixed Stage B budget.
Stage A/B is evaluation-efficiency implementation, not an additional method
module.

### 6.2 Stage B acceptance

A candidate must satisfy:

```text
candidate target correct count >= incumbent
candidate vote correct count >= incumbent
target or vote must strictly improve
terminal-invalid count must not increase
```

The acceptance identifier is:

```text
CANDIDATE_ACCEPTANCE_VERSION =
fixed_peer_monotone_target_or_vote_v2
```

The first three conditions imply that `(V_count, g_min, g_sum)` strictly
Pareto-dominates the incumbent. Runtime verifies this derived property with a
fail-fast invariant. It does not enter `passed`, `hard_feasible`, rejection
reasons, or candidate ranking. The legacy objective-dominance artifact fields
remain diagnostic aliases for this derived invariant.

In particular, vote-only progress remains valid:

```text
target gain = 0
vote gain > 0
```

provided every other guard passes. Strict target improvement is not required.
Target-only progress (`target gain > 0`, `vote gain = 0`) is equally valid.

Vote loss, soft utility, coverage, conversion, unique-correct loss, and
pivotal-correct loss remain diagnostics or late deterministic tie-break
evidence. None is an independent rejection guard.

### 6.3 Safe selection preferences

S1, S2, and S3 use the same four Stage B guards. S1 ranks feasible candidates
by individual target accuracy. S2 uses Vote-First Safe Selection. S3-S5 use
Member-First Safe Selection with this exact lexicographic order:

```text
minimum member gain
vote-correct count
target gain versus incumbent
net vote delta
assigned-residual repair count
mean soft vote utility
coverage gain count
fewer vote losses
fewer pivotal-correct losses
fewer unique-correct losses
earlier generation
stable prompt hash
```

`total_gain_count` is deliberately absent: under fixed-peer single-target
replacement its change equals target gain. Feasible candidates are ranked
directly; no candidate-level team-objective Pareto front filters them first.

## 7. State Lifecycle

The initial team is diagnosed once. Rejected candidates do not change active
prompts, profiles, team state, or responsibility state.

An accepted target prompt/profile replacement is atomic. It increments the
team-state version and triggers exactly one diagnosis and responsibility
refresh. The refreshed state is reused until the next accepted team
transition.

Before the accepted-state refresh, the target anchor is set to the active lane
used by the accepted update. The refresh then recomputes legal eligibility,
applies freeze/unfreeze transitions, clears anchors for newly unfrozen members,
and deterministically rebuilds service routing, service portfolios, and active
lane slices. Rejected and operationally failed updates change none of those
routing states.

On refresh failure, restore prompts, profiles, accepted counters, member wait,
anchors, freeze state, eligibility, routing, service/active portfolios, versions,
refresh count, and affected audit rows.

The persistent scheduling tie-break is:

```text
updates_since_selected_by_agent
```

Repairability failure streaks, frozen snapshots, accepted updates by other
members, freeze counts, anchors, service assignments, repair lanes, and active
slices are checkpointed exactly.

## 8. Optional Extension

The formal default is `proposal_memory_mode = off`.

Explicit `state_local_v1` Proposal Memory is a proposal-search extension. Its
complete key includes run, team state, target agent, prompt, and eligible
residual set. It may retain sanitized failure feedback only; the compact S5
context never exposes that feedback. It cannot alter:

```text
repair eligibility
responsibility portfolio
target Pareto
candidate acceptance
```

Historical owner, frontier, age, and compensation reports remain development
evidence tied to their original commits. They do not define v11.

## 9. Experiment Settings

The repository exposes exactly:

```text
shared_baseline
shared_independent_accuracy
shared_peer_state_vote_first
shared_peer_state_member_first_safe
shared_member_aware_responsibility
shared_member_aware_full
```

The settings have the following structure:

```text
S1-S3
    common monotone-safe update with different proposal/selection preferences

S3 -> S4
    add Member-Aware Responsibility scheduling

S4 -> S5
    add Responsibility-Conditioned Evolution context
```

S2 and S3 share the same safe feasible set. S2 ranks feasible candidates by
vote geometry; S3 ranks them by minimum member gain, then vote, then target
gain. Outside unique-weak-target updates, S3 normally reduces to vote followed
by target gain. The legacy key `shared_peer_state_member_pareto` is accepted as
an input alias only; new metadata records
`shared_peer_state_member_first_safe`.

All settings keep matched initialization, candidate budgets, five agents,
plurality voting, and tie-as-abstain.

`shared_member_aware_responsibility` and `shared_member_aware_full` share the
same legal eligibility, unique routing, anchors, active-lane scheduler, freeze,
and monotone target-or-vote acceptance. Their only method difference is generic peer-state
versus compact single-lane proposal context. The baseline and first three
optimization ablations do not create service assignments, anchors, active
lanes, or freeze state.

## 10. Final Active State And Test Isolation

The lifecycle is:

```text
initial team
→ planned train updates or all-actionable-members-frozen early stop
→ final active team
→ one test
```

Validation split hashes remain in run identity but validation has no role in
target selection, diagnosis, proposal, acceptance, early stopping, or
checkpoint selection. Test runs once only after the optimization lifecycle completes
and cannot influence training.

Formal selection uses integer counts. Cross-task reports additionally expose
normalized accuracy gains.

## 11. Persistence And Reproducibility

Checkpoint version is:

```text
20
```

Checkpoint state includes active prompts and profiles, initial profiles,
member-gain state, team/responsibility versions, eligibility sets, member wait,
accepted counts, target-attempt counts, seeded ranks, proposal-memory state
when explicitly enabled, TCS recovery state, training lifecycle, histories,
LLM accounting, Python random state, specialization anchors, residual repair
lanes, unique service assignments, service portfolios, active lanes, and active
residual hashes.

Per-residual age and target Pareto-front state are not persisted. Recomputed
eligibility must match the stored legal eligibility; checkpointed routing and
active slices must remain legal and deterministic for the restored state.

Checkpoint v18 and earlier fail with an explicit version mismatch. There is no
silent migration or restart in place.

Resume also requires exact run identity, code commit, split files, question
sets, probe, model request, parser, decoding, and output-contract identities.

## 12. Artifacts

Responsibility audit records, for every vote-wrong residual:

```text
question hash
candidate (DeltaV, DeltaM) values
eligible agent IDs
eligibility tie count
coverage/conversion diagnosis
```

Target-selection audit records:

```text
agent_id
D_i
S_i
g_i
d_i
updates_since_selected
frozen
target_pareto_front
active candidate IDs
legal/service/active portfolio sizes
active lane and specialization anchor
freeze/unfreeze events
selected agent
selection stage
```

`service_routing_audit_sanitized.jsonl` records one safe row per residual and
team state, including lane, legal and active eligible IDs, unique service
agent or freeze block, anchor-match level, pre-routing loads, and seeded rank.
`specialization_anchor_trajectory_sanitized.jsonl` records initialization,
accepted set/switch, retained rejection, freeze, and unfreeze-clear events.
Neither artifact contains prompts, questions, answers, or model output.

Candidate decisions record canonical acceptance and selection policies,
target and vote gains/losses, objective vectors before/after, the derived
Pareto invariant, target weakest-member status, minimum/total gain deltas,
acceptance booleans, rejection reasons, portfolio
repairs, and coverage transitions. Diagnostic error-structure metrics are not
acceptance objectives.

## 13. Implementation Map And Boundaries

```text
multi_dataset_diverse_rl/versions.py
multi_dataset_diverse_rl/member_objectives.py
multi_dataset_diverse_rl/peer_state.py
multi_dataset_diverse_rl/responsibility.py
multi_dataset_diverse_rl/diagnosis_aggregation.py
multi_dataset_diverse_rl/tcs.py
multi_dataset_diverse_rl/candidate_selection.py
multi_dataset_diverse_rl/evaluation/fixed_probe.py
multi_dataset_diverse_rl/system.py
multi_dataset_diverse_rl/persistence/checkpoint.py
multi_dataset_diverse_rl/persistence/identity.py
scripts/run_task_level_accuracy.py
scripts/preflight_member_aware.py
tests/
```

Do not change the following as part of responsibility maintenance:

```text
five-agent setting
plurality voting
tie-as-abstain
G/H/M
fixed peers
TCS retry protocol
Stage A/B candidate budget
Solver contract
no validation selection
final active state
test once
target-or-vote candidate acceptance
```

Do not add a scalar responsibility score, generic diversity reward,
same-wrong objective, prompt-distance objective, another target Pareto,
per-example hard preservation guard, or proposal-success predictor.
