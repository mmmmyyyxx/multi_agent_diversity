# Method

## 1. Scope

The current method is **Member-Aware Prompt-Team Optimization**:

```text
method_version = member_aware_peer_state_v8
```

It jointly searches a team of five prompts. Solver, optimizer, and evaluator
model weights remain frozen. Five equal-weight outputs are aggregated by
plurality vote, and a top-count tie abstains.

The paper method has exactly three modules:

```text
1. Member-Aware Responsibility
2. Responsibility-Conditioned Evolution
3. Pareto-Constrained Team Update
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
Paired Pareto Team Update
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

Teacher-Critic-Student, Stage A/B, fixed-peer rollout, max-wait, retry, cache,
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

### 4.2 Compact member portfolios

For each member:

```text
R_i = {x : i in E_x}
```

The two formal portfolio aggregates are:

```text
D_i = number of residuals in R_i with DeltaV_i,x = 1
S_i = sum over x in R_i of DeltaM_i,x
```

Portfolio size, coverage count, conversion count, dominant-wrong count, and
soft utility may be reported but do not enter target scheduling.

### 4.3 Uplift deficit

Let:

```text
g_max = max_j g_j
d_i   = max(0, g_max - g_i - 5)
```

`d_i` affects only member-level scheduling. It never changes per-residual
eligibility.

### 4.4 One member-level Pareto

For each member with a non-empty portfolio:

```text
T_i = (D_i, S_i, d_i)
```

The scheduler computes exactly one member-level Pareto frontier. Within its
first frontier, selection uses:

```text
1. larger updates_since_selected
2. stable seeded rank
```

There is no second responsibility frontier, joint frontier, hidden scalar, or
substantive reordering by `D_i`, `S_i`, or `d_i` after the frontier is formed.

### 4.5 Max-wait safeguard

The member-level safeguard is:

```text
responsibility_max_wait_updates = 8
```

It applies only when a member has a non-empty portfolio and:

```text
updates_since_selected >= 8
```

When overdue responsible members exist, no additional Pareto frontier is
computed. They are ordered by:

```text
1. longest wait
2. larger D_i
3. larger S_i
4. larger d_i
5. stable seeded rank
```

This is a starvation safeguard, not another optimization module. The method
stores member wait only; it has no per-agent-question or per-residual age.

## 5. Module 2: Responsibility-Conditioned Evolution

Different target members receive different residual portfolios and therefore
different proposal contexts. The default member-aware context contains:

```text
target current prompt
target member gain
uplift deficit
direct-fix responsibility summary
margin-gain responsibility summary
coverage residuals
conversion residuals
preservation evidence
representative evidence
```

It does not expose repair-front numbers, multiple target-front numbers,
per-residual age, ownership competition, frontier overlap as an objective, or
catch-up information in default mode.

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

Programmatic aggregation uses the complete fixed probe, then supplies at most
three pattern summaries and three representative evidence cases. LLM roles do
not aggregate cases, compute responsibility, predict candidate performance, or
decide acceptance.

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
contract, and requested candidate count. It returns:

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

## 6. Module 3: Pareto-Constrained Team Update

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
(V_count, g_min, g_sum) must strictly Pareto-dominate
terminal-invalid count must not increase
```

The frozen acceptance identifier remains:

```text
CANDIDATE_ACCEPTANCE_VERSION =
target_or_vote_strict_progress_v1
```

In particular, vote-only progress remains valid:

```text
target gain = 0
vote gain > 0
```

provided every other guard passes. Strict target improvement is not required.

Vote loss, soft utility, coverage, conversion, unique-correct loss, and
pivotal-correct loss remain diagnostics or late deterministic tie-break
evidence. None can make a non-Pareto candidate acceptable.

## 7. State Lifecycle

The initial team is diagnosed once. Rejected candidates do not change active
prompts, profiles, team state, or responsibility state.

An accepted target prompt/profile replacement is atomic. It increments the
team-state version and triggers exactly one diagnosis and responsibility
refresh. The refreshed state is reused until the next accepted team
transition.

On refresh failure, restore prompts, profiles, accepted counters, member wait,
eligibility, caches, versions, refresh count, and affected audit rows.

The only persistent fairness clock is:

```text
updates_since_selected_by_agent
```

## 8. Optional Extensions

The formal defaults are:

```text
member_catchup_mode = off
proposal_memory_mode = off
```

Explicit `fallback_v1` catch-up is a separately labelled research extension.
It is not in the default scheduler or main experiment and cannot borrow another
member's responsibility.

Explicit `state_local_v1` Proposal Memory is a proposal-search extension. Its
complete key includes run, team state, target agent, prompt, and eligible
residual set. It may provide sanitized failure feedback only. It cannot alter:

```text
repair eligibility
responsibility portfolio
target Pareto
max-wait
candidate acceptance
```

Historical v6/v7 owner, frontier, age, and catch-up reports remain development
evidence tied to their original commits. They do not define v8.

## 9. Experiment Settings

The repository exposes exactly:

```text
shared_baseline
shared_independent_accuracy
shared_peer_state_vote_first
shared_peer_state_member_pareto
shared_member_aware_responsibility
shared_member_aware_full
```

The core ablations map to the three modules:

```text
Member-Aware Responsibility
↔ round-robin versus compact responsibility target selection

Responsibility-Conditioned Evolution
↔ generic versus responsibility-conditioned proposal context

Pareto-Constrained Team Update
↔ individual/vote-first versus team Pareto acceptance
```

All settings keep matched initialization, candidate budgets, five agents,
plurality voting, and tie-as-abstain.

## 10. Final Active State And Test Isolation

The lifecycle is:

```text
initial team
→ fixed planned updates on train
→ final active team
→ one test
```

Validation split hashes remain in run identity but validation has no role in
target selection, diagnosis, proposal, acceptance, early stopping, or
checkpoint selection. Test runs once only after all planned updates complete
and cannot influence training.

Formal selection uses integer counts. Cross-task reports additionally expose
normalized accuracy gains.

## 11. Persistence And Reproducibility

Checkpoint version is:

```text
16
```

Checkpoint state includes active prompts and profiles, initial profiles,
member-gain state, team/responsibility versions, eligibility sets, member wait,
accepted counts, target-attempt counts, seeded ranks, proposal-memory state
when explicitly enabled, TCS recovery state, training lifecycle, histories,
LLM accounting, and Python random state.

Per-residual age and repair/target Pareto-front state are not persisted.
Member portfolios and target fronts are recomputed from the restored active
team, and recomputed eligibility must match the stored eligibility set.

Checkpoint v15 and earlier fail with an explicit version mismatch. There is no
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
overdue
target_pareto_front
selected agent
selection stage
```

Candidate decisions continue to record target and vote gains/losses, objective
vectors before/after, acceptance booleans, rejection reasons, portfolio
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
