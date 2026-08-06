# AGENTS.md

This file defines the normative semantics and engineering guardrails for the
current research method. Read it before `method.md`, `README.md`,
implementation modules, tests, or historical run artifacts.

`AGENTS.md` defines the normative method semantics. Actual runtime identifiers
must always be read from `multi_dataset_diverse_rl/versions.py`; method prose
must not override executable version identity.

Do not store the current Git commit, one-off pilot results, temporary
experimental conclusions, or migration progress in this file. Supply that
time-sensitive context in the opening message of each new Codex task.

## 1. Project Mission

This repository studies prompt-team optimization for LLM ensembles.

The project does not search for one globally best prompt and does not merely
collect several independently optimized prompts. It jointly optimizes a team
of five prompts whose outputs are aggregated by equal-weight plurality voting.

The formal method is:

```text
Member-Aware Prompt-Team Optimization
```

The central research question is:

> How can a prompt team be jointly optimized so that team voting performance
> and member competence improve together, without concentrating optimization
> on a small winning coalition?

The paper method has three modules:

```text
1. Member-Aware Responsibility
2. Responsibility-Conditioned Evolution
3. Robust Contribution Update
```

Joint voting state, Teacher-Critic-Student, Stage A/B, fixed-peer rollout,
online refresh, caching, retry, and checkpointing are diagnostic,
implementation, or reliability mechanisms inside those modules. They are not
additional research contributions.

### Normative method and runtime identity

The compact three-module method in this file is normative. The current runtime
implements it as:

```text
method_version = member_aware_peer_state_v12
checkpoint_version = 21
```

For every execution or artifact analysis, verify those and all other runtime
identifiers directly in `versions.py`.

## 2. Formal Research Objective

Let the five-prompt team be:

```text
Theta = (theta_1, ..., theta_K)
K = 5
```

For member `i`, let:

```text
c_i^0 = initial correct count on a fixed optimization probe
c_i   = current or candidate correct count on that same probe
g_i   = c_i - c_i^0
```

Define:

```text
g_min = min_i g_i
g_sum = sum_i g_i
```

Let:

```text
V_count = number of examples correctly answered by plurality vote
```

The formal team objective is:

```text
O(Theta) = (V_count, g_min, g_sum)
```

A candidate team Pareto-dominates the incumbent only when all three dimensions
are no worse and at least one is strictly better.

Under fixed-peer single-target replacement, target non-regression implies
that `g_sum` and `g_min` cannot regress, and `g_sum` changes by exactly the
target member's correct-count change. Vote non-regression therefore preserves
all three objective dimensions. For S1-S4, strict target-or-vote progress makes
team-objective Pareto improvement a derived invariant. S5 may additionally
accept a robust, objective-neutral active-lane improvement; it still cannot
regress the team objective. The objective remains an evaluation, trajectory,
and audit objective, not an additional candidate rejection guard.

`g_min` is candidate-discriminative only when the selected target is the
unique weakest member. It is not part of the common S1-S4 ranking. The uplift
deficit `d_i` in target scheduling is the weak-member mechanism with broad
operational effect.

Formal search and selection use integer correct counts. Normalized accuracies
are reporting metrics only. Do not replace the objective with a fixed weighted
sum.

The method does not directly optimize prompt distance, trace distance, generic
disagreement, or another standalone diversity reward. Member complementarity
is an intended result of differentiated repair responsibility, not a separate
optimization objective.

## 3. Three Research Modules

### Module 1: Member-Aware Responsibility

Research question:

> Which members can legitimately repair each residual team failure, and which
> eligible member should be updated now?

This module has two distinct decisions:

```text
Residual eligibility asks who can legitimately repair a residual.
Target scheduling asks which legitimate member should be updated now.
```

The separation is mandatory:

```text
eligibility != scheduling != candidate acceptance
```

Eligibility is decided per residual using only counterfactual vote correctness
and plurality-margin improvement. The complete Module-1 flow is
`E_x -> q_x -> P_i -> A_i`: legal eligibility, unique service routing,
disjoint service portfolios, and one active-lane slice. Scheduling aggregates
only the active slice at member level. Candidate acceptance is handled only by
Module 3 after empirical rollout. Routing and anchors remain mechanisms inside
Module 1, not a fourth module.

### Module 2: Responsibility-Conditioned Evolution

Research question:

> How should a member's compact repair responsibility be converted into
> testable replacement prompts?

The program constructs bounded numerical and diagnostic evidence. The
Teacher-Critic-Student pipeline turns that evidence into candidate prompts.
Rollouts, not role-model predictions, determine empirical value.

### Module 3: Robust Contribution Update

Research question:

> Which target-only replacement preserves target competence, team vote, and
> terminal validity while making robust vote or assigned-lane progress?

Only one target prompt is replaced at a time. The other four prompts and their
profiles remain fixed during paired candidate evaluation. S1-S4 use the common
target-or-vote scaffold; S5 adds assigned-lane utility, coalition contribution,
paired support robustness, and minimal edit.

## 4. Diagnostic Foundation: Joint Voting State

Joint voting state is the diagnostic foundation for Member-Aware
Responsibility. It is not a fourth paper module.

For each example:

```text
G = number of valid gold votes
H = size of the largest valid wrong-answer cluster
M = G - H
```

The plurality rule is:

```text
vote correct iff M > 0
top-count tie => abstain => incorrect
```

The system also constructs a leave-one-out `PeerVoteContext` for every target
member. Full-team voting state and leave-one-out peer state must remain
distinct typed concepts.

The diagnostic representation distinguishes:

- coverage failure: no current member supplies the gold answer;
- conversion failure: gold coverage exists but does not win the vote;
- fragile and stable correct votes;
- unique and pivotal correct members;
- members in the dominant wrong-answer cluster.

These quantities support:

```text
responsibility diagnosis
proposal context
behavior analysis
```

Coverage, conversion, dominant-wrong membership, unique/pivotal status, and
soft vote utility do not create another method module and do not add formal
dimensions to residual eligibility or target scheduling.

Do not replace `G`, `H`, and `M` with generic disagreement, textual prompt
distance, embedding diversity, or trace diversity.

## 5. Module 1: Compact Member-Aware Responsibility

### 5.1 Per-residual eligibility

Only vote-wrong examples produce team residuals. For a vote-wrong example `x`,
only members that are currently wrong on `x` can enter its eligibility set.

For every such member `i`, compute:

```text
DeltaV_i,x = counterfactual vote-correct gain
DeltaM_i,x = counterfactual plurality-margin gain
```

Define:

```text
eligibility_key(i, x) = (DeltaV_i,x, DeltaM_i,x)
```

Eligibility uses lexicographic maximization:

```text
1. direct vote flip first
2. larger plurality-margin gain second
3. retain all exact ties
```

Formally:

```text
E_x = lexicographic argmax over wrong members of (DeltaV_i,x, DeltaM_i,x)
```

One residual may therefore have multiple legitimate responsible members.
Eligibility is state-local repair legitimacy, not permanent ownership.

None of the following may influence `E_x`:

```text
member gain
uplift deficit
wait
accepted-update history
candidate-search history
Proposal Memory
coverage label
dominant-wrong label
soft utility
portfolio load
```

Coverage, conversion, dominant-wrong, unique/pivotal, and soft-utility
quantities are diagnostic or proposal-context evidence only.

### 5.2 Legal portfolios, repair lanes, and unique service

For each member:

```text
R_i = {x : i in E_x}
```

`R_i` is the legal portfolio. It may overlap across members and is retained for
eligibility audit, freeze signatures, and complete responsibility coverage.
Each residual has exactly one program-defined lane:

```text
coverage       if G_x = 0
direct_flip    if G_x > 0 and DeltaV_x = 1
margin_support otherwise
```

Dominant-wrong remains a diagnostic label and preservation remains separate
protection evidence. Neither is another repair lane.

For each residual, deterministic routing chooses exactly one service agent
`q_x in E_x`. Frozen members are excluded whenever any unfrozen eligible
member exists. If all eligible members are frozen, the residual is audited as
`service_blocked_by_freeze` and enters no service portfolio. Routing prefers:

```text
matching anchor
then no anchor
then a different anchor
then lower lane load
then lower total service load
then stable seeded rank
```

The service portfolios `P_i = {x : q_x = i}` are pairwise disjoint and cover
every currently serviceable residual.

### 5.3 Specialization anchor and active lane

Each S3-S5 member has `specialization_anchor_by_agent[i]`, initially `None`.
Only an accepted update sets or switches the anchor to the active repair lane
used by that update. Rejection and operational failure retain it. Unfreezing
clears it.

If the anchored lane is non-empty in `P_i`, it remains active. Otherwise choose
one lane lexicographically by `(D_i^lane, S_i^lane, N_i^lane, laneRank)`, with
`direct_flip > coverage > margin_support` only as the final exact tie-break.
The active slice is `A_i = {x in P_i : lane(x) = active_lane_i}`.

### 5.4 Uplift deficit

Let:

```text
g_max = max_j g_j
d_i = max(0, g_max - g_i - 5)
```

`d_i` affects member-level target scheduling only. It never changes
per-residual eligibility.

### 5.5 One member-level target Pareto

For every unfrozen member with a non-empty service portfolio and active slice,
aggregate only `A_i`:

```text
D_i = direct vote-fix count in A_i
S_i = sum of counterfactual margin gains in A_i
T_i = (D_i, S_i, d_i)
```

Compute exactly one Pareto frontier across those member vectors. There is no
nested or per-residual Pareto mechanism in the compact method.

Within the first member-level frontier, select deterministically using:

```text
1. larger updates_since_selected
2. stable seeded rank
```

Formally:

```text
A_t = {i : P_i and A_i are non-empty and frozen_i = false}
T_i = (D_i, S_i, d_i), i in A_t
F_t = ParetoFront({T_i : i in A_t})
i_t = lexicographic argmax over i in F_t of
      (updates_since_selected_i, -seeded_rank_i)
```

Do not add coverage, portfolio size, history, or an implicit weighted score to
`T_i`.

### 5.6 State-Conditioned Repairability Freeze

The active set is exactly the members with non-empty service and active-lane
portfolios that are not frozen. `d_i` is the sole weak-member protection term. Compute one
Pareto frontier over `(D_i, S_i, d_i)` for that active set; within the first
frontier use larger `updates_since_selected`, then stable seeded rank.

Two consecutive complete failures under the same sanitized portfolio signature
freeze a member. A complete failure requires a formally selected target, the
normal proposal and candidate budgets to finish, no infrastructure, protocol,
or parser failure, and no accepted candidate. Each update increments a streak
at most once.

Frozen members remain in eligibility, legal portfolios, anchors, and audits but cannot receive
new service or enter the active target pool. They unfreeze only after at least two accepted updates
by other members and a material portfolio change: residual-hash Jaccard below
`0.8` or changed `D_i`. A small `S_i` change alone is insufficient.

Rejected updates do not refresh routing. On an accepted team transition,
refresh legal eligibility, update freeze/unfreeze state, clear newly unfrozen
anchors, then rebuild unique routing, service portfolios, and active slices.

There is no waiting-time override, deficit-service lane, or generic
compensation mechanism. If all non-empty portfolios belong to frozen members,
stop optimization with `early_stop_reason = all_actionable_members_frozen`.

## 6. Optional Research Extension

The following extensions are outside the default compact three-module method.
They must be explicitly configured, separately labelled, and interpreted as
extensions rather than established method contributions.

### 6.1 Proposal Memory

The normative default is:

```text
proposal_memory_mode = off
```

An explicit `state_local_v1` mode is a proposal-search extension only. It may
retain sanitized historical failure feedback solely for the same complete
key, including the same agent, team state, prompt hash, and eligible residual
set. The compact S4/S5 context never exposes that feedback; it remains available
only to proposal contexts whose declared schema permits it and to audit state.

Proposal Memory must never affect:

```text
repair eligibility
legal and service portfolios
repair lane and specialization anchor
target scheduling
candidate acceptance
```

## 7. Module 2: Responsibility-Conditioned Evolution

The stable division of labor is:

```text
Program:
    compute and aggregate numerical and typed diagnostic evidence

Teacher:
    propose one bounded, testable repair hypothesis

Critic:
    check only hard semantic blockers

Student:
    realize the approved repair plan as replacement prompts

Rollouts:
    determine empirical candidate value
```

Teacher, Critic, and Student must not calculate vote counts, Pareto metrics,
responsibility scores, candidate accuracy, or final candidate value.

The S4/S5 responsibility-conditioned context includes only:

```text
parent prompt
one program-selected repair lane and fixed repair goal
one dominant (repair lane, target error role) pattern
at most two same-lane repair cases
at most one independent preservation case
compact previous status and one main rejection
```

It must not expose member identity, gain, uplift deficit, responsibility
aggregates, vote/peer-state numbers, proposal-failure feedback, routing loads,
seeded rank, freeze/anchor state, or all rejection reasons. The LLM does not
choose the lane, route residuals, compare members, or interpret scheduler
scores.

Programmatic aggregation uses the complete fixed optimization probe. S4/S5
provides exactly:

```text
1 repair pattern
at most 2 repair cases
at most 1 preservation case
min(config.tcs_context_max_chars, 6000) serialized characters
```

If necessary, remove the preservation case first and then the second repair
case. Never truncate the parent prompt, question, gold answer, or repair goal.
S3 uses the same eligibility, routing, anchors, active-lane scheduler, freeze,
and acceptance but retains generic peer-state TCS context. S1-S2 use none of
routing, anchors, active lanes, or freeze.

Do not use an LLM to cluster or numerically aggregate cases. The context bounds,
strict schemas, Student retry, and one upstream regeneration are bounded
proposal implementation, not separate research contributions.

### Teacher output

```json
{
  "failure_pattern": "...",
  "repair_rule": "...",
  "preservation_rule": "..."
}
```

Teacher proposes one concise, executable repair plan.

### Critic output

```json
{
  "failed_checks": [],
  "risk_case_ids": [],
  "feedback": ""
}
```

Allowed hard checks are:

```text
evidence_mismatch
actionable_specificity
shortcut_or_copying
preservation_or_output_risk
```

Critic approves when `failed_checks` is empty. It must not restate every case,
produce numerical scores, predict accuracy or vote gain, reproduce
program-known audit facts, or return long soft-concern essays.

After a valid Critic rejection, Teacher revision is stateless but grounded.
The next request includes the complete previous repair plan, `failed_checks`,
`risk_case_ids`, Critic feedback, and the same bounded diagnosis context. The
revision protocol requires replacement values for all three Teacher fields and
cumulative satisfaction of all hard checks.

### Student output

```json
{
  "candidate_prompts": [
    "...",
    "..."
  ]
}
```

Student sees only:

```text
parent prompt
approved repair plan
task output contract
requested candidate count
```

Student must not receive raw optimization examples or gold answers.

A zero-valid Student response receives structured rejection feedback and up to
three retries within the same approved-plan cycle. After four invalid calls,
the program allows exactly one fresh Teacher-Critic regeneration and one final
four-call Student cycle. A partially valid response stops recovery immediately,
and only its valid candidates may enter Stage A. The total bound is two cycles
and eight Student calls.

`PreviousUpdateOutcome` must distinguish pipeline execution from empirical
evaluation. Only a candidate that reached Stage A may produce model-facing
acceptance, deltas, or rollout rejection reasons. Transport, truncation, and
schema failures remain audit-only terminal failures and expose
`empirical_feedback_available=false` to the next Teacher.

## 8. Module 3: Robust Contribution Update

Only the target prompt is replaced. The other four active prompts and profiles
remain fixed during paired candidate evaluation.

Candidate evaluation computes:

- target correct and terminal-invalid counts;
- team vote-correct count;
- all five member correct counts;
- member gains relative to the initial team;
- vote gains and losses;
- residual, coverage, and conversion repairs;
- unique-correct and pivotal-correct gains and losses;
- soft vote utility.

### 8.1 Matched Stage A

Every main optimized setting uses `stage_a_policy=matched_all_generated` and
requires `stage_b_candidate_budget >= num_candidates_per_parent`. Every valid
generated candidate enters full paired evaluation. Diagnostic channels may be
recorded but cannot filter the main ablation matrix.

### 8.2 Stage B acceptance

Every acceptable S1-S4 candidate must satisfy all of:

```text
candidate target correct count >= incumbent target correct count
candidate vote correct count >= incumbent vote correct count
target correct count or vote correct count must strictly improve
terminal-invalid count must not increase
```

Whenever the first three guards pass, the candidate must strictly
Pareto-dominate the incumbent in `(V_count, g_min, g_sum)`. This is a fail-fast
derived invariant regardless of the terminal-invalid guard, not an independent
hard check, rejection reason, or ranking dimension.

A vote-only update remains valid when:

```text
target gain = 0
vote gain > 0
```

provided every other guard passes. Do not replace target-or-vote progress with
a strict-target-improvement requirement.

The acceptance identifier is:

```text
CANDIDATE_ACCEPTANCE_VERSION =
fixed_peer_monotone_target_or_vote_v2
```

Soft utility, coverage, conversion, vote loss, and unique/pivotal loss are
diagnostic or late tie-break evidence and are not independent hard rejection
guards.

S1-S4 use the same `common_monotone_safe` preference:

```text
1. larger vote-correct count
2. larger target correct count
3. larger mean soft vote utility
4. fewer vote losses
5. fewer invalid outputs
6. earlier generation
7. stable prompt hash
```

It excludes `g_min`, `g_sum`, uplift deficit, assigned-residual repair,
active-lane utility, and coalition contribution. S5 replaces only candidate
acceptance/ranking with RCRU:

```text
Layer 1:
    target, vote, and terminal-invalid non-regression
Layer 2:
    active-lane utility non-regression
    strict vote or active-lane utility progress
    leave-one-out coalition contribution and (V,U,C) Pareto
Layer 3:
    multi-example support and paired-bootstrap LCB for lane-only progress
    minimal edit as the final tie-break
```

## 9. End-to-End Method Flow

```text
Initial five-prompt team
    ↓
Fixed optimization-probe rollout
    ↓
Joint voting diagnosis: G, H, M and member gains
    ↓
Counterfactual vote-margin eligibility sets
    ↓
Member portfolios and compact target Pareto over (D, S, d)
    ↓
Responsibility-conditioned candidate generation
    ↓
Target-only paired rollout with four fixed peers
    ↓
Matched-all-generated Stage A
    ↓
Stage B monotone feasibility and derived Pareto assertion
    ↓
Atomic prompt/profile commit
    ↓
Refresh diagnosis and responsibilities after an accepted update
```

Rejected candidates do not change the active team, profiles, member state, or
responsibility state.

An accepted update must be atomic. On any refresh failure, restore:

- current and previous prompts;
- the active profile;
- accepted-update counters;
- responsibility and member state;
- cached eligibility sets, portfolios, and opportunities;
- team/responsibility versions and refresh count;
- affected peer, responsibility, and target audit records.

Diagnosis and responsibility are refreshed exactly once after a real accepted
team transition. Re-reading current state must be idempotent. This lifecycle
rule does not create residual ownership or per-residual scheduling age.

## 10. Validation and Test Isolation

The active final-state lifecycle performs no validation rollout or checkpoint
selection. Validation split hashes remain in run identity only. The final
active team after the planned budget or a valid repairability early stop is the
selected team.

Test data must never influence:

- target selection;
- residual eligibility;
- legal/service portfolios and active lanes;
- Teacher, Critic, or Student context;
- candidate acceptance;
- final-state selection.

Test runs exactly once, only after the optimization lifecycle completes its
budget or valid all-actionable-members-frozen stop, and only for the final
active team. Optimized runs do not separately evaluate their initial
team on test; a frozen matched baseline supplies the reporting-only initial-test
reference.

Final matched reports must distinguish:

```text
initial test team
final active test team
correct-count member gains
normalized accuracy member gains
```

Formal selection continues to use integer counts. Cross-task reports must
include:

```text
minimum_member_correct_count_gain
mean_member_correct_count_gain
minimum_member_accuracy_gain
mean_member_accuracy_gain
```

## 11. Experiment Protocols and Ablations

New runs use exactly these canonical protocol names:

```text
shared_baseline
shared_generic_evolution
shared_vote_state_diagnosis
shared_member_aware_responsibility
shared_responsibility_conditioned_evolution
shared_full_rcru
```

Their cumulative module vectors are exactly:

```text
S0 00000
S1 10000
S2 11000
S3 11100
S4 11110
S5 11111
```

### `shared_baseline`

- no optimization;
- shared initial prompt;
- initial-team reference.

Research question: Is any optimization better than the shared initial team?

### `shared_generic_evolution`

- round-robin target selection;
- individual-error evidence;
- generic accuracy proposal;
- the common fixed-peer monotone-safe update.

Research question: Is generic prompt evolution useful?

### `shared_vote_state_diagnosis`

- round-robin target selection;
- G/H/M and leave-one-out peer-state proposal evidence;
- the same target and candidate decision as S1.

Research question: Does structured vote-state diagnosis improve proposals?

### `shared_member_aware_responsibility`

- compact counterfactual eligibility;
- unique service routing, specialization anchors, and active-lane scheduling;
- compact target scheduling over active-slice `(D_i, S_i, d_i)`;
- State-Conditioned Repairability Freeze;
- generic peer-state proposal context;
- the same common candidate update as S1/S2;
- online responsibility refresh.

Research question: Does Member-Aware Responsibility improve target and
residual attribution?

### `shared_responsibility_conditioned_evolution`

- the same eligibility, routing, anchors, active-lane scheduling, and freeze;
- compact single-lane responsibility-conditioned proposal context;
- the same monotone target-or-vote acceptance;
- online responsibility refresh.

Research question: Does responsibility-conditioned context improve proposal
realization and final team performance?

### `shared_full_rcru`

This S5 full setting inherits S4 eligibility, routing,
anchors, active-lane scheduling, freeze, compact context, candidate generation,
and budgets. It replaces only candidate acceptance and ranking with the
three-layer Responsibility-Conditioned Robust Contribution Update:

```text
Layer 1: target, vote, and terminal-validity safety
Layer 2: active-lane utility plus leave-one-out coalition contribution
Layer 3: paired support robustness plus minimal token edit
```

Its candidate Pareto vector is `(V, U_i, C_i)`.

The setting progression is:

```text
S0 -> S1
↔ add Generic Prompt Evolution

S1 -> S2
↔ add Vote-State Diagnosis

S2 -> S3
↔ add Member-Aware Responsibility scheduling

S3 -> S4
↔ add Responsibility-Conditioned Evolution context

S4 -> S5
↔ replace only candidate decision with RCRU
```

S1-S4 share the exact acceptance policy
`fixed_peer_monotone_target_or_vote`, ranking `common_monotone_safe`, and
`matched_all_generated` Stage A. Vote-First and Member-First names are explicit
legacy controls only; they require `allow_legacy_setting=1` and never alias to
new main semantics.

Catch-up and Proposal Memory are not part of the default main-experiment
settings.

Do not silently add settings. Every new setting must isolate one research
hypothesis and use matched candidate and evaluation budgets.

## 12. Code Map

The map below describes semantic ownership in the current compact
implementation.

### Central orchestration

```text
multi_dataset_diverse_rl/system.py
```

Owns prompts and profiles; builds the protocol; initializes probes;
orchestrates diagnosis, responsibility, bounded proposal, Stage A/B, and final
test; manages atomic updates; and writes method-level audits.

Do not move all domain logic into `system.py`. Pure calculations belong in
their domain modules.

### Runtime versions

```text
multi_dataset_diverse_rl/versions.py
```

This is the source of truth for actual runtime identifiers. The compact method
uses checkpoint version 21; implementation, persistence, and tests must remain
consistent with it.

### Configuration and protocols

```text
multi_dataset_diverse_rl/config.py
multi_dataset_diverse_rl/protocol.py
```

`config.py` contains typed configuration sections. Every behavior-affecting
field needs a real read point and must enter the run-identity fingerprint.
Avoid flags for behavior that should be fixed by method semantics.

`protocol.py` is the source of truth for setting differences through
`ExperimentProtocol`, `CandidateBudgetContract`, and `InitializationMode`.
Do not infer settings through unrelated boolean combinations.

### Joint voting diagnosis

```text
multi_dataset_diverse_rl/peer_state.py
multi_dataset_diverse_rl/diagnosis_aggregation.py
```

`peer_state.py` defines full-team `TeamVoteState`, leave-one-out
`PeerVoteContext`, `G`, `H`, `M`, plurality results, and soft vote utility.

`diagnosis_aggregation.py` analyzes the complete fixed probe. Generic contexts
retain configured typed aggregation; S5 deterministically selects one lane
pattern, at most two repair cases, and at most one preservation case. These files provide the diagnostic foundation;
they do not define a separate paper module.

### Member objectives

```text
multi_dataset_diverse_rl/member_objectives.py
```

Defines `MemberGainMetrics`, `TeamMemberGainState`, `TeamObjectiveVector`,
Pareto dominance, and fronts. Formal selection uses integer counts. Candidate
team-objective dominance is a derived fixed-peer invariant; member-level target
Pareto scheduling remains active.

### Responsibility

```text
multi_dataset_diverse_rl/responsibility.py
```

The responsibility module defines:

- counterfactual vote-margin repair opportunities;
- per-residual eligibility sets;
- overlapping legal portfolios and unique service portfolios;
- specialization anchors and active-lane slices;
- compact active-slice `(D_i, S_i, d_i)` target scheduling;
- State-Conditioned Repairability Freeze.

There must be one lexicographic eligibility decision per residual and one
member-level target Pareto frontier. Coverage and dominant-wrong labels remain
diagnostic; no generic compensation path exists.

### Responsibility-conditioned proposal

```text
multi_dataset_diverse_rl/tcs.py
```

Defines isolated diagnosis contexts, Teacher/Critic/Student response types,
request builders, strict parsers, sample-memorization checks, revision, and
bounded invalid recovery. Preserve context isolation across settings.

### Candidate selection

```text
multi_dataset_diverse_rl/candidate_selection.py
```

Remains the source of truth for monotone target-or-vote acceptance and the
derived team-objective invariant. It also owns candidate and constraint
decisions, Stage A channel keys, vote-first-safe selection, member-first-safe
selection, and deterministic preference. It must not generate
`member_objective_regression` for new candidate decisions or filter feasible
candidates through a team-objective Pareto front.

Core decision paths must use typed fields. Do not interpret a missing metric as
zero.

### Evaluation and caches

```text
multi_dataset_diverse_rl/evaluation/fixed_probe.py
multi_dataset_diverse_rl/evaluation/validation.py
multi_dataset_diverse_rl/evaluation/prompt_question.py
multi_dataset_diverse_rl/evaluation/persistent_solver_cache.py
```

Fixed-probe evaluation replaces one target with four fixed peers and computes
target, team, member, residual, and protection metrics.

Validation utilities compute `DatasetMetrics` but are not used for active-state
selection. The same prompt, question, model request, parser, output contract,
temperature, and seed must map to the same observation across matched settings.
Do not put the experiment setting in the shared Solver-cache key.

### Solver contract

```text
multi_dataset_diverse_rl/evaluation/output_contract.py
multi_dataset_diverse_rl/evaluation/solver_output.py
```

The Solver must produce exactly one valid:

```text
FINAL_ANSWER: <answer>
```

The optimized prompt is only the mutable decision procedure. The program
appends the immutable task output interface in every Solver request:

```text
Follow the decision procedure below.

Decision procedure:
<mutable candidate prompt>

Mandatory output interface:
This interface is immutable and overrides any conflicting instruction above.
<strict task-specific FINAL_ANSWER contract>
```

The request-template version is part of Solver request and shared-cache
identity. Student sees the output contract to avoid conflicts, but reproducing
the immutable interface is not part of prompt search.

Every mutable prompt must pass the shared deterministic contract validator at
initial load, checkpoint restore, candidate parsing, accepted-state commit, and
Solver construction. Mutable prompts must not contain any recognizable
`FINAL_ANSWER` marker, copied Solver-interface heading, fixed answer payload, or
final-response formatting instruction. Reject contaminated candidates without
stripping or rewriting them. A protocol-only proposal failure must not advance
repairability-freeze state.

Do not loosen the parser to hide model-output failures. Invalid recovery occurs
before the prompt-question cache stores an observation. It retries only strict
parser-invalid results with identical requests, stops at the first valid
result, and stores only the resolved result. Formal invalid guards use terminal
invalid count.

### LLM access

```text
multi_dataset_diverse_rl/llm_client.py
```

Owns role endpoints, post-hoc token/call accounting, transient retries, and
timeouts. Transport retry and semantic proposal revision are distinct control
flows.

The canonical default credential environment is `DASHSCOPE_API_KEY` for all
three roles. The canonical OpenAI-compatible endpoint is the project-scoped
Beijing Model Studio URL declared in `provider_credentials.py`; an explicit
`DASHSCOPE_BASE_URL` environment value may override it. Never persist or print
the API key.

Teacher, Critic, and Student outputs are bounded structurally rather than by
experiment-level completion-token budgets. S5 uses one repair pattern, at most
two repair cases, at most one preservation case, and a 6000-character context
cap. Keep `solver_max_tokens=1800`
unchanged so Solver request identity and shared-cache semantics remain stable.
Treat provider `finish_reason=length` as an audited runtime failure, not as
evidence of method efficacy.

### Persistence

```text
multi_dataset_diverse_rl/persistence/identity.py
multi_dataset_diverse_rl/persistence/checkpoint.py
multi_dataset_diverse_rl/persistence/artifacts.py
```

Owns exact run identity, behavior fingerprint, atomic checkpointing,
incompatible-checkpoint rejection, and artifacts. The current checkpoint
version is 18; inspect `versions.py` for executable identity.

Do not add compatibility code for obsolete method versions unless explicitly
requested.

### Tasks, scripts, and tests

```text
multi_dataset_diverse_rl/tasks.py
configs/task_level_comparison_strict_bbh_seed42.yaml
scripts/run_task_level_accuracy.py
scripts/preflight_member_aware.py
scripts/deterministic_member_objective_unit_smoke.py
scripts/deterministic_member_aware_system_smoke.py
scripts/deterministic_member_aware_smoke.py
tests/
```

Task specifications define answer parsing and matching. Optimization,
validation, and test splits must remain disjoint.

No external API experiment may start unless the user explicitly asks in the
current task. Implementation work and real-API testing are separate tasks by
default. A request to "test", "verify", or "finish" means offline verification
unless real API usage is explicitly stated.

## 13. Engineering Invariants

The following invariants are mandatory unless the user explicitly changes the
research method:

```text
exactly five agents
frozen model weights
plurality aggregation
tie-as-abstain
integer formal objectives
single-target candidate replacement
four fixed peers during candidate comparison
strict optimization/validation/test separation
strict FINAL_ANSWER contract
exact run identity
deterministic same-seed behavior
atomic accepted update
one diagnosis/responsibility refresh per accepted team transition
test only after the optimization lifecycle completes
one per-residual eligibility decision based only on (DeltaV, DeltaM)
one unique service agent per serviceable residual in S3-S5
one active repair lane per serviceable member in S3-S5
one member-level target Pareto over active-slice (D_i, S_i, d_i)
accepted-update-only specialization anchors
state-conditioned repairability freeze in S3-S5 only
proposal_memory_mode = off by default
vote-only and target-only monotone acceptance preserved
```

No accepted candidate may silently bypass these invariants.

## 14. Anti-Bloat Rules

The paper method has three modules. Implementation details must not be promoted
into additional contributions.

Do not add:

- multiple nested Pareto mechanisms;
- per-residual scheduling age;
- permanent ownership or owner inertia;
- hidden weighted responsibility scores;
- coverage or dominant-wrong dimensions in formal eligibility;
- coverage or portfolio size in the formal target vector;
- waiting-time or deficit-service compensation lanes;
- default Proposal Memory;
- new agents or roles without a new research question;
- duplicate settings or duplicate metric implementations;
- generic plugin abstractions;
- archive or beam search;
- MAP-Elites;
- embedding diversity objectives;
- prompt-text or trace-distance rewards;
- extra LLM critics after candidate rollout;
- compatibility wrappers for deleted semantics;
- configuration flags without a real methodological purpose.

Caching, retry, checkpointing, audit, and concurrency are experiment
reliability infrastructure. Do not describe them as method contributions.

Prefer one clear domain type over a generic dictionary. Do not reuse a class or
function when its name and semantics conflict with the compact method merely
to minimize the diff.

## 15. Required Workflow for Every Codex Task

Before editing:

```text
1. Read AGENTS.md.
2. Read method.md and README.md.
3. Inspect current git HEAD, latest commit, and working-tree status.
4. Inspect versions.py, config.py, and protocol.py.
5. Read the exact modules involved in the requested change.
6. Inspect relevant tests and recent compact experiment reports.
7. Separate normative target semantics, confirmed runtime implementation,
   requested changes, and assumptions.
```

Before implementation, identify:

```text
research question affected
current code semantics
desired target semantics
files that should change
files that should not change
tests proving semantic alignment
```

During implementation:

- make the smallest coherent semantic change;
- delete superseded logic rather than leave dormant branches;
- preserve strict data isolation and existing valid infrastructure;
- avoid unrelated refactors;
- do not overwrite historical experiment directories;
- keep code modification and real-API testing separate by default;
- do not run external APIs without explicit authorization in the current task.

All Python execution in this Windows workspace uses the `DL` Conda
environment. Prefer:

```text
D:\Anaconda\envs\DL\python.exe
```

For code changes, run the relevant task-specific tests and, unless the current
task explicitly narrows verification, run:

```powershell
D:\Anaconda\envs\DL\python.exe -m pytest -q
D:\Anaconda\envs\DL\python.exe -m compileall -q multi_dataset_diverse_rl scripts
D:\Anaconda\envs\DL\python.exe scripts\preflight_member_aware.py --workspace . --allow_dirty 1
D:\Anaconda\envs\DL\python.exe scripts\deterministic_member_objective_unit_smoke.py
D:\Anaconda\envs\DL\python.exe scripts\deterministic_member_aware_system_smoke.py
git diff --check
```

Documentation-only tasks may use documentation-specific validation when the
user explicitly excludes code tests. LF/CRLF warnings on Windows are
informational; `git diff --check` remains the whitespace gate.

## 16. Completion Report

A Codex completion report must include:

```text
starting commit
ending commit
files changed
method semantics changed
method semantics deliberately unchanged
old code or stale documentation deleted
new or updated tests
pytest result, or explicit reason not run
compileall result, or explicit reason not run
preflight result, or explicit reason not run
smoke result, or explicit reason not run
git diff --check result
external API calls performed or not performed
remaining real-API risks
working-tree state
commit and push state
```

Do not report only "tests passed." Explain how the change maps to the affected
research question and distinguish target specification from runtime
implementation status.

## 17. Historical Methods and Results

Historical pilot and smoke directories are evidence, not current method code.
In particular, v6 and v7 owner/frontier/compensation reports are historical
development evidence. They do not define the active compact method.

Historical terms such as `repair-frontier`, `primary owner`, `owner_age`,
`oldest responsibility age`, `responsibility_pareto_front`,
`joint_target_pareto_front`, and
`frontier_responsibility_and_catchup_context_v2` are deprecated semantics, not
instructions for the compact target method.

Do not delete or rewrite historical reports to make them appear consistent
with the current target. Interpret every historical experiment according to
its original:

```text
commit
method version
configuration
artifact schema
```

Do not:

- infer current behavior from old artifact names;
- restore old settings for compatibility;
- overwrite old results;
- compare unmatched commits as a formal method comparison;
- reinterpret an operationally failed run as a method efficacy result;
- retroactively apply current compact semantics to an older run.

When analyzing a run, first verify:

```text
git commit
method version
experiment protocol
dataset hashes
solver request identity
candidate funnel reached
final-state selection completed
```

Raw run directories, SQLite databases, LLM logs, and checkpoints remain
ignored unless the user explicitly requests otherwise. Prefer compact,
secret-free Markdown and JSON reports.

Keep local raw run roots under `experiments/runs_*`; do not create new raw
`runs_*` directories directly at the repository root. Historical report paths
remain evidence of their original locations and must not be rewritten merely
because the local raw directories were reorganized.

## 18. Required Sanitized Research Artifacts

Every optimization pilot uploaded for analysis must include versioned,
secret-free artifacts:

```text
run_meta_sanitized.json
candidate_decisions_sanitized.jsonl
responsibility_assignments_sanitized.jsonl
member_opportunities_sanitized.jsonl
target_priority_audit_sanitized.jsonl
repairability_freeze_events_sanitized.jsonl
repairability_unfreeze_events_sanitized.jsonl
service_routing_audit_sanitized.jsonl
specialization_anchor_trajectory_sanitized.jsonl
g_transition_audit_sanitized.jsonl
specialization_trajectory_sanitized.jsonl
token_cost_breakdown_sanitized.json
final_behavior_matrices_sanitized.json
```

Each artifact declares a schema version and coverage counts. Per-example files
must be checked for question-hash uniqueness, join consistency, and expected
rows per team state. Unavailable historical fields must be reported as
`unavailable`, never as zero. Candidate-search outcomes are audit-only and do
not enter eligibility or target scheduling.

Never upload questions, gold labels, literal agent answers, prompts, raw TCS
content, API responses or credentials, absolute paths, checkpoints, caches, or
traces that can reconstruct dataset content. Question hashes, prompt hashes,
anonymized pattern identifiers, booleans, counts, aggregate statistics, Pareto
fronts, and sanitized request identities are permitted.

## 19. Git and Publication

Preserve user changes and inspect the worktree before staging. Never add
ignored raw runs, API secrets, caches, or unrelated files to a commit.

The established remote is:

```text
git@github.com:mmmmyyyxx/multi_agent_diversity.git
```

The user has historically requested direct updates to `main`. Do not create a
new branch or pull request unless the current task asks for one. Never commit
or push unless the current task explicitly authorizes it.
