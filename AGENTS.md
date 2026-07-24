# AGENTS.md

This file contains stable project memory and engineering guardrails. Read it
before `method.md`, `README.md`, implementation modules, tests, or historical
run artifacts.

Do not store the current Git commit, one-off pilot results, or the objective of
a single task in this file. Supply that time-sensitive context in the opening
message of each new Codex task.

## 1. Project Mission

This repository studies prompt-team optimization for LLM ensembles.

The project does not search for one globally best prompt and does not merely
collect several independently optimized prompts. It jointly optimizes a team
of five prompts whose outputs are aggregated by equal-weight plurality voting.

The central research question is:

> How can a prompt team be jointly optimized so that team-level performance
> and member-level competence achieve Pareto improvement, without obtaining
> gains by sacrificing or permanently ignoring some members?

The current formal method is:

```text
Member-Aware Peer-State Prompt-Team Optimization
```

The current implementation version is `member_aware_peer_state_v3`. It keeps
the v2 responsibility, TCS, Pareto, Stage A/B, and immutable Solver contract
semantics, while adding potential-aware target priority and request-local
first-valid Solver invalid recovery. Checkpoint version is 8; v7 checkpoints
are intentionally incompatible.

Read the current formal method version from:

```text
multi_dataset_diverse_rl/system.py
multi_dataset_diverse_rl/config.py
method.md
```

Do not infer current method semantics or versions from historical run
directories.

## 2. Research Problem

Let the prompt team be:

```text
Theta = (theta_1, ..., theta_K)
K = 5
```

For member `i`, let:

```text
c_i^0 = initial correct count on a fixed probe
c_i   = current or candidate correct count
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

Formal search and selection use integer correct counts. Normalized accuracies
are reporting metrics, not substitutes for the formal objectives.

Do not replace this objective with a fixed weighted scalar such as:

```text
lambda_vote * V
+ lambda_min * g_min
+ lambda_sum * g_sum
```

unless the user explicitly changes the research question.

## 3. Four Research Modules

Every method module must correspond to one research question.

### RQ1. How should joint prompt-team failures be represented?

Module:

```text
Peer-State Representation
```

For each example:

```text
G = number of valid gold votes
H = size of the largest valid wrong-answer cluster
M = G - H
```

Plurality vote is correct when `M > 0`. A top-count tie abstains and is counted
as incorrect.

The system also constructs a leave-one-out `PeerVoteContext` for every target
member. It distinguishes:

- no-gold-coverage failures;
- minority-gold but dominant-wrong failures;
- fragile and stable correct votes;
- unique and pivotal correct members;
- dominant wrong-cluster members.

Do not replace `G, H, M` with generic disagreement, textual prompt distance,
embedding diversity, or trace diversity.

### RQ2. Which member should be updated and what should it repair?

Module:

```text
Member-Aware Counterfactual Responsibility
```

For each wrong member on a vote-wrong example, the program computes:

- direct vote-fix potential;
- oracle soft-vote-utility gain;
- coverage opportunity;
- dominant-wrong membership or exit potential;
- current member improvement need.

Member improvement need is:

```text
improvement_need_i = max(0, g_sum_current - K * g_i_current)
```

This identifies members that are falling behind the team-wide improvement
trajectory, including a member that has remained unchanged while other members
improve.

Responsibility assignment combines:

```text
team-repair potential
+
member-improvement need
```

Competence preservation is not an extra responsibility dimension.

Responsibility answers:

```text
who should be updated
what residual failure should be repaired
```

Preservation answers:

```text
what must not be damaged during that update
```

Preservation is enforced through:

- preservation evidence in proposal generation;
- active and initial competence floors;
- the invalid-output guard;
- the unique-correct loss guard;
- the pivotal-correct loss guard.

All members with current individual errors remain eligible even when they own
no residual case. Target selection must be deterministic for the same seed and
must include max-wait protection against member starvation.

### RQ3. How are differentiated responsibilities converted into prompt changes?

Module:

```text
Responsibility-Conditioned Prompt Proposal
```

Teacher-Critic-Student is a proposal mechanism, not the core credit-assignment
innovation.

Different members receive different optimization signals because they own
different residual responsibilities. Complementarity should emerge from
different repair responsibilities, not from predefined personas or generic
diversity rewards.

The intended role division is:

```text
Program:
    compute and aggregate all numerical evidence

Teacher:
    propose one testable repair hypothesis

Critic:
    reject only proposals with clear hard semantic blockers

Student:
    realize the approved repair hypothesis as replacement prompts

Stage A / Stage B:
    empirically determine whether candidates are useful
```

Teacher, Critic, and Student must not calculate vote counts, Pareto metrics,
responsibility scores, candidate accuracy, or final candidate value.

### RQ4. How should a candidate prompt team be selected?

Module:

```text
Member-Aware Pareto Team Update
```

Only the target prompt is replaced. The other four active prompts and their
profiles remain fixed during paired candidate evaluation.

Candidate evaluation computes:

- target correct and invalid counts;
- team vote-correct count;
- all five member correct counts;
- vote gains and losses;
- residual and coverage repairs;
- unique-correct and pivotal-correct losses;
- member gains relative to the initial team.

A candidate must first pass hard feasibility constraints. For the formal
member-aware method, it must then strictly Pareto-dominate the incumbent in:

```text
(V_count, g_min, g_sum)
```

Soft vote utility is only a dense diagnostic and tie-break signal. It must not
make a non-dominating candidate acceptable.

## 4. End-to-End Method Flow

```text
Initial five-prompt team
    鈫?Fixed optimization-probe rollout
    鈫?TeamVoteState and leave-one-out PeerVoteContext
    鈫?Member gains and improvement needs
    鈫?Member-aware counterfactual repair opportunities
    鈫?Residual owner assignment and target-member selection
    鈫?Responsibility-conditioned prompt proposal
    鈫?Target-only candidate rollout with four fixed peers
    鈫?Stage A multi-channel shortlist
    鈫?Hard competence and preservation guards
    鈫?Stage B Pareto comparison against incumbent
    鈫?Atomic prompt/profile commit
    鈫?Exactly one responsibility refresh for the new team state
```

Rejected candidates do not change team state or responsibility state.

An accepted update must be atomic. On any refresh failure, restore:

- current and previous prompts;
- the active profile;
- accepted-update counters;
- responsibility state;
- cached ownership, assignments, and opportunities;
- team/responsibility versions and refresh count;
- affected peer, responsibility, and target audit records.

`owner_age` advances once per real team-state refresh, not once per function
call. If responsibility is already current for the team-state version,
recomputation must be an idempotent no-op.

## 5. Stage A and Stage B

### Stage A

The formal member-aware settings use three shortlist channels:

```text
team_vote
worst_member
mean_member
```

They correspond directly to the three team objectives.

`team_vote` prioritizes:

- candidate vote-correct count;
- net vote delta;
- fewer vote losses;
- soft utility;
- assigned residual repair.

`worst_member` prioritizes:

- minimum member gain;
- improvement in minimum gain;
- improved-member count;
- target gain versus incumbent;
- lower invalid count.

`mean_member` prioritizes:

- total member gain;
- target gain versus incumbent;
- improved-member count;
- assigned residual repair;
- lower invalid count.

Channel ranks are merged through deterministic Pareto-front ordering. Do not
fill the Stage B budget using arbitrary prompt-hash order except as the final
tie-break after substantive metrics are equal.

### Stage B

Hard guards include:

```text
active target competence floor
initial target competence floor
invalid-output guard
vote-loss limit
unique-correct loss limit
pivotal-correct loss limit
```

The formal method accepts only candidates that Pareto-dominate the incumbent.

The canonical preference among acceptable candidates is:

```text
1. larger minimum member gain
2. larger vote-correct count
3. larger total member gain
4. larger improved-member count
5. fewer vote losses
6. larger soft vote utility
7. more assigned repairs
8. larger target correct count
9. fewer invalid outputs
10. earlier generation
11. stable prompt hash
```

## 6. Validation and Test

Validation compares each epoch with the initial validation team.

A validation state is feasible only when:

```text
no member violates the initial competence floor
invalid rate satisfies its guard
vote-correct count is not below the initial team
```

Feasible states are ordered by:

```text
1. minimum member gain
2. vote-correct count
3. total member gain
4. improved-member count
5. soft vote utility
6. fewer C0 examples
7. lower invalid rate
8. earlier epoch
```

Test data must never influence:

- target selection;
- responsibility assignment;
- Teacher, Critic, or Student context;
- candidate acceptance;
- validation best-state selection.

The final test runs only after validation selects the prompt team. Final
reports must distinguish:

```text
initial test team
validation-selected test team
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

## 7. Experiment Protocols

The repository currently supports exactly:

```text
shared_baseline
shared_independent_accuracy
shared_peer_state_vote_first
shared_peer_state_member_pareto
shared_member_aware_responsibility
shared_member_aware_full
```

### `shared_baseline`

- no optimization;
- shared initial prompt;
- initial-team reference.

Research question: Is any optimization better than the shared initial team?

### `shared_independent_accuracy`

- round-robin target selection;
- individual-error evidence;
- generic accuracy proposal;
- individual-accuracy candidate selection.

Research question: Is independent prompt optimization sufficient?

### `shared_peer_state_vote_first`

- round-robin target selection;
- generic peer-state evidence;
- pure vote-first candidate selection.

Research question: Does vote-only selection form a narrow winning coalition?

This is a pure vote-first ablation, not an exact recreation of every historical
Peer-State version.

### `shared_peer_state_member_pareto`

- the same round-robin target policy;
- the same generic peer-state proposal context;
- member-aware Pareto candidate selection.

Research question: What is the effect of replacing vote-first selection with
team Pareto selection?

### `shared_member_aware_responsibility`

- member-aware responsibility and target selection;
- generic peer-state proposal context;
- member-aware Pareto selection;
- online responsibility refresh.

Research question: What is the effect of member-aware attribution and target
assignment?

### `shared_member_aware_full`

- the same member-aware responsibility;
- the same member-aware Pareto selection;
- responsibility-conditioned proposal context.

Research question: Does exposing assigned responsibility improve prompt
proposal quality?

Do not silently add settings. Every new setting must isolate one research
hypothesis and use matched candidate and evaluation budgets.

## 8. Small-Model Role Pipeline

Optimizer and evaluator roles may use small models. The stable principle is:

```text
Program computes and aggregates.
LLMs perform only bounded semantic tasks.
Rollouts verify empirical effects.
```

`PreviousUpdateOutcome` must distinguish pipeline execution from empirical
evaluation. Only a candidate that reached Stage A may produce model-facing
acceptance, deltas, or rollout rejection reasons. TCS transport, truncation,
and schema failures remain audit-only terminal failures and expose
`empirical_feedback_available=false` to the next Teacher.

The current aggregated role pipeline must satisfy:

```text
raw evidence cases per TCS context <= 3
aggregated failure patterns <= 3
```

Programmatic aggregation should use the full fixed probe and summarize:

- vote distributions and `G, H, M`;
- leave-one-out peer states;
- answer-role signatures;
- target correctness;
- dominant-wrong membership;
- direct vote-fix potential;
- member gains and improvement need;
- residual responsibility;
- unique/pivotal protection.

Do not use an LLM to cluster or aggregate cases.

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

Allowed hard checks:

```text
evidence_mismatch
actionable_specificity
shortcut_or_copying
preservation_or_output_risk
```

Critic is approved when `failed_checks` is empty. Critic must not:

- restate every case;
- produce numerical scores;
- predict candidate accuracy or vote gain;
- reproduce program-known audit facts;
- return long soft-concern essays.

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

This pipeline is implemented by `member_aware_peer_state_v3` with
`aggregated_small_model_tcs_v1`. Do not revert to raw parallel case lists or
expand language-model responsibilities. Do not change member objectives,
responsibility assignment, Stage A/B, validation, or experiment-setting
semantics as part of role-pipeline maintenance.

## 9. Code Map

### Central orchestration

```text
multi_dataset_diverse_rl/system.py
```

Owns prompts/profiles; builds the protocol; initializes probes; orchestrates
responsibility, TCS, Stage A/B, validation, and test; manages atomic updates;
and writes method-level audits.

Do not move all domain logic into `system.py`. Pure calculations belong in
their domain modules.

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

### Peer-state domain

```text
multi_dataset_diverse_rl/peer_state.py
```

Defines and computes `TeamVoteState`, `PeerVoteContext`, `G, H, M`, plurality
results, and soft vote utility. Full-team and leave-one-out states must remain
distinct types.

### Member objectives

```text
multi_dataset_diverse_rl/member_objectives.py
```

Defines `MemberGainMetrics`, `TeamMemberGainState`, `TeamObjectiveVector`,
Pareto dominance, and fronts. Formal selection uses integer counts.

### Responsibility

```text
multi_dataset_diverse_rl/responsibility.py
```

Defines `MemberAwareRepairOpportunity`, `ResponsibilityState`, primary owner
assignment, target priorities, target selection, and member improvement need.
Responsibility lifecycle must be versioned by real team state.

Potential-aware scheduling state is stored in `ResponsibilityState`:
`best_observed_target_gain_by_agent`, `no_positive_candidate_streak_by_agent`,
`next_regular_eligible_update_by_agent`, and `target_attempt_count_by_agent`.
These fields are checkpointed and fingerprinted. They do not alter
`updates_since_selected` attempt-aware max-wait behavior.

### TCS proposal mechanism

```text
multi_dataset_diverse_rl/diagnosis_aggregation.py
multi_dataset_diverse_rl/tcs.py
```

The aggregation module analyzes the complete fixed probe, assigns answer roles,
groups typed failure patterns, and deterministically selects at most three
patterns and cases. `tcs.py` defines isolated diagnosis contexts,
Teacher/Critic/Student response types, request builders, strict parsers,
sample-memorization checks, and context limiting. Preserve context isolation
across settings.

### Candidate selection

```text
multi_dataset_diverse_rl/candidate_selection.py
```

Defines candidate and constraint decisions, Stage A channel keys, hard
constraint checks, vote-first and member-aware selection, and Pareto
acceptability. Core decision paths must use typed fields; do not interpret a
missing metric as zero.

### Evaluation and caches

```text
multi_dataset_diverse_rl/evaluation/fixed_probe.py
multi_dataset_diverse_rl/evaluation/validation.py
multi_dataset_diverse_rl/evaluation/prompt_question.py
multi_dataset_diverse_rl/evaluation/persistent_solver_cache.py
```

Fixed-probe evaluation replaces one target with four fixed peers and computes
target, team, member, residual, and protection metrics.

Validation computes `DatasetMetrics` without leaking test information into
selection.

The same prompt, question, model request, parser, output contract, temperature,
and seed must map to the same observation across matched settings. Do not put
the experiment setting in the cache key.

### Solver contract

```text
multi_dataset_diverse_rl/evaluation/output_contract.py
multi_dataset_diverse_rl/evaluation/solver_output.py
```

The solver must produce exactly one valid:

```text
FINAL_ANSWER: <answer>
```

The optimized prompt is only the mutable decision procedure. The program must
append the immutable task output interface after that procedure in every
Solver request:

```text
Follow the decision procedure below.

Decision procedure:
<mutable candidate prompt>

Mandatory output interface:
This interface is immutable and overrides any conflicting instruction above.
<strict task-specific FINAL_ANSWER contract>
```

The request-template version is part of Solver request and shared-cache
identity. Student sees the output contract to avoid conflicts, but preserving
or reproducing the full interface is not part of the prompt search problem.

Do not loosen the parser to hide model-output failures.

Invalid recovery is implemented inside `system.solve()` before the
prompt-question cache stores an observation. It retries only strict-parser
invalid results, uses identical requests, stops at the first valid result, and
stores only the resolved result. `PromptAnswer` carries attempt audit fields;
formal invalid guards use `terminal_invalid_count`.

### LLM access

```text
multi_dataset_diverse_rl/llm_client.py
```

Owns role endpoints, post-hoc token/call accounting, transient retries, and
timeouts. Transport retry and semantic TCS revision are distinct control flows.

Teacher, Critic, and Student outputs are not truncated by experiment-level completion-token budgets. Their search space is bounded structurally through strict schemas, at most three representative cases, bounded text fields, a fixed candidate count, and prompt-length constraints. Actual token usage is recorded for post-hoc analysis but does not terminate the experiment.

Keep `solver_max_tokens=1800` unchanged so Solver request identity and shared
cache semantics remain stable. Treat provider `finish_reason=length` as an
audited runtime failure, not as evidence that the method has no gain.

### Persistence

```text
multi_dataset_diverse_rl/persistence/identity.py
multi_dataset_diverse_rl/persistence/checkpoint.py
multi_dataset_diverse_rl/persistence/artifacts.py
```

Owns exact run identity, behavior fingerprint, atomic checkpoint, incompatible
checkpoint rejection, and artifacts. Checkpoint member state must be the
target-free `TeamMemberGainState`.

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

Task specs define answer parsing and matching. Optimization, validation, and
test splits must remain disjoint.

No external API experiment may start unless the user explicitly asks in the
current task.

Implementation work and real-API testing are separate tasks by default. A task
that changes code may run offline unit tests, integration tests, deterministic
fake-model smokes, compile checks, and preflight checks, but it must not also
start a real-API smoke, pilot, or experiment. Completing an implementation does
not imply authorization to spend API calls.

Only combine code modification and real-API testing when the user explicitly
requests both in the same task. A request to 鈥渢est鈥? 鈥渧erify鈥? or 鈥渇inish鈥?means
offline verification unless real API usage is stated explicitly.

Tests must verify method semantics, not merely implementation details. Every
important method change needs a unit test, an integration test, and a
deterministic system smoke when appropriate.

## 10. Engineering Invariants

The following invariants are mandatory unless the user explicitly changes the
research method:

```text
exactly five agents
frozen model weights
plurality aggregation
tie-as-abstain
single-target candidate replacement
four fixed peers during candidate comparison
strict optimization/validation/test separation
integer formal objectives
member-wise competence guards
strict FINAL_ANSWER contract
exact run identity
deterministic same-seed behavior
atomic accepted update
one responsibility refresh per real team transition
test data used only after validation selection
```

No accepted candidate may silently bypass these invariants.

## 11. Anti-Bloat Rules

Do not turn the research method into a generic engineering framework.

Do not add:

- new agents or roles without a new research question;
- duplicate settings;
- duplicate metric implementations;
- generic plugin abstractions;
- archive or beam search;
- MAP-Elites;
- embedding diversity objectives;
- prompt-text distance rewards;
- extra LLM critics after candidate rollout;
- compatibility wrappers for deleted semantics;
- configuration flags without a real methodological purpose.

Caching, retry, checkpointing, audit, and concurrency are experiment reliability
infrastructure. Do not describe them as method contributions.

Prefer one clear domain type over a generic dictionary. Do not reuse a class or
function when its name and semantics do not match the new method merely to
minimize the diff.

## 12. Required Workflow for Every Codex Task

Before editing:

```text
1. Read AGENTS.md.
2. Read method.md and README.md.
3. Inspect current git HEAD and working-tree status.
4. Inspect config.py and protocol.py.
5. Read the exact modules involved in the requested change.
6. Inspect relevant tests and recent compact experiment reports.
7. Separate confirmed implementation from requested changes and assumptions.
```

Before implementation, identify:

```text
research question affected
current code semantics
desired code semantics
files that should change
files that should not change
tests proving semantic alignment
```

During implementation:

- make the smallest coherent semantic change;
- delete superseded logic rather than leave dormant branches;
- preserve strict data isolation;
- preserve existing valid infrastructure;
- avoid unrelated refactors;
- do not overwrite historical experiment directories;
- keep code modification and real-API testing in separate tasks by default;
- do not run external APIs without explicit authorization in the current task.

All Python execution in this Windows workspace uses the `DL` Conda
environment. Prefer the direct interpreter to avoid `conda run` encoding
problems:

```text
D:\Anaconda\envs\DL\python.exe
```

After implementation, run:

```powershell
D:\Anaconda\envs\DL\python.exe -m pytest -q
D:\Anaconda\envs\DL\python.exe -m compileall -q multi_dataset_diverse_rl scripts
D:\Anaconda\envs\DL\python.exe scripts\preflight_member_aware.py --workspace . --allow_dirty 1
D:\Anaconda\envs\DL\python.exe scripts\deterministic_member_objective_unit_smoke.py
D:\Anaconda\envs\DL\python.exe scripts\deterministic_member_aware_system_smoke.py
git diff --check
```

Also run task-specific tests for modified behavior. LF/CRLF warnings on Windows
are informational; `git diff --check` is the whitespace gate.

## 13. Completion Report

A Codex completion report must include:

```text
starting commit
ending commit
files changed
method semantics changed
method semantics deliberately unchanged
old code deleted
new or updated tests
pytest result
compileall result
preflight result
smoke result
git diff --check result
external API calls performed or not performed
remaining real-API risks
working-tree state
push state
```

Do not report only 鈥渢ests passed.鈥?Explain how the implementation maps back to
the research question.

## 14. Historical Results

Historical pilot and smoke directories are evidence, not current method code.

Do not:

- infer current behavior from old artifact names;
- restore old settings for compatibility;
- overwrite old results;
- compare unmatched commits as formal method results;
- interpret an operationally failed run as a method efficacy result.

When analyzing a run, first verify:

```text
git commit
method version
experiment protocol
dataset hashes
solver request identity
candidate funnel reached
validation selection completed
```

Raw `runs*`, SQLite databases, LLM logs, and checkpoints remain ignored unless
the user explicitly requests otherwise. Prefer compact, secret-free Markdown
and JSON reports for version control.

## 15. Git and Publication

Preserve user changes and inspect the worktree before staging. Never add ignored
raw runs, API secrets, caches, or unrelated files to a commit.

The established remote is:

```text
git@github.com:mmmmyyyxx/multi_agent_diversity.git
```

The user has historically requested direct updates to `main`. Do not create a
new branch or pull request unless the current task asks for one.
