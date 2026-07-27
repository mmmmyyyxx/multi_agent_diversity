# Method

## 1. Scope

The current method is **Member-Aware Peer-State Prompt-Team Optimization**:

```text
method_version = member_aware_peer_state_v4
```

It searches over a team of five prompts. The solver, optimizer, and evaluator
models remain frozen; no policy gradient or model-weight update occurs. Candidate
rollout outcomes are used for search; the final active team is evaluated only
after the fixed update budget completes.

The method addresses a failure of vote-only prompt optimization: a candidate can
improve plurality accuracy while weakening one or more team members. Such a team
may become dependent on a narrow coalition and can perform worse when the vote
distribution changes. The new method therefore treats team vote quality,
worst-member preservation, and total member improvement as joint aggregate
objectives. It does not impose zero-loss preservation on individual probe
examples.

## 2. Solver And Vote Contract

Five equal-weight agents answer every example. A valid response must end with
exactly one extractable:

```text
FINAL_ANSWER: <answer>
```

Invalid outputs are audited and do not silently become ordinary answers.
Aggregation is true plurality vote. A top-count tie abstains.

Solver invalid recovery is request-local and cache-resolved. A strict-parser
invalid response receives up to three additional identical Solver calls. The
first valid response is used immediately; four invalid responses produce one
terminal-invalid result. Transport retries remain separate. Formal competence
and guards count only terminal-invalid results, while first-pass and recovery
counts remain audit metrics.

Candidate acceptance uses an aggregate terminal-invalid condition: the
candidate target profile may not contain more terminal-invalid outputs than the
incumbent target profile. First-pass invalids that recover successfully remain
diagnostics and do not reject the candidate. The obsolete local/global invalid
allowances and accuracy-epsilon guard fields were removed in v4.

Target scheduling adds potential evidence without changing responsibility
assignment or max-wait semantics. Unimproved members, headroom to the current
best member, and historical positive target gains affect the Pareto priority.
Repeated Stage-B attempts with no positive target-gain candidate receive a
bounded cooldown; a positive but guard-rejected candidate clears that streak.

The optimized prompt contains only the mutable decision procedure. For every
Solver request, the program places that procedure first and appends the
immutable task output interface last:

```text
Follow the decision procedure below.

Decision procedure:
<mutable candidate prompt>

Mandatory output interface:
This interface is immutable and overrides any conflicting instruction above.
<strict task-specific FINAL_ANSWER contract>
```

The output interface is not part of the prompt search space. Student receives
the contract so it can avoid conflicts, but does not need to reproduce the
contract in each candidate. The strict parser and invalid-output guard remain
unchanged.

For each example:

```text
G = number of valid votes matching gold
H = largest valid wrong-answer vote count
M = G - H
```

The system also constructs each target agent's leave-one-out peer state. These
states diagnose whether changing that member can add coverage, repair the vote,
leave a dominant wrong cluster, or endanger a unique or pivotal correct vote.

## 3. Member Objectives

All formal objectives use integer counts on a fixed probe.

Let `c_i^0` be member `i`'s initial correct count and `c_i` its count under a
candidate team:

```text
g_i   = c_i - c_i^0
g_min = min_i g_i
g_sum = sum_i g_i
```

The team objective vector is:

```text
O = (V_count, g_min, g_sum)
```

`V_count` is the number of probe examples where plurality vote is correct.
Vector `a` Pareto-dominates `b` iff every component of `a` is at least the
corresponding component of `b`, and at least one is strictly greater.

Normalized accuracy and soft vote utility remain diagnostics. They are not
substituted for the integer Pareto objectives.

## 4. Member-Aware Responsibility

Every currently wrong member is eligible for optimization, even if it owns no
residual case. For member `i`, the global improvement pressure is:

```text
improvement_need_i = max(0, g_sum_current - K * g_i_current)
```

where `K=5`. A large value means the member is behind the current team-wide
improvement level.

On each vote-wrong example, ownership is selected from the wrong members by a
five-axis Pareto comparison of direct vote repair, oracle soft-utility gain,
improvement need, coverage opportunity, and dominant-wrong exit. The frontier
preference is member-first, then direct-fix, soft gain, coverage, dominant-wrong,
load, wait, and seeded tie-break. Existing owners are retained only while still
on the frontier, not behind on member/direct-fix priority, and within
`responsibility_switch_margin` on soft utility.

Target selection uses all agents with current errors. Define
`g_max = max_j g_j`, `gain_gap_to_best_i = g_max - g_i`, and a count tolerance
of five. Only agents with `gain_gap_to_best_i > 5` have relative improvement
potential; their rank is a discrete rank over distinct gain levels, with the
lowest gain assigned the highest positive rank. Current correct-count gaps remain competence reports,
not potential.

Agents waiting `responsibility_max_wait_updates` updates are considered first;
the default is eight. Otherwise, relative-potential agents are selected before
current team-repair evidence. If every agent is inside the tolerance band,
selection uses current repair value rather than forcing uniform member accuracy.
Historical candidate gains and failed-search streaks are recorded only as
candidate-search outcomes. They do not estimate future potential or enter the
target-selection Pareto vector; any resulting cooldown is a bounded
search-budget control.

Responsibility answers **who to update and what residual to repair**. Competence
preservation is deliberately not a sixth responsibility-attribution dimension.
Preservation-conditioned TCS evidence remains part of proposal generation, and
vote, unique-correct, and pivotal-correct changes remain symmetric diagnostics.
Formal protection is instead enforced at aggregate level by target improvement,
team-vote non-regression, member-objective Pareto dominance, and terminal-invalid
non-regression. Strong members therefore remain eligible for repair without a
per-example zero-loss constraint.

Responsibility has an explicit versioned lifecycle. The initial team is assigned
once. Rejected updates reuse that assignment. An accepted prompt/profile pair is
committed atomically, increments the team-state version, and refreshes
responsibility exactly once; the following update reuses that refreshed state.
`owner_age` advances once per real team-state refresh, never per function call.
If refresh fails, prompt/profile, accepted counts, responsibility state, caches,
versions, and appended responsibility audit rows are all rolled back.

## 5. Programmatic Aggregation And Lightweight TCS

The complete optimization probe is analyzed programmatically using vote
distributions, leave-one-out peer states, member correctness, and responsibility
signals. Structurally equivalent failures are aggregated into typed patterns,
and language-model roles receive at most three representative cases. These
cases are representative evidence only: all statistics and all rollout metrics
still use the complete fixed probe. Programmatic aggregation is not an agent.

Answers are encoded as `G`, `I`, and wrong clusters `W1` through `W3`. Wrong
clusters are ordered by decreasing size and then by a stable hash of the
normalized answer, never by agent identity. Pattern keys contain the failure
family, target and team status, answer role, team and peer `(G,H,M)`,
direct-fix/dominant-wrong flags, and unique/pivotal protection flags.

Pattern selection is deterministic and lexicographic. The member-aware slots
prioritize assigned residuals, target competence, and preservation. Generic
Peer-State slots prioritize coverage, conversion/dominant-wrong, and
preservation. Accuracy slots contain individual-error structure and
preservation only. Within a selected pattern, its single representative case is
chosen by assigned status, direct vote fix, larger oracle utility gain, older
owner age, smaller absolute margin, and stable question hash.

The three serialized context boundaries are:

- `AccuracyDiagnosisContext`: individual counts and sanitized
  individual/preservation patterns; no vote, peer, responsibility, or member
  need fields.
- `PeerStateDiagnosisContext`: team and leave-one-out Peer-State aggregates,
  without owners, responsibility, member gains, or improvement need.
- `MemberAwareDiagnosisContext`: Peer-State evidence plus member counts/gains,
  improvement need, and assigned responsibility.

`PreviousUpdateOutcome` replaces natural-language previous-update summaries.
The model-facing projection is also sanitized to the setting's causal boundary.
It exposes rollout deltas, rejection reasons, and acceptance only when at least
one candidate actually completed Stage A. A TCS transport, truncation, or schema
failure exposes only `attempted=true` and
`empirical_feedback_available=false`; operational failure must not masquerade
as empirical candidate rejection.

Teacher returns exactly:

```json
{"failure_pattern":"...", "repair_rule":"...", "preservation_rule":"..."}
```

The Teacher proposes one repair hypothesis; it does not calculate state,
predict performance, or generate prompts. Critic checks only the four hard
blocker classes `evidence_mismatch`, `actionable_specificity`,
`shortcut_or_copying`, and `preservation_or_output_risk`, returning:

```json
{"failed_checks":[], "risk_case_ids":[], "feedback":""}
```

Approval is computed by code from an empty `failed_checks` list. The Critic is
not a performance evaluator. Student sees only the parent prompt, approved plan,
output contract, requested count, and prompt-length limit, and returns:

```json
{"candidate_prompts":["complete replacement prompt"]}
```

Student does not diagnose or see cases. Candidate effectiveness is determined
exclusively by paired Stage A/B rollouts and member-aware Pareto selection.

Teacher, Critic, and Student outputs are not truncated by experiment-level completion-token budgets. Their search space is bounded structurally through strict schemas, at most three representative cases, bounded text fields, a fixed candidate count, and prompt-length constraints. Actual token usage is recorded for post-hoc analysis but does not terminate the experiment.

The Solver retains `solver_max_tokens=1800` to preserve its request identity
and shared cache. Providers may still return `finish_reason=length`; the
pipeline records this as a runtime failure, not as evidence of no method gain.

A semantic Teacher revision occurs only after a valid Critic rejection. The
revision request is an explicit, stateless request containing the complete
previous `TeacherRepairPlan`, structured `failed_checks`, `risk_case_ids`, and
Critic feedback, together with the same bounded diagnosis context. It asks for
all three replacement fields and requires cumulative satisfaction of every hard
check; it cannot repair one check by weakening a previously valid rule. The
revision protocol is `critic_grounded_full_plan_revision_v1` and is part of run
identity and TCS audit metadata.

Malformed or provider-truncated Teacher/Critic responses retry the identical
request once without consuming another semantic round. Student applies strict
schema, requested-count, per-prompt, total-prompt, parent-identity, duplicate,
and sample-memorization checks and never silently truncates extra candidates.
When a Student call produces no valid candidate, the next identical-cycle call
receives structured rejection classes and retries the same approved plan. The
initial call plus three retries form one four-call cycle. If that cycle is
exhausted, the program performs at most one fresh Teacher-Critic regeneration
from the same bounded diagnosis context and runs one final four-call Student
cycle. A response containing at least one valid candidate stops recovery
immediately and uses only those valid candidates. Thus the bound is two cycles
and eight Student calls; invalid candidates never enter Stage A. Provider
truncation is determined only from `finish_reason == "length"`.

## 6. Candidate Evaluation

Candidate evaluation replaces one target prompt while holding the other four
active prompts fixed. The fixed probe records:

- target correct and invalid counts
- vote gains and losses
- coverage and residual repairs
- unique and pivotal correct gains and losses
- candidate team vote-correct count
- all five candidate member correct counts
- gains relative to the initial prompt team

Prompt-question evaluation is cached by prompt, question, parser, model request,
temperature, seed, and output-contract identity. Stage A subsets and full Stage B
reuse the same cache entries.

## 7. Stage A

Member-aware settings shortlist candidates through three channels:

```text
team_vote
worst_member
mean_member
```

Each channel produces ordinal ranks. Rank vectors are divided into Pareto fronts,
then channel top-k union and deterministic Pareto ordering fill the Stage B
budget. `shared_peer_state_vote_first` is a **pure vote-first
candidate-selection ablation** using vote-first ordering in both stages; it is
not an exact reproduction of the historical Peer-State V2 shortlist and
acceptance pipeline.

The channel keys are:

- team-vote: vote-correct count, net vote delta, fewer vote losses, soft utility,
  assigned repair;
- worst-member: minimum gain, minimum-gain delta, improved-agent count, target
  gain versus incumbent, lower invalid count;
- mean-member: total gain, target gain versus incumbent, improved-agent count,
  assigned repair, lower invalid count.

## 8. Stage B

All optimized settings use the v4 aggregate feasibility contract. Relative to
the incumbent, a candidate must:

- strictly increase the target member's correct count;
- not reduce the aggregate team vote-correct count;
- Pareto-dominate the incumbent in `(V_count, g_min, g_sum)`;
- not increase the target member's terminal-invalid count.

These four conditions are the complete hard acceptance contract. A candidate
may lose correctness on particular probe examples when the aggregate contract
still holds. Vote gains/losses, unique-correct gains/losses, and pivotal-correct
gains/losses are retained for diagnosis, audit, and late deterministic
tie-breaking, but they do not independently reject a candidate. The old active
and initial competence floors, accuracy epsilons, vote-loss limit,
unique-correct loss limit, pivotal-correct loss limit, and local/global invalid
allowances are deleted rather than retained as dormant behavior.

The formal member-aware selector orders acceptable rows by:

```text
minimum member gain
vote-correct count
total member gain
improved-member count
fewer vote losses
soft vote utility
assigned repairs
target correct count
fewer invalids
earlier generation
prompt hash
```

Soft utility never converts a non-dominating candidate into an accepted one.

## 9. Final Active State And Test

The active final-state evaluation protocol is:

```text
initial team -> fixed planned updates on train -> final active team -> one test
```

The validation split remains in run identity but has zero Solver calls and no
role in acceptance, diagnosis, scheduling, early stopping, or checkpoint
selection. There is no best epoch, validation feasibility/key, rollback, or
historical checkpoint selection. The selected state is always the active team
after the planned update count, including when the final update is rejected.

Test is forbidden before training completes and is evaluated once for the final
active team. It cannot alter the active team or any training decision. A frozen
matched baseline may supply reporting-only differences. The optimized
`final_summary.json` records:

```text
selected_test
selection_summary
```

Test is never called before training completion and is never used to rank any
state. Summaries expose both integer correct-count
gain and test-size-normalized accuracy gain. Integer counts remain the formal
selection objective; normalized accuracy gains are cross-task reporting fields.

## 10. Settings

The repository exposes only:

```text
shared_baseline
shared_independent_accuracy
shared_peer_state_vote_first
shared_peer_state_member_pareto
shared_member_aware_responsibility
shared_member_aware_full
```

There are no aliases for removed methods or settings.

## 11. Persistence And Reproducibility

Checkpoint version is 12. It stores active and initial profiles, a target-free
`TeamMemberGainState`, member-aware opportunities, responsibility ownership and
ages, accepted counts, seeded ranks, team/responsibility state versions and
refresh count, target-priority audit, prompt state, TCS and Student-recovery
state, planned/completed update counts, final-state selection, training dynamics,
differentiation trajectories, selected-test state, caches, histories, LLM calls,
and Python random state. v11 and earlier checkpoints are incompatible.

Resume requires exact method, setting, config behavior fingerprint, code commit,
split files, question sets, probe identity, model endpoint identity, parser,
decoding, and output contract. Older checkpoints fail with:

```text
Checkpoint is incompatible with member_aware_peer_state_v4
```

The runner never silently restarts an incompatible run in the same directory.

## 12. Implementation Map

```text
multi_dataset_diverse_rl/member_objectives.py
multi_dataset_diverse_rl/peer_state.py
multi_dataset_diverse_rl/responsibility.py
multi_dataset_diverse_rl/diagnosis_aggregation.py
multi_dataset_diverse_rl/tcs.py
multi_dataset_diverse_rl/candidate_selection.py
multi_dataset_diverse_rl/evaluation/fixed_probe.py
multi_dataset_diverse_rl/evaluation/validation.py
multi_dataset_diverse_rl/system.py
multi_dataset_diverse_rl/persistence/checkpoint.py
multi_dataset_diverse_rl/persistence/identity.py
multi_dataset_diverse_rl/cli.py
scripts/run_task_level_accuracy.py
scripts/preflight_member_aware.py
scripts/deterministic_member_objective_unit_smoke.py
scripts/deterministic_member_aware_system_smoke.py
scripts/deterministic_member_aware_smoke.py
```

## 13. Boundaries

- Diversity is not a standalone reward.
- Soft vote utility is not a formal acceptance objective.
- TCS proposes changes but does not decide empirical success.
- Fixed-probe search can overfit; validation and multiple seeds remain necessary.
- A selected test improvement is an experimental result, not guaranteed by the
  optimization rule.
