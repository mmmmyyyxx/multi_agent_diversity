# Method

## 1. Scope

The current method is **Member-Aware Peer-State Prompt-Team Optimization**:

```text
method_version = member_aware_peer_state_v1
```

It searches over a team of five prompts. The solver, optimizer, and evaluator
models remain frozen; no policy gradient or model-weight update occurs. Candidate
rollout outcomes are used for search and validation selection.

The method addresses a failure of vote-only prompt optimization: a candidate can
improve plurality accuracy while weakening one or more team members. Such a team
may become dependent on a narrow coalition and can perform worse when the vote
distribution changes. The new method therefore treats team vote quality,
worst-member preservation, and total member improvement as joint objectives.

## 2. Solver And Vote Contract

Five equal-weight agents answer every example. A valid response must end with
exactly one extractable:

```text
FINAL_ANSWER: <answer>
```

Invalid outputs are audited and do not silently become ordinary answers.
Aggregation is true plurality vote. A top-count tie abstains.

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

Target selection uses all agents with current errors. Agents waiting
`responsibility_max_wait_updates` updates are considered first; the default is
four. The remaining target comparison is member-aware and deterministic.

After an accepted update, active profiles, peer states, opportunities, owners,
loads, target priorities, and TCS summaries are refreshed immediately.

## 5. Responsibility-Conditioned TCS

TCS retains three typed context boundaries:

- `AccuracyProposalContext`: individual errors and competence preservation only.
- `PeerStateProposalContext`: peer/team state without ownership metadata.
- `MemberAwareResponsibilityProposalContext`: member count/gain state,
  improvement need, assigned residuals, member errors, preservation evidence,
  and the previous member update summary.

The member-error evidence limit is `tcs_member_error_limit`, defaulting to six.
Contexts are deterministically truncated and audited.

Teacher always returns the same six semantic fields:

```text
observed_failure_pattern
generalizable_mechanism
decision_rule
uncertainty_or_abstention_rule
preservation_conditions
evidence_summary
```

Critic applies the existing hard gate: context consistency, no sample
memorization, executable and internally consistent change, explicit preservation,
output-contract safety, no peer copying, no stereotype forcing, and a non-generic
change. Critic score is diagnostic only. Rejected proposals return feedback to a
new Teacher round. Student runs only after approval and must produce strict JSON.

## 6. Candidate Evaluation

Candidate evaluation replaces one target prompt while holding the other four
active prompts fixed. The fixed probe records:

- target correct and invalid counts
- vote gains and losses
- coverage and residual repairs
- unique and pivotal correct losses
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
budget. The vote-first ablation uses only its vote-first Stage A ordering.

The channel keys are:

- team-vote: vote-correct count, net vote delta, fewer vote losses, soft utility,
  assigned repair;
- worst-member: minimum gain, minimum-gain delta, improved-agent count, target
  gain versus incumbent, lower invalid count;
- mean-member: total gain, target gain versus incumbent, improved-agent count,
  assigned repair, lower invalid count.

## 8. Stage B

Before formal selection, a candidate must pass:

- active target-member correct-count floor
- initial target-member correct-count floor
- invalid-count guard
- vote-loss limit
- unique-correct loss limit
- pivotal-correct loss limit

For `member_aware_pareto`, feasibility is necessary but not sufficient. The
candidate objective must Pareto-dominate the incumbent objective. Candidate
preference among acceptable rows is:

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

## 9. Validation And Final Test

Validation compares every epoch with the initial validation team. A state is
feasible only when:

- no member falls below its initial count beyond the configured epsilon
- invalid rate does not exceed the initial rate beyond its guard
- vote-correct count is not below the initial count

Feasible states are ordered by:

```text
minimum member gain
vote-correct count
total member gain
improved-member count
soft vote utility
fewer C0 examples
lower invalid rate
earlier epoch
```

After validation selects prompts, test evaluation runs both the initial and
selected prompt teams. `final_summary.json` contains:

```text
initial_test
selected_test
member_gain
selection_summary
```

This makes test improvement and member regression directly auditable.

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

Checkpoint version is 5. It stores active and initial profiles, member-aware
opportunities, responsibility ownership and ages, accepted counts, seeded ranks,
target-priority audit, prompt state, TCS state, caches, histories, LLM calls, and
Python random state.

Resume requires exact method, setting, config behavior fingerprint, code commit,
split files, question sets, probe identity, model endpoint identity, parser,
decoding, and output contract. Older checkpoints fail with:

```text
Checkpoint is incompatible with member_aware_peer_state_v1
```

The runner never silently restarts an incompatible run in the same directory.

## 12. Implementation Map

```text
multi_dataset_diverse_rl/member_objectives.py
multi_dataset_diverse_rl/peer_state.py
multi_dataset_diverse_rl/responsibility.py
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
scripts/deterministic_member_aware_smoke.py
```

## 13. Boundaries

- Diversity is not a standalone reward.
- Soft vote utility is not a formal acceptance objective.
- TCS proposes changes but does not decide empirical success.
- Fixed-probe search can overfit; validation and multiple seeds remain necessary.
- A selected test improvement is an experimental result, not guaranteed by the
  optimization rule.
