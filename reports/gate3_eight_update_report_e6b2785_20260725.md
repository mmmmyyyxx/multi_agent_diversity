# Gate 3 Eight-Update Analysis Report

## Scope

- Code commit: `e6b2785008db0f602ddfd6b0affc1ea6861a0bde`
- Working tree at run start: clean
- Method: `member_aware_peer_state_v3`
- TCS protocol: `aggregated_small_model_tcs_v3`
- Teacher revision protocol: `critic_grounded_full_plan_revision_v1`
- Setting: `shared_member_aware_full`
- Task: `disambiguation_qa` (BBH/MARS)
- Seed: `42`
- Split sizes: `75 / 50 / 125`
- Configuration: `epochs=8`, `update_every=75`,
  `solver_max_tokens=1800`, `eval_solver_call_concurrency=8`
- Local source artifact:
  `runs_gate3_eight_update_v3_e6b2785_20260725_153657`

This run used a fresh output directory and fresh Solver cache. It is a
single-setting, single-seed viability run, not a matched efficacy result.

## Decisions

### Method and research gate

**Gate 3: STRONG PASS.**

- Eight updates reached explicit terminal states.
- Six of eight updates reached both Stage A and Stage B.
- Six updates were accepted.
- All five members gained on optimization, validation, and test.
- Validation selected epoch 8.
- Validation vote improved from 26 to 34.
- Test vote improved from 52 to 71.
- No member regressed and selected validation/test outputs had zero terminal
  invalids.

### Operational scaling gate

**HOLD before a larger matched efficacy pilot.**

The run completed without a terminal `429`, but Solver recorded 522 rate-limit
retry attempts over 2,075 unique resolved requests (25.2%). This exceeds the
previously defined 20% stop-expansion threshold even after concurrency was
reduced from 20 to 8. Lower concurrency should be calibrated before a larger
matched run, and the chosen concurrency must then be frozen across settings.

## Eight-Update Funnel

| Update | Target | Critic result | Student | Stage A/B | Accepted | Terminal state |
|---:|---:|---|---:|---:|:---:|---|
| 0 | 0 | rejected twice | 0 | 0 / 0 | no | `critic_semantic_rejection_exhausted` |
| 1 | 4 | rejected twice | 0 | 0 / 0 | no | `critic_semantic_rejection_exhausted` |
| 2 | 2 | revision approved | 1 | 2 / 2 | yes | completed |
| 3 | 3 | revision approved | 1 | 2 / 2 | yes | completed |
| 4 | 1 | first-round approved | 1 | 2 / 2 | yes | completed |
| 5 | 0 | first-round approved | 1 | 2 / 2 | yes | completed |
| 6 | 4 | first-round approved | 1 | 2 / 2 | yes | completed |
| 7 | 2 | first-round approved | 1 | 2 / 2 | yes | completed |

Totals:

```text
Teacher calls:             12
Critic calls:              13
Critic semantic rejections: 6
Critic approvals:           6
Student calls:              6
Valid candidates:          12
Stage A evaluations:       12
Stage B evaluations:       12
Hard-feasible candidates:   9
Pareto-acceptable:           9
Accepted updates:            6
```

Three candidates were rejected by both the zero-vote-loss and
zero-pivotal-loss guards. No candidate was rejected by local competence,
initial competence, invalid-output, or unique-correct guards.

The two pre-Student terminal failures equal, but do not exceed, the allowed
`2/8` TCS stability threshold.

## Live Teacher Revision

Four updates exercised the stateless revision branch:

| Update | Previous-plan binding | Critic binding | Plan changed | Round-2 result |
|---:|:---:|:---:|:---:|---|
| 0 | yes | yes | all three fields | rejected again |
| 1 | yes | yes | no | rejected again |
| 2 | yes | yes | all three fields | approved |
| 3 | yes | yes | all three fields | approved |

Updates 2 and 3 are the first live-provider validation that:

```text
Critic round-1 rejection
  -> previous_plan_hash and revision_critic_hash bind correctly
  -> Teacher emits a changed full replacement plan
  -> Critic round 2 re-evaluates and approves
  -> Student runs
```

Update 1 returned an unchanged plan, which the audit detected through an equal
plan hash and empty `revision_changed_fields`; Critic correctly rejected it
again and Student was not called. This is model non-compliance, not a Critic
bypass, but it remains a TCS stability signal.

Update 2 also produced one over-length Critic feedback payload. The identical
request format retry succeeded; the semantic round count was not consumed by
the schema error.

## Scheduler And Responsibility

Target sequence:

```text
[0, 4, 2, 3, 1, 0, 4, 2]
```

- All five erroneous agents were targeted by update 4.
- Every selected target belonged to the actual selection pool and Pareto
  frontier.
- Updates 4 through 7 used the overdue pool and selected only its member.
- Updates 0 and 1 had no Stage B evaluation and did not update potential state.
- Updates 2 through 7 found positive target-gain candidates and updated
  potential state without assigning cooldown.
- Max-wait remained attempt-aware: failed targets 0 and 4 were still considered
  previously attempted, then became overdue again in the expected order.

Six accepted transitions produced seven responsibility snapshots with
contiguous team-state versions `0..6`, exactly:

```text
initial snapshot + accepted update count = 1 + 6 = 7
```

No checkpoint artifact was emitted by this runner, so checkpoint
serialize/resume behavior was not independently exercised in this run. Cache
identity was fresh and internally consistent: 0 hits, 2,075 misses, and 2,075
ready entries.

## Optimization Trajectory

Accepted objective transitions use `(vote_correct, minimum_gain, total_gain)`:

| Update | Target | Incumbent | Accepted | Vote gains/losses | Target gain |
|---:|---:|---|---|---|---:|
| 2 | 2 | `(25, 0, 0)` | `(25, 0, 17)` | `0 / 0` | +17 |
| 3 | 3 | `(25, 0, 17)` | `(25, 0, 20)` | `0 / 0` | +3 |
| 4 | 1 | `(25, 0, 20)` | `(30, 0, 56)` | `5 / 0` | +36 |
| 5 | 0 | `(30, 0, 56)` | `(32, 0, 63)` | `2 / 0` | +7 |
| 6 | 4 | `(32, 0, 63)` | `(39, 3, 85)` | `7 / 0` | +22 |
| 7 | 2 | `(39, 3, 85)` | `(44, 3, 92)` | `5 / 0` | +7 |

The third accepted distinct-member improvement, target 1 at update 4, produced
the first vote gain. This supports the intended mechanism: member competence
improvements began converting into plurality gains after the team crossed the
multi-member threshold.

Final optimization-probe member gains were:

```text
[7, 36, 24, 3, 22]
```

All five members improved, `g_min=3`, `g_sum=92`, and vote correct increased
from 25 to 44.

## Validation

| Epoch | Feasible | Vote correct | Member correct counts | Gains |
|---:|:---:|---:|---|---|
| 1 | yes | 26 | `[26, 26, 26, 26, 26]` | `[0, 0, 0, 0, 0]` |
| 2 | yes | 26 | `[26, 26, 26, 26, 26]` | `[0, 0, 0, 0, 0]` |
| 3 | yes | 26 | `[26, 26, 30, 26, 26]` | `[0, 0, 4, 0, 0]` |
| 4 | yes | 26 | `[26, 26, 30, 30, 26]` | `[0, 0, 4, 4, 0]` |
| 5 | yes | 31 | `[26, 38, 30, 30, 26]` | `[0, 12, 4, 4, 0]` |
| 6 | yes | 30 | `[29, 38, 30, 30, 26]` | `[3, 12, 4, 4, 0]` |
| 7 | yes | 32 | `[29, 38, 30, 30, 34]` | `[3, 12, 4, 4, 8]` |
| 8 | yes | 34 | `[29, 38, 33, 30, 34]` | `[3, 12, 7, 4, 8]` |

Epoch 8 was selected. It is a non-initial feasible checkpoint with all members
improved, no competence regression, and an eight-question vote gain over the
initial validation team.

## Test

| Metric | Initial | Validation-selected | Delta |
|---|---:|---:|---:|
| Vote correct | 52 / 125 | 71 / 125 | +19 |
| Vote accuracy | 0.416 | 0.568 | +0.152 |
| Member correct counts | `[52, 52, 52, 52, 52]` | `[57, 85, 78, 57, 79]` | `[+5, +33, +26, +5, +27]` |
| Minimum member gain | 0 | 5 | +5 |
| Total member gain | 0 | 96 | +96 |
| Improved members | 0 | 5 | +5 |
| Regressed members | 0 | 0 | 0 |
| Terminal invalid | 0 | 0 | 0 |

This is a strong generalization signal, but it remains descriptive: only one
Full setting and one seed were run.

## Solver Invalid Recovery

| Metric | Result |
|---|---:|
| Unique resolved requests | 2,075 |
| First-pass valid | 2,047 |
| First-pass invalid | 28 |
| Recovered invalid | 27 |
| Terminal invalid | 1 |
| Eventual valid rate | 99.9518% |
| Recovery API calls | 36 |
| Recovery call overhead | 1.73% |
| Recovery token overhead | 2.66% |

This passes the Gate 3 recovery criteria (`eventual >= 99.5%`,
`terminal_invalid <= 1`). The lone terminal invalid belonged to a non-selected
candidate at update 3. That candidate remained within the configured allowance
but was not accepted; selected validation and test checkpoints had zero
terminal invalids.

## Rate Limits

| Metric | Result |
|---|---:|
| Solver `429` retry attempts | 522 |
| All-role `429` retry attempts | 530 |
| Solver calls with at least one retry | 375 / 2,111 |
| Maximum Solver retries before success | 4 |
| Terminal `429` failures | 0 |
| Solver `429` latency | 3,258.7 seconds |
| All-role `429` latency | 3,306.8 seconds |

The run satisfies the hard requirement of zero terminal `429` failures, but
the retry-attempt rate `522 / 2,075 = 25.2%` exceeds the 20% stop-expansion
threshold. Before matched efficacy work, run a small concurrency calibration
(for example 4 and, if necessary, 2), then freeze the first configuration that
keeps the retry-attempt rate at or below 10% without terminal failures.

## Final Conclusion

Gate 3 strongly validates the intended research mechanism:

```text
member-aware scheduling
  -> all members receive useful updates
  -> minimum and total member gains become positive
  -> plurality vote gains emerge
  -> gains generalize to validation and test
```

It also validates the real-provider Teacher revision branch in two successful
updates and confirms strict rejection when revision remains inadequate.

The method is ready for matched efficacy testing conceptually. Operationally,
do not begin the larger matched pilot at concurrency 8: first resolve the
25.2% rate-limit retry pressure with a small, method-neutral concurrency
calibration.

## Publication Boundary

The public directory contains compact, secret-free metrics and audit hashes.
Raw API responses, SQLite cache, complete LLM logs, and prompt bodies remain
local.
