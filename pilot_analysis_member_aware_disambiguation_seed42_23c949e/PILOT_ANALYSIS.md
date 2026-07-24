# Member-Aware Disambiguation Viability Pilot Analysis

## Run identity

- Code commit: `23c949e23f03c569395e55c3b98637aee1ed1e22`
- Task: `disambiguation_qa` (BBH)
- Seed: `42`
- Settings: `shared_baseline`, `shared_member_aware_full`
- Split sizes: optimization `75`, validation `50`, test `125`
- Split overlap: `0` for optimization/validation, optimization/test, and
  validation/test
- Raw artifact directory:
  `runs_member_aware_disambiguation_viability_seed42_23c949e`

The raw run directory remains ignored. This report records the compact,
reviewable evidence needed for source control.

## Result

The pilot is **operationally inconclusive**, not a negative efficacy result.
The member-aware run never produced a candidate because every Critic response
was truncated at its configured completion-token limit.

| Metric | Baseline | Member-aware full |
|---|---:|---:|
| Initial test vote accuracy | 0.488 (61/125) | 0.488 (61/125) |
| Selected test vote accuracy | 0.488 (61/125) | 0.488 (61/125) |
| Vote gain | 0.000 | 0.000 |
| Minimum member accuracy gain | 0.000 | 0.000 |
| Mean member accuracy gain | 0.000 | 0.000 |
| Improved/regressed members | 0 / 0 | 0 / 0 |
| Selected epoch | 0 | 0 |
| Selection changed | false | false |

Validation at epoch 1 remained identical to the initial team:
`32/50 = 0.64` for the vote and for every member. The epoch was feasible, but
its key lost to epoch 0 on the earlier-epoch tie-break because no prompt changed.

## Search funnel diagnosis

The configured epoch produced eight real update attempts.

- Target sequence: `0, 4, 2, 1, 3, 0, 4, 2`
- Selection counts: agent 0 = 2, agent 1 = 1, agent 2 = 2, agent 3 = 1,
  agent 4 = 2
- All five eligible members were selected within the first five attempts.
- Max-wait fairness activated from update 4 onward.
- Initial responsibility state was computed once at team-state version 0.
- The 45 vote-wrong optimization examples were distributed evenly: nine owners
  per member.
- With no accepted update, no second responsibility refresh occurred.

This is positive evidence for target eligibility, starvation prevention,
balanced ownership, and the repaired responsibility lifecycle.

The proposal funnel stopped before empirical candidate evaluation:

| Stage | Count |
|---|---:|
| Update attempts | 8 |
| Teacher rounds | 24 |
| Teacher schema-valid | 24 |
| Critic calls | 72 |
| Critic schema-valid | 0 |
| Student calls | 0 |
| Raw candidates | 0 |
| Stage A evaluations | 0 |
| Stage B evaluations | 0 |
| Accepted updates | 0 |

All 72 Critic parse failures were reported as `critic response is not JSON`.
The underlying responses begin as JSON (sometimes fenced), but end before the
object closes. Every evaluator call used exactly `1800` completion tokens:

- calls at the configured ceiling: `72/72`
- minimum completion tokens: `1800`
- maximum completion tokens: `1800`
- mean prompt tokens: `9746.1`
- selected proposal-context size: approximately `17.9k` characters /
  `4.47k` estimated tokens

The Critic had to restate 18 selected case facts per attempt (six assigned
coverage, six member errors, and six representative cases). The resulting JSON
did not fit in the current `critic_max_tokens=1800` budget. Transport-level
calls all succeeded, so ordinary API retry logic could not detect this semantic
failure.

## Cost

| Cost counter | Baseline | Member-aware full |
|---|---:|---:|
| Solver calls | 125 | 125 |
| Optimizer calls | 0 | 24 |
| Evaluator calls | 0 | 72 |
| Total calls | 125 | 221 |
| Total tokens | 51,040 | 1,018,481 |
| Failed transport attempts | 0 | 0 |

Evaluator calls consumed `831,321` tokens, about 81.6% of the member-aware
run. The shared prompt-question cache avoided another 125 solver calls when the
unchanged final team was tested.

## Interpretation

The pilot establishes that:

- exact run identity and split isolation worked;
- the solver path produced no invalid outputs;
- target selection covered every member;
- max-wait fairness activated as designed;
- responsibility was not redundantly refreshed;
- shared solver caching worked.

It does **not** evaluate:

- Student proposal quality;
- the three Stage A channels;
- Stage B competence guards or Pareto acceptance;
- whether the method improves vote or member accuracy.

Zero gain must therefore not be reported as evidence against the method.

## Required follow-up before rerunning

1. Make Critic completion truncation observable (record provider finish reason
   and distinguish token-limit truncation from generic invalid JSON).
2. Ensure the required strict Critic payload fits its output budget. Candidate
   options are a larger Critic completion budget, a more compact fact
   restatement schema, or smaller context case limits.
3. Fail fast when all Critic attempts hit the completion ceiling instead of
   spending all Teacher/Critic rounds on predictably truncated responses.
4. Re-run the same seed and split using the shared solver cache, then require at
   least one schema-valid Critic response and one Student invocation before
   treating the run as a viability result.
5. Do not expand to a formal multi-task pilot until the full candidate funnel
   reaches Stage A and Stage B.

## Implementation follow-up

The subsequent `member_aware_peer_state_v2` behavioral protocol addresses this
pilot finding without changing the formal vote, responsibility, Stage A, or
Stage B objectives. The full fixed probe is now analyzed programmatically and
compressed into at most three typed patterns and three representative cases.
Teacher emits a three-field repair plan, Critic checks only four semantic hard
blocker classes, and Student sees only the approved plan and realizes complete
replacement prompts. Role-specific completion budgets are 600/300/1400 tokens.

Provider finish reason, completion limit, and truncation classification are now
recorded. Format retries do not consume semantic revision rounds, and two
same-role truncations stop the update. These changes are covered only by
offline deterministic tests in this implementation task; the real-API
viability risk remains unverified until a separate explicitly authorized pilot.
