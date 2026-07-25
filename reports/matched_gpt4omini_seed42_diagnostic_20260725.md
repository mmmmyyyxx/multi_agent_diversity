# GPT-4o-mini Matched Pilot Directed Diagnostic

## Scope And Provenance

- Source run: `runs_matched_gpt4omini_seed42_20260725`
- Source commit: `a51843fdab103b96d24c932ca46ba14c4ced09f0`
- Method: `member_aware_peer_state_v3`
- Task / seed: `disambiguation_qa` / `42`
- Models: `gpt-4o-mini` for Solver, Optimizer, and Evaluator roles
- Settings inspected: `shared_independent_accuracy` and
  `shared_member_aware_full`
- Diagnostic mode: local artifact analysis only; zero API calls

All update numbers below are zero-based. `Target gain` means target correct-count
gain versus the active incumbent unless explicitly labeled `vs initial`.
Objective vectors use `(V_count, g_min, g_sum)`.

For this report, a member is **successfully improved** when an accepted update
has positive target gain versus its incumbent. A **positive-but-rejected**
member has at least one non-accepted update with a positive target-gain
candidate. Waiting length is `first_success_update - first_attempt_update`;
members without a success are right-censored at update 7.

## 1. Full Update 4: Target Agent 2

The incumbent objective was `(29, 0, 29)`, with member gains
`[0, 13, 0, 16, 0]`. Both generated candidates reached Stage A and Stage B and
both improved the formal objective, but neither passed the hard guards.

| Candidate hash | Target gain vs incumbent / initial | Vote gain / loss / net | Unique gain / loss | Pivotal gain / loss | Objective | Hard feasible | Rejection reasons |
|---|---:|---:|---:|---:|---:|:---:|---|
| `3c8a2ba50b2c` | `+2 / +2` | `5 / 1 / +4` | `1 / 0` | `5 / 1` | `(33, 0, 31)` | no | `vote_loss`, `pivotal_correct` |
| `6306a237f25a` | `+3 / +3` | `4 / 1 / +3` | `2 / 0` | `4 / 1` | `(32, 0, 32)` | no | `vote_loss`, `pivotal_correct` |

Both candidates passed:

```text
local_accuracy
initial_accuracy
invalid
unique_correct
```

Both failed exactly:

```text
vote_loss: one incumbent-correct plurality case was lost
pivotal_correct: one incumbent pivotal-correct target case was lost
```

The objective vectors strictly dominate the incumbent vector, so the rejection
was not caused by a weak Pareto objective. Hard preservation constraints are
evaluated first and correctly blocked both candidates.

`unique_correct_loss_count` and `pivotal_correct_loss_count` are directly
audited fields. Gain counts are reconstructed from the uniquely matching local
incumbent peer-state snapshot and the formal leave-one-out definitions. The
reconstructed loss counts exactly equal the stored audit values (`0` unique,
`1` pivotal for each candidate).

## 2. Full Student Parsing: Updates 1, 3, 6, And 7

| Update | Target | JSON extracted | Schema valid | Raw / valid candidates | Per-candidate parse rejection | Student failure class | Downstream result |
|---:|---:|:---:|:---:|---:|---|---|---|
| 1 | 4 | yes | yes | `2 / 2` | `[[], []]` | none | Both candidates reached rollout; both failed `local_accuracy` and `initial_accuracy` |
| 3 | 0 | yes, both attempts | yes, both attempts | `2 / 0` per attempt | `[[empty_or_non_string], [empty_or_non_string]]` per attempt | `zero_valid_student_candidates` | No Stage A/B candidate |
| 6 | 4 | yes, both attempts | yes, both attempts | `2 / 0` per attempt | `[[empty_or_non_string], [empty_or_non_string]]` per attempt | `zero_valid_student_candidates` | No Stage A/B candidate |
| 7 | 3 | yes | yes | `2 / 1` | `[[parent_identical], []]` | none; partial-valid response | The valid candidate reached rollout and failed `local_accuracy` |

Exact classification:

- Update 1 was not a Student parsing failure. Both prompts were non-empty,
  non-parent, distinct, and within length limits.
- Updates 3 and 6 were not invalid JSON, schema errors, provider truncation,
  prompt-length failures, duplicates, or parent-identical outputs. The JSON
  object and `candidate_prompts` list were structurally valid, but both list
  elements were rejected as `empty_or_non_string`. The identical request was
  retried once; both attempts produced the same rejection class.
- Update 7 contained one exact parent-equivalent candidate and one valid
  non-parent candidate. There was no duplicate or length rejection. The single
  valid candidate subsequently lost one target correct answer relative to the
  incumbent and failed only the local competence floor.

## 3. Independent Accuracy: Four Accepted Updates

| Update | Target | Target gain vs incumbent / initial | Member gains after commit | Vote gain / loss / net | Objective transition |
|---:|---:|---:|---|---:|---|
| 1 | 1 | `+18 / +18` | `[0, 18, 0, 0, 0]` | `0 / 0 / 0` | `(29, 0, 0) -> (29, 0, 18)` |
| 2 | 2 | `+14 / +14` | `[0, 18, 14, 0, 0]` | `0 / 0 / 0` | `(29, 0, 18) -> (29, 0, 32)` |
| 3 | 3 | `+12 / +12` | `[0, 18, 14, 12, 0]` | `11 / 1 / +10` | `(29, 0, 32) -> (39, 0, 44)` |
| 7 | 2 | `+11 / +25` | `[0, 18, 25, 12, 0]` | `3 / 0 / +3` | `(39, 0, 44) -> (42, 0, 55)` |

The first two distinct-member improvements were vote-neutral. Update 3 was the
first improvement of a third distinct member, Agent 3, and the same commit
produced the first vote transition: 11 wrong-to-correct flips, one
correct-to-wrong flip, and a net gain of 10. This is a direct within-run
correspondence, not evidence of a general causal threshold.

## 4. Member Coverage And Outcome Classes

| Setting | Target sequence | Attempted distinct | Successfully improved distinct | Positive-but-rejected members | Zero-valid-candidate members |
|---|---|---:|---|---|---|
| Independent Accuracy | `[0, 1, 2, 3, 4, 0, 1, 2]` | `5` (`0,1,2,3,4`) | `3` (`1,2,3`) | none | `0,4` |
| Member-Aware Full | `[1, 4, 3, 0, 2, 1, 4, 3]` | `5` (`0,1,2,3,4`) | `2` (`1,3`) | `2` | `0,4` |

The zero-valid class is member-level aggregation over attempts:

- Independent Agent 0 had zero-valid Student outcomes at updates 0 and 5;
  Agent 4 had one at update 4.
- Full Agent 0 had one at update 3; Agent 4 had one at update 6.
- All of these were `empty_or_non_string` candidate-element failures rather
  than JSON/schema/length failures.

Independent had no positive-but-rejected member. Its non-accepted evaluated
update, Agent 1 at update 6, had only negative target-gain candidates (`-8`
and `-2`). Full Agent 2 is positive-but-rejected because update 4 produced
target gains `+2` and `+3`, both blocked by vote and pivotal-loss guards.

## 5. First Attempt To First Success

`0` means success on the first attempt. A value such as `>7 (right-censored)`
means no success occurred by update 7 and seven update-index intervals elapsed
after the first attempt.

| Setting | Agent | First attempt | First success | Waiting length |
|---|---:|---:|---:|---|
| Independent Accuracy | 0 | 0 | none | `>7` (right-censored) |
| Independent Accuracy | 1 | 1 | 1 | `0` |
| Independent Accuracy | 2 | 2 | 2 | `0` |
| Independent Accuracy | 3 | 3 | 3 | `0` |
| Independent Accuracy | 4 | 4 | none | `>3` (right-censored) |
| Member-Aware Full | 0 | 3 | none | `>4` (right-censored) |
| Member-Aware Full | 1 | 0 | 0 | `0` |
| Member-Aware Full | 2 | 4 | none | `>3` (right-censored) |
| Member-Aware Full | 3 | 2 | 2 | `0` |
| Member-Aware Full | 4 | 1 | none | `>6` (right-censored) |

## 6. Diagnostic Conclusion

Both target policies attempted all five members, so the observed Full outcome
is not explained by target starvation within this eight-update horizon. The
member-level bottlenecks were downstream:

1. Agents 0 and 4 were reached but lost attempts to zero-valid Student
   candidate elements.
2. Agent 2 produced positive candidates, but both violated the frozen
   zero-vote-loss and zero-pivotal-loss guards.
3. Agent 3 was improved successfully on its first attempt; its later update 7
   produced one parent-identical candidate and one locally regressing
   candidate.

Independent Accuracy improved three distinct members. Its first plurality gain
appeared exactly when the third distinct member improved, while Full improved
only two distinct members and did not reach that within-run configuration.
This diagnosis identifies the realized proposal/feasibility funnel for this
single task and seed; it does not establish statistical efficacy or justify
changing the method guards.

## Publication Boundary

The report contains only aggregate counts, enumerated failure classes, update
indices, and prompt hashes. It excludes raw API responses, response excerpts,
prompt text, Teacher/Critic/Student text, questions, gold answers, member
answers, SQLite contents, credentials, endpoint values, and local absolute
paths.
