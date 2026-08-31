# GEPA Critic Gate Failure Audit

## Scope

This is a zero-API, read-only audit of the two blocked parents from the GEPA
proposal-breadth pilot. It does not change Teacher, Critic, Student, Common-Safe,
ranking, target allocation, or the N=2/N=4 design. It does not access validation
or test and does not reinterpret a pre-Student failure as a breadth result.

## Main diagnosis

```text
PRE_STUDENT_CRITIC_GATE_BOTTLENECK_CONFIRMED
```

Both frozen parents completed two schema-valid Teacher/Critic rounds. There were
no JSON/schema errors, parser errors, provider truncations, or infrastructure
failures. Each Critic response was a semantic rejection, both branches exhausted
the two-round budget, and Student was never called.

Therefore:

```text
candidate selection is not primary
proposal breadth was not evaluated
the immediate operational bottleneck is the Critic semantic gate
```

The requested candidate count is first consumed when building the Student
request, after Critic approval. N=4 could not act before these failures.

## Historical comparison

The frozen V18 evidence contains 96 proposal branches under the same v15
Teacher-Critic-Student semantics:

| Outcome | Branches | Rate |
|---|---:|---:|
| Student reached | 27 | 28.125% |
| Critic semantic gate exhausted | 69 | 71.875% |

Across 148 semantic-rejection responses, historical failed-check counts were:

| Failed check | Count |
|---|---:|
| `preservation_or_output_risk` | 73 |
| `actionable_specificity` | 68 |
| `evidence_mismatch` | 8 |

All 96 historical branches used three selected evidence cases. Mean serialized
context size was approximately 5,004 characters for Student-reached branches and
4,952 for exhausted branches. Larger context therefore does not explain the
observed semantic exhaustion.

These aggregate failed-check frequencies describe the historical comparison
population only. They are not assigned to the two blocked probe cases because
the breadth runner did not persist their Critic `failed_checks`, feedback, plan
hashes, or per-round audit rows.

## Same-state witnesses and comparability limit

The historical branch corresponding to each frozen parent/target previously
reached Student after one approved Critic response. The parent-target pairs are
therefore not intrinsically incapable of candidate generation.

- Seed59/update3 reproduced the exact historical proposal-context hash, yet the
  new Teacher/Critic path exhausted. This localizes the difference to the
  unpersisted plan/decision path rather than parent context structure.
- Seed61/update5 did not reproduce the exact historical context hash. Its frozen
  reconstruction omitted a prior same-target semantic-rejection outcome; the
  reconstructed context differs by one serialized character. This is a fixed-
  parent representation limitation and prevents an exact same-input comparison
  for that case.

The relevant v15 TCS source did not change between the V18 execution source and
the breadth execution source; only unrelated compatibility-repair code changed
inside the method package. Nevertheless, exact over-strictness cannot be inferred
without the blocked Critic decisions.

## Answers to the three audit questions

1. **Why did the two parents fail repeatedly?** They exhausted two rounds of
   schema-valid Critic semantic rejection before Student. The exact rejected
   criteria are not recoverable from the frozen probe artifacts.
2. **Format/parser failure or semantic rejection?** Semantic rejection. Format,
   schema, truncation, and infrastructure explanations are excluded.
3. **What differs from normal historical branches?** Both exact historical
   parent/target witnesses reached Student. Seed59 had the same context hash;
   Seed61 had a prior-outcome reconstruction mismatch. Across all V18 branches,
   semantic exhaustion was already common and was dominated by preservation/
   output-risk and actionable-specificity decisions.

## Decision

This audit does **not** justify removing or relaxing Critic, and it does not
support or reject proposal breadth. A future original-gate versus minimally
changed-gate experiment requires a new preregistration and must first persist
complete per-round, analysis-safe Critic decision telemetry. The current raw
experiments must not be rerun or rewritten.
