# V17 W1 Target-Allocation Mechanism Audit

## 1. Objective

This is a zero-API, historical-artifact-only diagnosis of why W1 allocated the
fixed proposal budget to worse local optimization opportunities than
Round-Robin (RR) in the completed V17 Module1 2×2 fixed-parent isolation.
It does not change W1, Common-Safe, ranking, max-one, proposal generation, or
any trajectory. It performs no new validation or test evaluation.

The answer is not a single failure mode. Under these six selected frozen
parents, the frozen diagnostic classifier is:

```text
MULTIPLE_TARGET_ALLOCATION_FAILURES
```

The supported components are target coverage/exploration loss, target-value
estimation mismatch, and branch realizability mismatch. Repeated-target
diminishing returns is not classified as a primary mechanism.

## 2. Frozen Prior Evidence

The input is the already completed 2×2 experiment executed from source commit
`66a0276dc61e77fe71e8add94eb4865d1235b7b5`. Its frozen matrix is:

| Cell | Allocation | Search context |
| --- | --- | --- |
| A | RR | Generic |
| B | W1 | Generic |
| C | RR | Member-aware |
| D | W1 | Member-aware |

The prior gate passed with 6 parents, 24 cells, 48 branches, 9 hypothetical
updates, zero actual commits, zero trajectory mutations, and zero test calls.
Its allocation contrasts were `B-A = (-1 vote, -2 oracle)` and
`D-C = (-2 vote, +5 oracle)`. The prior local conclusion was
`TARGET_ALLOCATION_DOMINANT`.

## 3. Audit Scope

The audit reconstructs:

- each frozen parent's RR and W1 target choices;
- all W1 score components and total ranks for every observed branch target;
- target and accepted-update history preceding the frozen S2 parent;
- source/revision validity, Critic exhaustion, Common-Safe rejection counters,
  feasible counts, branch/cell winners, and hypothetical commit status;
- train deltas for branch winners and validation deltas for cell winners.

No model, Solver, evaluator, optimizer, validation-selection, or test endpoint
was called. Existing cache rows were opened read-only and their file hash was
verified unchanged.

## 4. Data Reconstruction Gate

| Check | Result |
| --- | ---: |
| Parents | 6/6 |
| Cells | 24/24 |
| Branches | 48/48 |
| W1 score reconstruction mismatches | 0 |
| Branch target identity mismatches | 0 |
| Validation metric mismatches | 0 |
| Cache hash changed | no |
| ZERO_API | true |
| NEW_TEST_CALLS | 0 |
| Gate | PASS |

The W1 score was reconstructed as

```text
B = 0.5 Dhat + 0.3 Shat + 0.2 dhat
A = (B + 0.05 * normalized_wait) / (1 + failure_count)
```

and matched all persisted scores. Candidate-level analysis cannot be fully
reconstructed because the probe did not persist every candidate hash and every
per-candidate Common-Safe decision. It is therefore explicitly marked
`NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS`; no `candidate_level.csv` is
fabricated.

## 5. Parent-Level Allocation Comparison

The six parents contain two concentration witnesses, two throughput witnesses,
and two neutral controls. RR and W1 shared one selected target at five parents
and shared none at one parent. Aggregate realized validation outcomes were:

| Cell | WOULD_COMMIT | Vote delta | Oracle delta |
| --- | ---: | ---: | ---: |
| A: RR + Generic | 3 | +1 | +2 |
| B: W1 + Generic | 1 | 0 | 0 |
| C: RR + Member-aware | 3 | +2 | -3 |
| D: W1 + Member-aware | 2 | 0 | +2 |

Thus W1 produced no positive aggregate validation-vote change in either
context. The `D-C` oracle improvement prevents the result from being described
as uniform W1 harm: W1's negative local effect is clearer for vote conversion
than for oracle coverage.

## 6. Branch Funnel Analysis

Counts in parentheses are candidate counts; the main funnel uses branch counts.

| Group | Branches | Valid-source branches (sources) | Feasible branches (candidates) | Branch winners | Cell winners | Positive val vote | Positive val oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 12 | 6 (11) | 4 (9) | 4 | 3 | 1 | 1 |
| B | 12 | 3 (6) | 1 (3) | 1 | 1 | 0 | 0 |
| C | 12 | 4 (7) | 3 (3) | 3 | 3 | 1 | 1 |
| D | 12 | 6 (11) | 2 (5) | 2 | 2 | 0 | 2 |
| RR | 24 | 10 (18) | 7 (12) | 7 | 6 | 2 | 2 |
| W1 | 24 | 9 (17) | 3 (8) | 3 | 3 | 0 | 2 |

The loss starts at different places by context:

- Generic (`B-A`): W1 falls behind at generation/semantic gating—valid-source
  branches fall from 6 to 3 and Critic-exhausted branches rise from 6 to 9.
- Member-aware (`D-C`): W1 has more valid-source branches (6 versus 4), but
  fewer branches reach feasibility (2 versus 3). The downstream Common-Safe
  counters are also less favorable: D records 10 target regressions, 2 vote
  regressions, and 9 no-progress decisions, versus C's 6, 0, and 4.
- Overall: source-candidate volume is almost equal (18 RR versus 17 W1), yet
  feasible candidates fall from 12 to 8, feasible branches from 7 to 3, and
  cell winners from 6 to 3.

This is inconsistent with a single explanation based only on source validity.

## 7. Selection Opportunity Loss

The matched allocation split gives the clearest missed-opportunity evidence:

| Relation | Branches | Valid-source branches | Feasible branches | Cell winners | Positive val vote | Positive val oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RR-only | 14 | 4 | 4 | 3 | 2 | 1 |
| W1-only | 14 | 7 | 3 | 3 | 0 | 2 |
| Overlap | 20 | 8 | 3 | 3 | 0 | 1 |

Every RR-only branch that produced a valid source also produced a feasible
candidate (4/4), and RR-only targets supplied both positive validation-vote
returns observed in the audit. W1-only targets generated more valid-source
branches (7) but converted only 3 to feasibility and none to positive
validation-vote return. This supports
`TARGET_COVERAGE_EXPLORATION_FAILURE`, without implying RR is globally optimal.

## 8. Repeated-Target Analysis

Within W1-selected branches, historical target frequency is negatively
associated with current feasible count (Spearman `rho = -0.579`, `n = 24`).
This is a useful diminishing-return warning, but it does not establish the
proposed selector-concentration mechanism:

| Group | Mean historical target count |
| --- | ---: |
| RR | 1.583 |
| W1 | 1.417 |
| RR-only | 1.429 |
| W1-only | 1.143 |

W1 did not select historically serviced members more often than RR in these
parents. Therefore the frozen classifier sets
`repeated_target_diminishing_returns=false`. The observed negative association
remains descriptive, not causal.

## 9. W1 Score Calibration

Top-rank calibration is poor:

| W1-selected rank | Branches | Valid-source branches | Feasible branches | Cell winners | Positive val vote |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank 1 | 12 | 5 | 1 | 1 | 0 |
| Rank 2 | 12 | 4 | 2 | 2 | 0 |

Eleven of twelve rank-1 branches produced no feasible candidate. Among all 48
observed branches, W1 score has weak negative rank association with valid-source
count (`-0.138`), feasible count (`-0.142`), and becoming the cell winner
(`-0.128`). Within W1 branches, the corresponding associations are `-0.239`,
`-0.117`, and `-0.128`.

Component directions are not cleanly calibrated. Within W1 branches,
`dhat` is negatively associated with feasible count (`-0.267`), normalized
wait is positively associated (`+0.365`), and failure count is weakly negative
(`-0.114`). Validation calibration is underidentified because only three W1
cell winners have validation observations and all three have zero vote delta.

Together with the feasible RR targets at W1 ranks 3 and 4, this supports
`TARGET_VALUE_ESTIMATION_FAILURE` locally. It does not identify a replacement
weighting formula.

## 10. Branch Realizability Analysis

The W1 responsibility score describes current repair opportunity, not whether
generic evolution can realize a legal prompt for that target. The overall
funnel isolates that mismatch: RR and W1 have nearly equal valid source volume,
but W1 loses four feasible candidates, four feasible branches, and three cell
winners.

The failure is context-dependent. Under Generic context, high-ranked W1 targets
often never pass the Critic. Under Member-aware context, exposure improves but
the resulting candidates more often fail target/vote/no-progress constraints.
This supports `BRANCH_REALIZABILITY_MISMATCH`: responsibility value and proposal
realizability are not interchangeable. The probe used generic evolution and
does not test M20/M2F as an explanation.

## 11. Mechanism Classification

```text
Primary: MULTIPLE_TARGET_ALLOCATION_FAILURES

Supported:
  TARGET_COVERAGE_EXPLORATION_FAILURE
  TARGET_VALUE_ESTIMATION_FAILURE
  BRANCH_REALIZABILITY_MISMATCH

Not supported as primary:
  REPEATED_TARGET_DIMINISHING_RETURNS
```

The deterministic classifier rules are stored in the audit script and were
declared before this narrative was written. They are a post-hoc diagnostic
classifier, not a preregistered causal endpoint.

## 12. Limitations

- These are six deliberately selected V17 S2 parents, not the full trajectory.
- Candidate-level identities and decisions are incomplete in the frozen
  artifact, so candidate-level conclusions are not reported.
- Validation return is identifiable only for each cell's globally selected
  hypothetical transition, not for every branch winner.
- Score correlations are small-sample associations with repeated parent/cell
  structure; they are not independent-sample significance tests.
- Failure discount has little identifying variation here: only two observed
  score rows have nonzero failure count, both with zero normalized wait.
- The result does not show that RR is optimal, that member awareness is harmful,
  or that W1 explains all V17 generalization loss.

## 13. Implications for Next Design

The design space should be narrowed to a combination of:

1. better calibration of target value against downstream realizability;
2. explicit exploration or anti-concentration coverage;
3. a proposal-realizability signal that remains separate from responsibility
   eligibility and value.

The current evidence does not justify implementing a new selector, changing
Common-Safe, or treating historical target count as a causal penalty. This task
ends at diagnosis.

## 14. Reproducibility / Zero-API Audit

Regenerate the analysis-safe tables from immutable project-local artifacts:

```powershell
python scripts\audit_v17_w1_target_allocation.py `
  --out reports\v17_w1_target_allocation_mechanism_audit_20260820
```

The generator opens the historical SQLite cache with `mode=ro`, enables
`query_only`, checks its SHA256 before and after reconstruction, verifies the
24-cell/48-branch inventory, replays the W1 score formula, and checks reconstructed
validation metrics against frozen cell results. Published files contain only
IDs, hashes, counts, scores, ranks, deltas, classifier labels, and sanitized
failure counters.

Verification on the report-producing worktree:

```text
audit-specific tests: 4 passed
canonical suite excluding 3 known historical failures: 622 passed, 3 deselected
full suite: 622 passed, 3 pre-existing failures
compileall: PASS
preflight: PASS
deterministic reconstruction: PASS
sanitization scan: PASS
git diff --check: PASS
```
