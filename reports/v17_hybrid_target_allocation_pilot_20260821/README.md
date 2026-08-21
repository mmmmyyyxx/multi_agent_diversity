# V17 Responsibility-Guided Exploit–Explore Target Allocation Pilot

## 1. Objective

This prospective fixed-parent pilot asks whether one responsibility-guided W1
exploitation branch plus one responsibility-constrained Round-Robin exploration
branch can reduce W1 target-allocation misalignment without adding a learned
realizability model.

The result is **HYBRID_THROUGHPUT_ONLY**. Hybrid recovered two feasible branches
and two hypothetical Common-Safe updates that W1 Top-2 missed, but none produced
a positive realized validation vote delta. The experiment therefore supports a
local search-realization benefit, not a validation-vote or full online
generalization benefit.

## 2. Frozen Prior Evidence

The prior V17 decomposition identified target allocation, rather than the
member-aware residual context itself, as the dominant local negative source.
The prior W1 audit found multiple allocation failures: target-coverage
exploration failure, target-value estimation failure, and branch-realizability
mismatch. Repeated-target diminishing returns was not treated as the primary
cause.

Those earlier diagnostic parents were used only for zero-API reconstruction
checks. None enters this prospective primary sample.

## 3. Minimal Hybrid Mechanism

The three compute-matched arms were frozen before Phase B:

| Arm | Target 1 | Target 2 |
|---|---|---|
| `W1_TOP2` | W1 rank 1 | W1 rank 2 |
| `RR_TOP2` | first eligible in frozen RR order | second eligible in frozen RR order |
| `HYBRID_EXPLOIT_EXPLORE` | W1 rank 1 | first remaining eligible in frozen RR order |

Responsibility defines the legal opportunity set. RR only protects one scarce
proposal branch from W1 ranking misalignment. Target selection is frozen before
proposal generation and cannot use candidate feasibility, validation, or test
outcomes.

Everything else is held fixed: member-aware residual/search context, generic
proposal evolution, two targets, two source slots per target, one loss-blind
generic revision per valid source, fixed peers, Common-Safe, ranking, and the
max-one hypothetical update rule. M20 and M2F are not used.

## 4. Prospective Parent Selection

Six new V17 S2 parents were deterministically selected: two each from Seeds 56,
57, and 58, at the nearest eligible states to the one-third and two-thirds
positions of each seed's ordered unused eligible-state sequence. Selection did
not inspect validation/test results, historical branch feasibility, selector
outcomes, or WOULD_COMMIT outcomes.

All six parents had at least two responsibility-eligible members and passed
deterministic W1, RR, responsibility-state, train, and validation
reconstruction. Hybrid substituted a different second target from W1 rank 2 in
all 6/6 parents.

## 5. Source Freeze

Phase A froze execution commit
`43230a5734a971ead70841d71277105512fa0adc`, 237 tracked source files, six
prospective parents, the canonical branch identity, the compute contract, the
evaluation protocol, and the classifiers. Its registry content hash is
`02fb91798c3739b724a2f5d29b602b2d21d6ba25549bdee58a08f0d1ca885b75`.

Phase A was strictly zero API. The old six diagnostic parents were excluded
from the primary sample.

## 6. Execution Gate

Both gates passed.

| Check | Result |
|---|---:|
| prospective parents | 6/6 |
| arm cells | 18/18 |
| conceptual branches | 36/36 |
| deduplicated actual branches | 20 |
| candidate records | 36 |
| Phase B recorded API calls | 3,015 |
| new test calls | 0 |
| actual prompt commits | 0 |
| trajectory mutations | 0 |

Exact-match branch reuse reduced actual generation while preserving two
conceptual branches and four source slots per arm cell. Cell decisions were
frozen using train-side evidence before validation evaluation.

## 7. Target Selection Comparison

All arms selected 12 conceptual branches and covered all five member IDs across
the six parents. Hybrid always retained W1 rank 1, and its exploration target
was always responsibility-eligible, distinct, and first in the frozen RR order
after excluding the exploitation target.

## 8. Branch Feasibility Funnel

| Funnel stage | W1 Top-2 | RR Top-2 | Hybrid |
|---|---:|---:|---:|
| selected branches | 12 | 12 | 12 |
| branches with valid source | 2 | 4 | 3 |
| valid sources | 4 | 8 | 6 |
| valid revisions | 4 | 8 | 6 |
| feasible branches | 0 | 3 | 2 |
| feasible candidates | 0 | 4 | 3 |
| cell-best branches | 0 | 3 | 2 |
| cells with WOULD_COMMIT | 0 | 3 | 2 |
| positive validation vote delta | 0 | 0 | 0 |
| positive validation oracle delta | 0 | 2 | 2 |

Hybrid recovered 2/3 of the feasible-branch gap between W1 and RR. The recovery
occurred at branch realization and Common-Safe selection, not at validation
vote conversion.

## 9. WOULD_COMMIT Analysis

W1 produced no hypothetical update. RR produced three and Hybrid produced two.
All decisions used the unchanged train-side Common-Safe, ranking, and max-one
rules. No winner was written back to a parent or future trajectory.

## 10. Validation Vote Results

| Arm | realized vote-delta sum | mean per parent |
|---|---:|---:|
| W1 Top-2 | 0 | 0.000 |
| RR Top-2 | 0 | 0.000 |
| Hybrid | 0 | 0.000 |

Hybrid minus W1 is 0/6/0 for parent-level vote wins/ties/losses. Hybrid minus
RR is also 0/6/0. Thus `HYBRID_VOTE_BENEFIT_SUPPORTED=false`.

## 11. Validation Oracle Results

| Arm | realized oracle-delta sum | mean per parent |
|---|---:|---:|
| W1 Top-2 | 0 | 0.000 |
| RR Top-2 | +4 | +0.667 |
| Hybrid | +4 | +0.667 |

Hybrid minus W1 is 2/4/0 for oracle wins/ties/losses, so the frozen secondary
classifier reports `HYBRID_ORACLE_BENEFIT_SUPPORTED=true`. This suggests that
the two exploration-sourced hypothetical updates broadened member-level
coverage without converting any additional validation plurality votes in this
sample.

## 12. Exploit vs Explore Attribution

Both Hybrid feasible branches, both cell winners, and both WOULD_COMMIT events
are attributable to the RR exploration branch. Their combined realized
validation contribution is 0 vote and +4 oracle. The retained W1 exploitation
branches produced no feasible candidate in these six parents.

This is direct selector-local evidence that the second constrained-exploration
branch found opportunities missed by W1 Top-2. It is not evidence that RR is
optimal or that realizability has been solved.

## 13. Candidate-Level Failure Structure

All 36 published candidate records are sanitized. Every recorded candidate is
parser-valid. Across arm-expanded records, 29 candidates were rejected for
`no_target_or_vote_progress` with or without `target_regression`; seven passed
Common-Safe. Because exact-match branches are reused across arms, these counts
describe conceptual arm exposure rather than 36 independent generations.

W1 had eight valid candidate records but zero feasible candidates. Hybrid had
12 valid records, three feasible candidates, and two selected winners. RR had
16 valid records, four feasible candidates, and three selected winners.

## 14. Frozen Classifiers

| Frozen classifier | Result |
|---|---|
| `HYBRID_VOTE_BENEFIT_SUPPORTED` | false |
| `HYBRID_FEASIBILITY_RECOVERY_SUPPORTED` | true |
| `HYBRID_ORACLE_BENEFIT_SUPPORTED` | true |
| recovered fraction | 0.667 |
| final diagnosis | `HYBRID_THROUGHPUT_ONLY` |

The final label follows the preregistered rule: feasibility recovery is
supported, validation vote benefit is not, and the Hybrid-minus-W1 mean vote
difference is non-negative.

## 15. Limitations

This is a six-parent, selector-local fixed-parent pilot. Branch reuse controls
generation noise for exact matches but also means arm-expanded records are not
independent samples. WOULD_COMMIT is hypothetical, no successor trajectory is
formed, and validation is used only for post-decision evaluation. No candidate
or state receives a new test evaluation.

The positive oracle result and null vote result should not be interpreted as
proof of online generalization, optimality of RR, or a solved realizability
problem.

## 16. Implications for Next Stage

The preregistered `HYBRID_THROUGHPUT_ONLY` action is to inspect why recovered
feasible branches and broader oracle coverage did not transfer into validation
vote gains. This task does not change the method or start an online trajectory.

## 17. Reproducibility and Sanitization

Published files contain only relative identifiers, hashes, counters, ranks,
scores, deltas, booleans, and sanitized rejection labels. They exclude prompts,
questions, gold/model answers, raw responses, credentials, endpoints, SQLite
content, checkpoints, private caches, and absolute paths. The included SHA-256
manifest covers the generated analysis artifacts; verification and test results
are recorded separately.
