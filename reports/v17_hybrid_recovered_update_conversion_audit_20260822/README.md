# V17 Hybrid/RR Recovered-Update Validation Vote-Conversion Audit

## Conclusion

This zero-API row-level audit supports:

```text
SINGLETON_COVERAGE_RECOVERY_WITHOUT_VOTE_CONVERSION
```

The two Hybrid and three RR conceptual WOULD_COMMIT cells correspond to only
three unique candidate transitions because two Hybrid winners exactly reuse RR
branches. Across those three unique transitions, four validation examples gain
Oracle coverage. Every gain is `G=0 -> G=1`; every new correct member remains a
singleton in a vote-wrong team. There are no validation vote gains and no vote
losses, so the zero vote delta is not gain/loss cancellation.

The evidence therefore refines the prior `HYBRID_THROUGHPUT_ONLY` result:
responsibility-constrained exploration recovered realizable member-level
opportunities and previously absent validation coverage, but that local
coverage remained vote-neutral under plurality.

## Audit Boundary and Provenance

- Audit mode: immutable-artifact reconstruction.
- Model/API calls: 0.
- New validation or test evaluation: 0.
- Prompt commits: 0.
- Trajectory mutations: 0.
- Source experiment commit: `43230a5734a971ead70841d71277105512fa0adc`.
- Auditor commit: `0a4dd702d4fac546905cc8c0914ef1fd624c41f2`.
- Source cache and run-artifact SHA-256 values were identical before and after.

No algorithm, selector, Common-Safe rule, candidate, parent, or frozen metric
was changed. The audit reads the existing validation observations from the
frozen cache in SQLite immutable/read-only mode and reconstructs all 50 rows
for each unique candidate transition.

## Conceptual Versus Unique Updates

| Scope | Count |
|---|---:|
| Hybrid conceptual WOULD_COMMIT cells | 2 |
| RR conceptual WOULD_COMMIT cells | 3 |
| all conceptual cells | 5 |
| deduplicated candidate transitions | 3 |
| validation rows per unique transition | 50 |

The Seed56 and Seed58 Hybrid winners are the exact same canonical candidate
transitions used by RR. They are counted once in causal row-level totals and
retained twice only in the conceptual arm table.

## Unique-Transition Summary

| Parent | Conceptual arms | Train target / vote | Validation target / vote / oracle | Target gains / losses |
|---|---|---:|---:|---:|
| Seed56 prospective parent 1 | Hybrid + RR | `+7 / 0` | `0 / 0 / +2` | `7 / 7` |
| Seed57 prospective parent 2 | RR | `+1 / 0` | `-7 / 0 / 0` | `1 / 8` |
| Seed58 prospective parent 2 | Hybrid + RR | `+1 / 0` | `-4 / 0 / +2` | `2 / 6` |

The two coverage-producing transitions are the two transitions exposed through
Hybrid exploration. The third RR-only transition produces no validation Oracle
gain and has substantial target-member regression.

## Oracle Coverage Structure

The four deduplicated Oracle-gain rows are:

| Count | G transition | H transition | M transition | State after update |
|---:|---|---|---|---|
| 1 | `0 -> 1` | `4 -> 3` | `-4 -> -2` | singleton correct, vote wrong |
| 3 | `0 -> 1` | `5 -> 4` | `-5 -> -3` | singleton correct, vote wrong |

Thus the Oracle gains do not come from `G=1 -> G=2`. They introduce the first
correct member on previously uncovered examples.

There are also five Seed56 target gains of `G=1 -> G=2`; these do not count as
new Oracle coverage because the parent team was already covered. They remain a
nonwinning correct coalition and do not create vote gains.

## Distance to Plurality

All four new singleton-correct rows remain far from a plurality win:

- One row needs three additional gold votes if the dominant wrong count is not
  reduced further.
- Three rows need four additional gold votes under that same conservative
  definition.
- Every row needs at least two dominant-wrong-to-gold member flips under the
  most favorable two-margin-points-per-flip interpretation.

This is not a one-vote near miss. Hybrid recovered the first layer of coverage,
but the recovered examples still require coalition-level conversion.

## Vote Gains and Losses

Across all 150 reconstructed validation observations for the three unique
transitions:

```text
vote gains = 0
vote losses = 0
```

The net-zero vote result is therefore not caused by positive and negative vote
changes cancelling. The plurality outcome is unchanged row by row.

## Wrong-Coalition Structure

The dominant wrong coalition is not completely static on the new coverage
rows. It decreases by one on all four Oracle gains. However, the remaining
wrong coalition is still size three or four, leaving margins of `-2` or `-3`.

Across all 33 structurally changed rows, not just Oracle gains:

| H direction | Count |
|---|---:|
| reduced | 11 |
| increased | 22 |

So the coverage-producing rows show aligned movement in both `G` and `H`, but
the candidate transitions also cause broader wrong-coalition reshuffling that
more often increases than decreases `H`. This broader behavior helps explain
why local member repair does not imply team-vote conversion.

## Why Common-Safe Accepted on Train

All three unique winners satisfy the frozen train-side acceptance path:

```text
train target deltas = [+7, +1, +1]
train vote deltas   = [ 0,  0,  0]
train vote gains    = 0
train vote losses   = 0
```

Common-Safe allows a candidate when target correct count strictly improves,
team vote count does not regress, terminal invalid does not increase, and the
remaining member objectives do not regress. It does not require a strict train
vote gain, and it is not a validation generalization guarantee.

On validation, the same three transitions have target deltas `[0, -7, -4]`,
vote deltas `[0, 0, 0]`, and Oracle deltas `[+2, 0, +2]`. The discrepancy is
therefore a transfer/structure issue: train-side target improvement passes the
frozen safety rule, while held-out behavior redistributes target competence and
occasionally creates first-member coverage without reaching plurality.

## Interpretation

The observed chain is:

```text
allocation
  -> realizable member-level update
  -> first-member coverage on some held-out rows
  -> no plurality conversion
```

This is evidence that Hybrid repairs part of the allocation-to-coverage
bottleneck. It simultaneously exposes a second bottleneck between singleton
coverage and coalition-level vote conversion. It does not establish that
Hybrid improves a full online trajectory, that RR is optimal, or that a method
change is currently justified.

## Published Artifacts

- `summary.json`: audit gates and aggregate causal facts.
- `conceptual_update_level.csv`: the two Hybrid and three RR conceptual cells.
- `unique_update_level.csv`: three deduplicated candidate transitions.
- `transition_level.csv`: sanitized hashes and G/H/M row transitions only.
- `source_evidence.json`: immutable evidence hashes and source/auditor identity.
- `sha256_manifest.json`: published artifact integrity.

No prompt text, question text, gold/model answer, raw response, credential,
endpoint, private cache content, checkpoint content, or absolute local path is
published.
