# Train75 / Validation50 / FormerTest125 Split Balance Audit

This is a zero-API retrospective audit. It does not modify or rerun D0-D5.
FormerTest125 is already converted to development validation; no untouched held-out test remains.

**Verdict: `SPLIT_IMBALANCE_AND_METHOD_INTERACTION_SUPPORTED`**

## Arm ranking by split

| Arm | Train75 Vote | Validation50 Vote | FormerTest125 Vote |
|---|---:|---:|---:|
| D0 | 0.6578 | 0.6467 | 0.6800 |
| D1 | 0.8267 | 0.6733 | 0.7547 |
| D2 | 0.8356 | 0.7067 | 0.7440 |
| D3 | 0.8533 | 0.6933 | 0.7280 |
| D4 | 0.8400 | 0.6933 | 0.7440 |
| D5 | 0.8267 | 0.5867 | 0.7333 |

## Interpretation

The audit separates split composition from method transfer. Label, lexical-cluster,
structural, and D0 difficulty effects are reported independently. Method-by-split
difference-in-differences subtract the D0 split gap, so a large remaining value is
not explained by overall split difficulty alone.

The result is development evidence. It must not be presented as a fresh test confirmation.
