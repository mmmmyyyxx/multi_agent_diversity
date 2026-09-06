# Cross-Repo P0 Parity Audit

This is a zero-API, read-only audit of Seed75/76/77 P0 ExternalValidation50
evidence from MARS, Independent-GEPA, and Diversity. Test50 was not accessed.

## Decision

`PARITY_VIOLATION_IDENTIFIED`

The 131-byte P0 and the complete 598-byte rendered system prompt are byte
identical. The logical 50-case set, order, and gold labels also match. However,
the actual requests are not identical:

- Diversity preserves CRLF bytes in the question payload; MARS and GEPA
  normalize the same logical cases to LF.
- MARS sends `seed=75/76/77` to the provider; GEPA and Diversity omit a
  provider seed.
- operational retries, parser-invalid retries, and cache identity/reuse scopes
  differ.
- the model is the same `qwen3-8b` alias, but no immutable provider snapshot is
  recorded and exact historical endpoint parity is not recoverable zero-API.

All 450 historical P0 rows were recovered and valid. Pairwise parsed-label and
correctness disagreements are reported per seed. These disagreements are
compatible with stochastic hosted-model behavior, but provider nondeterminism
is **not** the only remaining explanation because request and execution
contracts already differ.

## Baseline-table consequence

The three raw P0 accuracies must not be presented as one formally parity-matched
baseline table. They may be shown as repository-specific observed baselines
with explicit contract annotations. A formal shared baseline requires one
canonical byte-level question renderer, one provider-seed policy, one retry and
invalid-output policy, one cache policy, and a newly frozen common evaluator
contract before any new calls.
