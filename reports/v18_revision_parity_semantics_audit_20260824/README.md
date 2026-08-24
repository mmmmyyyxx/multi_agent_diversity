# V18 Revision-Parity Semantics Audit

This was an independent zero-API audit. It did not rerun trajectories, add
revision calls, modify raw evidence, run the scientific analyzer, or inspect
online-accumulation outcomes.

## Status

```text
V18 execution completed
original frozen audit = FAIL / HOLD
independent semantics audit = PASS
post-hoc corrected gate = PASS
```

The original gate remains preserved and is not superseded. The separately
named corrected gate changes only the representation used for revision parity.

## Finding

The frozen execution policy spends one revision opportunity per valid source.
An invalid revision output is a legal terminal outcome of that opportunity and
does not create an evaluable candidate row. Therefore attempt parity must join
valid source rows to revision-attempt events; evaluable-row parity is a
separate valid-output persistence check.

Across all six trajectories:

- conceptual source budget: 192
- valid sources: 54
- revision attempts: 54
- valid revision outputs/evaluable rows: 50
- invalid revision outputs: 4
- evaluable revision rows: 50

Every valid source had exactly one attempt. Every valid revision output had
exactly one evaluable row. The four invalid outputs consumed their frozen
opportunities and correctly produced no evaluable row.

The arms had equal prospective budgets and the same one-attempt-per-valid-source
policy. Absolute realized attempt counts differ because the online trajectories
produced different valid-source counts; that is not a compute-policy mismatch.

## Scope

No scientific result is reported or interpreted here. Running the frozen V18
scientific analyzer requires a separate decision after accepting or rejecting
this post-hoc gate correction.
