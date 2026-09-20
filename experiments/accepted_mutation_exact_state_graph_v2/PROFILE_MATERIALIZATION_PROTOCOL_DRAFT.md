# Accepted-Mutation Profile Materialization v1 — Draft

## Status

```text
AUTHORIZATION_REQUIRED
READY_TO_RUN=false
```

This draft exists only because cache-only recovery was incomplete. It is not an
authorization record and must not be executed without a separate explicit API
authorization and versioned handoff.

The historical model was an unversioned provider alias and the frozen request
had no provider seed. Therefore a future call with the same request identity is
a new provider realization. It cannot recover or overwrite the historical v2
answer profile. Any resulting graph is a prospective replication under a new
experiment identity.

## Frozen scope

Materialize only the 500 missing unique Optimize100 Solver request identities
listed in the v2 `cache_miss_manifest.json`:

```text
M1-P2: 100
M2-P8: 100
M3-P5: 100
M4-P4: 100
M4-P2 unsafe comparison: 100
```

The homogeneous baseline profile is already available and requires no call.
Deduplicate by exact request identity. Do not rerun TeamMiniBatch, Shadow, team
Full evaluation, GEPA, or Reflection.

## Output policy

Persist only example ID, candidate identity, normalized option or `INVALID`,
validity, correctness, request identity, and frozen prompt hash. Raw reasoning
and raw provider text may be parsed transiently but must not be persisted.

## Budget and isolation

```text
successful Solver calls <= 500
transport attempts <= 2000
GEPA = 0
Reflection = 0
Validation50 = 0
Test50 = 0
write-back = 0
scheduler = 0
persistent realizability update = 0
```

No efficacy-based stopping or quota expansion is permitted. Any identity,
model, split, cache, parsing, ledger, or budget mismatch fails closed.

After materialization, stop. Exact graph construction is a separate zero-API
audit over the frozen sanitized profile matrix.
