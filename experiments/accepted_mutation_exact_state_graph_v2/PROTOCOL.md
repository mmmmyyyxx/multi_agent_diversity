# Accepted-Mutation Exact State Graph v2

## Purpose

This is a cache-only attempt to recover the exact Optimize100 answer profiles
used by `accepted_local_mutation_team_transfer_v2` and, only if recovery is
complete, construct the exact 24-state compositional graph. It preserves the
bounded v1 proof and may replace only the previously unidentified state and
edge identities.

## Zero-API boundary

Provider access is disabled. Solver, GEPA, Reflection, Validation50, Test50,
write-back, scheduler, and persistent-realizability calls are all zero.
Historical run and execution-bundle files are read-only and must remain
byte-identical.

## Required evidence

The audit reconstructs the exact request identity for each frozen candidate
prompt and each Optimize100 example, verifies it against the historical v2
provider ledger, and searches existing exact-request JSON caches and persistent
SQLite caches. A response is usable only if request identity, prompt identity,
example identity, model contract, and historical provenance all match.

Only normalized option, validity, and correctness may enter a published profile
matrix. Raw provider responses and reasoning must never enter the report.

## Fail-closed rule

If any required response is unavailable, the audit stops before graph
construction with:

```text
CACHE_ONLY_PROFILE_RECOVERY_INCOMPLETE
```

It publishes a sanitized `cache_miss_manifest.json` containing request hashes,
candidate identities, example IDs, and counts. It must not create empty or
estimated files bearing exact-state names.

The existing conclusion remains frozen:

```text
SAFE_SYMMETRY_BREAKING_CAN_UNLOCK_PLURALITY_GAIN
```

This means at least one Common-Safe path reaches a positive triple. It does not
claim that the concrete triple is known or that all four safe mutations can be
committed sequentially.

## Complete-recovery branch

Only complete recovery permits exact enumeration of the 16 safe-component
states, the full 24-state comparison catalog, exact Common-Safe edges,
structural pivotality, positive triples, shortest safe paths, pivotal prefixes,
and four-mutation sequential reachability.
