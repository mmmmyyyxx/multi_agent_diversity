# Split and access audit

No project split for the five requested benchmarks is frozen in the tracked repository. No real benchmark example IDs or available counts were discovered. Consequently no real proposed ID assignment can be generated honestly; all five record `BENCHMARK_SPLIT_NOT_FROZEN`. A synthetic two-row fixture proves deterministic file hashing, ordered IDs, SHA mismatch refusal, split coverage, pairwise disjointness, and a `PROPOSED` manifest that refuses `require_frozen()`.

The `SplitManifest` stores benchmark/source/split identity, explicit selection rule and optional seed, ordered IDs, search/shadow/validation/test IDs, group hashes, status and total identity hash. It never samples or promotes a proposed assignment implicitly. A real proposed split awaits an explicitly selected upstream materialization and scientific split rule.

BBH Validation50 calls = 0; BBH Test50 calls = 0. No benchmark heldout was used in fake search.
