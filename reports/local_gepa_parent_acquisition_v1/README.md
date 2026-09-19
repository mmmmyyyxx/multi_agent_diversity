# Phase A parent acquisition

Status: **PARENTS_FROZEN_PHASE_B_NOT_AUTHORIZED**. Acquired 100 Optimize100 baseline responses,
constructed 5 eligible complete parent tasks, and selected 4.
Source commit: `83cad045f0fc6a796c9d6244aed7db7d5ac05671`.

- `seed78_update0_member1`
- `seed78_update0_member2`
- `seed78_update0_member3`
- `seed78_update0_member4`

All tasks originate from one shared-P0 baseline state, with five equal-weight
members sharing exact-request realizations. They are distinct local tasks, not
independent source-state replicates. Selection used the preregistered exact
source/target/lane coverage rule and stable subset hash, without GEPA outcomes.
The frozen local validation set is Optimize-derived under the existing 4/4/4
task definition and overlaps search evidence; it is not held-out Validation50.

Parent acquisition cost: 100 provider attempts,
100 successful Solver requests,
19665 input tokens and 6448 output tokens.
Acquisition contributes zero to the acceptance-rate denominator.
Layer-1 optimization cost, Reflection, GEPA proposals, team evaluations,
write-backs, persistent realizability updates and Validation50/Test50 access
are all zero. No optimization-effectiveness conclusion is supported by Phase A.

The original zero-API preparation remains PARENT_CATALOG_INSUFFICIENT.
This is an explicit separately versioned acquisition amendment. Complete tasks
and raw observations remain private; published identities and hashes bind them.
Phase B requires separate authorization; the 4x8 pilot has not run.

Validation: 10 new tests and 21 frozen focused tests passed, with compileall,
governance/method preflights and independent zero-API reconstruction. The main
full suite has one reproduced historical cache-coverage failure. Full-suite
execution in the isolated checkout also lacks historical ignored run data;
verification_summary.json records this limitation without claiming a full pass.
