# V18 accepted-commit transfer decomposition

This retrospective audit freezes the diagnostic rules before reading per-commit
Seed68-70 outcomes. It compares the same A/C arm labels across seeds but never
treats commits after trajectory divergence as matched causal pairs.

For each accepted commit it records train Vote/target/Oracle deltas, validation
Vote gains and losses, validation target and Oracle deltas, coverage recovery,
support deepening, wrong-coalition changes, and persistence or overwrite of each
validation Vote gain. Losses are separated into initial-capability collateral
and overwrite of a capability created earlier in the same trajectory.

Every trajectory must satisfy the telescoping identity between the sum of its
accepted-transition validation Vote deltas and its initial-to-final Vote change.
The audit uses no API, performs no new validation or test evaluation, and does
not modify historical artifacts, the method, or Critic behavior.
