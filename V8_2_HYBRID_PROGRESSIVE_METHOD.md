# V8 Stable QD Lineage Method

The historical setting name `shared_vote_tcs_competence_depth2_progressive_residual_hybrid` now runs `v8_stable_qd_lineage`. It no longer uses a safe/exploit/explore beam interpretation.

## Current Pipeline

```text
hybrid competence/boundary target selection
-> Teacher-Critic-Student candidate generation
-> competence and validity guards
-> per-agent quality-diversity archive
-> fixed-probe evaluation of unique beam prompts
-> offline joint team enumeration
-> quality feasible region
-> hierarchical integer-count quality bands
-> behavioral complementarity selection
-> committed lineage stability and peer-collapse control
```

Candidate reward measures candidate quality. Behavioral and mechanism diversity act only in the QD archive and joint active-team selector.

## Early And Late Search

Before commitment, self-drift penalties are disabled so identical initial prompts can separate. After one stable quality-passing snapshot, an agent is provisional; after two it can commit a lineage. Later prompt changes are evaluated relative to that anchor and use two-snapshot switch hysteresis.

Behavioral residual complementarity on the fixed optimization probe is primary. Normalized mechanism embedding is secondary. Prompt textual diversity is not optimized.

## Search-Space Preservation

The current setting keeps a long-term Safe archive separate from its three-item joint representative beam. Mildly regressing but novel candidates go to Probation for bounded parent-only exploration. Rejected or duplicate candidate batches trigger failure-aware TCS refill, while archived niches receive round-robin reproduction opportunities. Joint combinations recompute team-relative rescue and use two-fold stable diversity after hierarchical quality bands.

## Compatibility

The setting name is unchanged, but old V8.2 checkpoints cannot be restored because candidate retention, active-team selection, and persistent state semantics changed.
