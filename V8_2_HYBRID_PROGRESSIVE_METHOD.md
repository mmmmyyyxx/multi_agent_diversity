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
-> epsilon-Pareto quality frontier
-> behavioral complementarity selection
-> committed lineage stability and peer-collapse control
```

Candidate reward measures candidate quality. Behavioral and mechanism diversity act only in the QD archive and joint active-team selector.

## Early And Late Search

Before commitment, self-drift penalties are disabled so identical initial prompts can separate. After three stable quality-passing selections, an agent can commit a lineage. Later prompt changes are evaluated relative to that anchor and use two-epoch switch hysteresis.

Behavioral residual complementarity on the fixed optimization probe is primary. Normalized mechanism embedding is secondary. Prompt textual diversity is not optimized.

## Compatibility

The setting name is unchanged, but old V8.2 checkpoints cannot be restored because candidate retention, active-team selection, and persistent state semantics changed.
