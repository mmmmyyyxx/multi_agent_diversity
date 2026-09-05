# Vote-Aligned Generic Shadow Pilot v1

This experiment reuses the frozen anti-overfitting split and cross-fit mapping:
TrainDev150 is divided into three 50-row folds; each seed uses Optimize100 and a
disjoint Shadow50, while Validation50 is evaluated only once after the final
state is frozen. Test50 is prohibited.

The paired arms both use experimental D2 RR + Generic, Common-Safe, two targets,
two source candidates per target, the same loss-blind revision opportunity, and
the same winner-only Shadow gate. P0 uses canonical actionable-member RR. P1
changes only target scheduling to a deterministic hierarchy:

1. direct flip;
2. near margin, defined as a non-flipping positive-margin correction whose
   exact post-correction plurality margin is zero;
3. pure coverage;
4. canonical actionable-member RR fallback.

Lane-local choices use a per-seed, per-protocol, stateful deterministic RR
cursor. The second slot reruns the hierarchy after removing the first target.
No W1 score, weighted objective, responsibility-conditioned proposal context,
new Critic, M2F, transfer predictor, validation feedback, or test access is
permitted.

By explicit user amendment before Phase B, the initial execution is limited to
Seed75 (Optimize folds A+B, Shadow fold C), comprising exactly two trajectories
in P0 then P1 order. Seeds76/77 are not part of this run and may not be started.
This single-seed result is descriptive pilot evidence, not a multi-seed claim.

The primary endpoint is paired final Validation Vote accuracy. Interpretation
is frozen to one of: `VOTE_ALIGNED_SPECIALIZATION_SUPPORTED`,
`VOTE_STRUCTURE_IMPROVED_WITHOUT_FINAL_GAIN`,
`NO_CLEAR_VOTE_ALIGNED_BENEFIT`, or `SPECIALIZATION_HARMFUL`.
