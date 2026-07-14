# Stage 3 Failure Report

Stage 3 stopped after the first completed ablation run because the mandatory
candidate-delta audit failed.

- Run: `shared_accuracy_only_tcs_vote_first_seed42`
- Exit code: `0`
- Final test vote accuracy: `0.624`
- Final test mean individual accuracy: `0.616`
- Selected epoch: `1`
- TCS optimizer candidates with valid metadata: `130/130`
- Accuracy-only reward identity failures: `0`
- Candidate delta inconsistency rows: `204/204`

The reward was correct: every candidate used the target agent's own accuracy.
The logging contract was not correct. Accuracy-only candidate evaluation did
not evaluate the baseline team on the same batch, so baseline target/team
accuracy and their deltas were serialized as zero. This run is preserved for
audit but is excluded from Stage 3 comparisons.

The next model run requires a protocol upgrade and a fresh output root.
