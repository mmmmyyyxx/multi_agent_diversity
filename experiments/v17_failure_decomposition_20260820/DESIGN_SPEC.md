# V17 Frozen Failure Decomposition Audit

This is a post-hoc, zero-provider reconstruction of the completed V17 five-arm,
three-seed trajectory.  It explains the frozen S1-to-S2 generalization result
and the S2-to-S3-to-S4 recovery without selecting a method or changing runtime
behavior.

Frozen population: S0--S4, Seeds 56--58, the already completed train,
validation, and test final states.  The source baseline is
`ef9124eb2ddbfbe8d04c3c849c08a9e6875c7e61`.

The audit is read-only over checkpoints, final-state caches, evaluation
summaries, and trajectory metadata.  It must make zero API, model, solver,
optimizer, and evaluator calls.  It never reruns training or held-out
evaluation, changes prompts, changes selection, changes W1, changes routing,
or changes any historical classifier.

Analyses are descriptive: split transfer, paired vote transitions,
complementarity-to-vote conversion, target concentration, update throughput,
member transfer, and the M20/M2F recovery trajectory.  Causal wording is not
permitted unless separately established by frozen fixed-parent evidence.
