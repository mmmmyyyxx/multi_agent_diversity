# V17 Formal Five-Arm Experiment

This preregistration freezes a 5-arm by 3-seed comparison on BBH
`disambiguation_qa`. Seeds 56-58 are the formal seeds; Seeds 47-55 are
development evidence and are excluded from formal averages.

Validation is a post-training generalization audit only. It cannot select or
mutate a final state. Test evaluates the same frozen final states once after a
pre-test seal. The test split was historically exposed during development, so
this experiment is not described as an untouched paper heldout evaluation.

S0 is static. S1 is the new explicit generic 2x2 dual-round-robin baseline.
S2, S3, and S4 reuse the frozen G-Matched, R-M20, and R-M2F settings. Equal
opportunity ceilings and common hard run ceilings are used; realized compute
need not be identical. No outcomes may change the method, cases, seeds, order,
metric, classifier, or thresholds.
