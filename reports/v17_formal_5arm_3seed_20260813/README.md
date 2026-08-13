# V17 Formal Five-Arm Results

Frozen-method comparison on Seeds 56-58. Validation is a post-training audit and did not select states. The test split was historically exposed during development; these results are not an untouched paper-heldout claim.

All 15 training cells, 15 validation evaluations, and 15 one-shot test evaluations passed their final protocol gates. The execution-source commit is `76e6960969320786a767c9e94ef3e357735f4b32`.

Two post-execution evaluation-infrastructure corrections were required and are disclosed in `audit_corrections.json`. They did not change prompts, checkpoints, final-state hashes, method semantics, datasets, seeds, budgets, metrics, or classifiers. The original training-gate failure and failed pre-call evaluation attempts remain preserved in the private run directory and are not substituted for experimental observations.

Frozen test classifiers:

- C01 Generic vs Static: `MAJORITY_POSITIVE`
- C12 Module1 vs Generic: `NOT_SUPPORTED`
- C23 R-M20 vs G-Matched: `MAJORITY_POSITIVE`
- C34 R-M2F vs R-M20: `MAJORITY_POSITIVE`
- C14 Full method vs Generic: `NOT_SUPPORTED`
- C04 Full method vs Static: `CONSISTENT_POSITIVE`
