# v15 three-seed Development Formal

This directory contains sanitized, analysis-ready evidence for Seeds 48/49/50 across Static, S0, S1, and S2. Training used no validation or training-time test. Frozen final states were tested exactly once per seed-setting after all three training runs completed.

All training and test-extension gates passed. Training artifact mutation count is zero. The historical disambiguation_qa test split was previously exposed during method development, so these results are development generalization evidence, not untouched paper-heldout evidence.

See `three_seed_summary.md` for the primary table and interpretation; CSV files support paired and aggregate reanalysis. `train_diagnostics/` and `test_diagnostics/` contain sanitized per-run mechanism evidence.
