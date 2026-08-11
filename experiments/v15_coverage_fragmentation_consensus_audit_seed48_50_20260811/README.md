# v15 Coverage Fragmentation → Consensus Conversion Audit

This directory contains a deterministic, train-trajectory-only audit of v15 Development Formal seeds 48–50 for S0, S1, and S2.

No API/model call, training run, validation evaluation, or final-test evaluation was performed. No test artifact was read. The analysis consumed only existing train artifacts plus the published train-only consistency table. Raw prompts, questions, gold answers, model answers, responses, caches, and checkpoints are not copied into this directory.

The user requested no images, so this audit intentionally emits only Python, CSV, JSON, and Markdown artifacts. `audit_manifest.json` records every consumed path and SHA256.

Start with [AUDIT_REPORT.md](AUDIT_REPORT.md), then use [audit_summary.json](audit_summary.json) for the machine-readable verdict and `tables/` for event-level evidence. The audit script is `scripts/audit_coverage_consensus.py`.

This is descriptive method-development evidence over three seeds. It is not a causal, significance, test-set, or generalization claim.
