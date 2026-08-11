# v15 Bottleneck Isolation Offline Audit

This directory contains the second and final planned offline diagnosis for v15 seeds 48–50, S2 only. It separates opportunity-normalized responsibility propagation from rejected-candidate common-safe conflict geometry.

The audit used existing train trajectories and opened the setting-local solver SQLite caches strictly read-only. Cache payloads were projected in memory to answer/validity only; the output contains no prompt text, question text, gold/model answers, raw responses, traces, credentials, endpoints, cache contents, or absolute paths.

No API/model call, training run, validation/test evaluation, method modification, or image generation occurred. Start with `AUDIT_REPORT.md`; machine-readable verdicts are in `audit_summary.json` and exact derived evidence is under `tables/`.

This is method-development diagnosis over three seeds, not a test-set, causal, statistical-significance, or generalization claim.
