# Local Experiments

Local raw experiment and deterministic-smoke roots belong in this directory:

```text
experiments/runs_*/
```

Raw run contents, caches, checkpoints, model logs, prompts, and responses are
ignored by Git. Publish only explicitly sanitized, secret-free analysis under
`reports/`.

Historical reports may retain the original root-relative path recorded when a
run was produced. Those paths are provenance, not the current storage layout.
