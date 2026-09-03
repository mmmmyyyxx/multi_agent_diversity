# Qwen3-8B No-Semantic-Critic Light Replication

Official frozen gate: **PASS**. This is a one-seed mechanism replication, not a formal efficacy claim and not a causal comparison with Qwen3-14B.

## Frozen model allocation

- Solver: `qwen3-8b`, thinking disabled.
- Teacher and Student: `qwen3.7-flash`.
- Canonical semantic Critic: `qwen3.7-flash`.
- C uses the deterministic hard-safety gate and makes no semantic-Critic API call.

## Results

| Metric | A Canonical | C No Semantic Critic |
|---|---:|---:|
| Student reach | 13 | 16 |
| Feasible candidates | 12 | 17 |
| Accepted commits | 6 | 6 |
| Final train Vote | 50/75 | 56/75 |
| Final validation Vote | 29/50 | 26/50 |
| Final validation Oracle | 39/50 | 40/50 |

Frozen front-end label: **FRONTEND_THROUGHPUT_REPLICATED**. C again increased Student reach and feasible supply, while commit count tied.

Frozen transfer label: **TRANSFER_INSTABILITY_OBSERVED**. C's 6 commits comprise 3 positive, 2 zero, and 1 negative validation-Vote transitions. Final C validation Vote is -3 relative to A, despite a +6 train-Vote difference.

Frozen ranking label: **RANKING_NOT_IMPLICATED_BY_AVAILABLE_TRAIN_EVIDENCE**. Across C's 6 committed feasible pools, 0 unselected alternative train-Pareto-dominated its winner. This does not eliminate ranking: unselected alternatives received no new validation replay, so the counterfactual remains **UNRESOLVED_UNOBSERVED_ALTERNATIVES**. Pools with no feasible alternative provide a direct lower bound on feasible-set limitations.

Every trajectory passes the accepted-transition telescoping identity. Validation was replayed only after training freeze and never affected selection or write-back. Test access and counterfactual alternative replay were both zero.

## Provenance note

The top-level execution summary inherited an old non-authoritative runtime label from a reporting constant. The frozen registry and both per-trajectory summaries carry the correct runtime identity. The raw run was not modified; `execution_summary_normalization.json` records this reporting-only reconciliation. Scientific and model identities are unaffected.
