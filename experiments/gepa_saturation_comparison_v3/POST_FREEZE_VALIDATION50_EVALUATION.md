# Post-freeze Validation50 evaluation policy

This is a separate, future Stage 0 endpoint. It is not part of adaptive Formal V3 search. Each of the six Formal V3 attempt2 search cells has zero Validation50 calls and zero Test50 calls. No evaluation occurs before all six search cells finish and their final search states and execution artifacts are frozen.

After separate explicit real-API authorization, each cell's final state hash is recorded and search and prompt updates are prohibited. The private `final_team_materialization.json` must reconstruct exactly five prompts. For Native, the first returned official GEPA candidate (the search engine's final candidate identity) is replicated to all five members; if no candidate was returned, the homogeneous initial team is retained. For Layer2, the five final committed prompts are retained in member order. This choice is fixed before any Validation50 or Test50 observation and uses no held-out data.

Evaluate exactly Validation50 `fold_c` (50 IDs) from `experiments/anti_overfitting_split_v1/fold_assignment.json` for each frozen five-member team, once. Report equal-weight plurality VoteAcc with ties incorrect, all five member accuracies, and oracle coverage. The results are written separately and cannot select a search checkpoint, alter a Formal run, or change either final team. Each evaluation needs its own governed authorization and ledger.

Test50 stays sealed with zero calls throughout Stage 0. Its later use requires a separately frozen final method or paper comparison and explicit authorization after Pattern and Memory decisions are complete.

The read-only `formal_trajectory_trace.jsonl` is derived after each cell from a completed lifecycle and a verified `execution_evidence_freeze.json` inventory. Neither this trace nor the Validation50 endpoint is imported into adaptive search.
