# Post-freeze Validation50 evaluation policy

This is a separate, future Stage 0 endpoint. It is not part of adaptive Formal V3 search. Each of the six Formal V3 attempt2 search cells has zero Validation50 calls and zero Test50 calls. No evaluation occurs before all six search cells finish and their final search states and execution artifacts are frozen.

After separate explicit real-API authorization, each cell's final state hash is recorded and search and prompt updates are prohibited. The private `final_team_materialization.json` must reconstruct exactly five prompts. For Native, the first returned official GEPA candidate (the search engine's final candidate identity) is replicated to all five members; if no candidate was returned, the homogeneous initial team is retained. For Layer2, the five final committed prompts are retained in member order. This choice is fixed before any Validation50 or Test50 observation and uses no held-out data.

Evaluate exactly Validation50 from `anti_overfitting_split_v1/validation` in `experiments/anti_overfitting_split_v1/split_manifest.json` for each frozen five-member team, once. This is the independent 50-question validation split, with sorted unique question-hash-set SHA-256 `d6514ca588f124460502afbc44a5314e9ef7f141ee826411cb6f02abad736df5`. The split manifest has schema `anti_overfitting_split_manifest_v1` and normalized-LF SHA-256 `9639db6f1a83bbe54704d012edd7b843e31cbb1141f9957437daa3803a954886`. These identities are frozen in each attempt2 manifest and protocol and checked again before provider construction.

`fold_a + fold_b = Optimize100` for adaptive search. `fold_c = Shadow50` for the adaptive winner-only write-back gate. `validation = Validation50` for post-freeze generalization only. `test = Test50` remains sealed. The four question-hash sets are pairwise disjoint. Fold_c may never be represented as Validation50.

Report equal-weight plurality VoteAcc with ties incorrect, all five member accuracies, and oracle coverage. The results are written separately and cannot select a search checkpoint, alter a Formal run, or change either final team. Each evaluation needs its own governed authorization and ledger.

Test50 stays sealed with zero calls throughout Stage 0. Its later use requires a separately frozen final method or paper comparison and explicit authorization after Pattern and Memory decisions are complete.

The read-only `formal_trajectory_trace.jsonl` is derived after each cell from a completed lifecycle and a verified `execution_evidence_freeze.json` inventory. Neither this trace nor the Validation50 endpoint is imported into adaptive search.
