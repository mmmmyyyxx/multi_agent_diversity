# Gate 3 Public Summary

- Source commit: `e6b2785008db0f602ddfd6b0affc1ea6861a0bde`
- Local source artifact: `runs_gate3_eight_update_v3_e6b2785_20260725_153657`
- Method: `member_aware_peer_state_v3`
- TCS protocol: `aggregated_small_model_tcs_v3`
- Teacher revision: `critic_grounded_full_plan_revision_v1`
- Setting/task/seed: `shared_member_aware_full` / `disambiguation_qa` / `42`
- Configuration: 8 updates, Solver concurrency 8
- Gate 3 research decision: `STRONG PASS`
- Operational scaling decision: `HOLD FOR CONCURRENCY CALIBRATION`

Six of eight updates reached Stage A/B and were accepted. All five members
improved on optimization, validation, and test. Validation selected epoch 8;
test vote increased from 52 to 71.

The live Teacher revision branch was exercised four times and led to round-2
approval twice. Solver eventual validity was 99.9518% with one terminal invalid
on a non-selected candidate.

There were zero terminal `429` failures, but 522 Solver rate-limit retry
attempts over 2,075 unique requests (25.2%). Raw responses, SQLite cache, full
LLM logs, and prompt bodies remain local.
