# Gate 2 Rerun Public Summary

- Source commit: `5728037a91b5b23d4137ca6089e737b72206d0eb`
- Local source artifact: `runs_gate2_one_update_v3_5728037_20260725_141328`
- Method: `member_aware_peer_state_v3`
- TCS protocol: `aggregated_small_model_tcs_v3`
- Teacher revision protocol: `critic_grounded_full_plan_revision_v1`
- Setting/task/seed: `shared_member_aware_full` / `disambiguation_qa` / `42`
- Split sizes: `75 / 50 / 125`
- Gate decision: `PASS`

The pipeline completed one accepted update: one Critic approval, one Student
call, two valid candidates, two Stage A evaluations, two Stage B evaluations,
and two hard-constraint-feasible candidates. The live Critic approved the
first Teacher plan, so this run did not trigger a second-round Teacher revision.

Solver recovery had 575 unique requests, one recovered `length` failure, zero
terminal invalid requests, and a 100% eventual valid rate. Raw API responses,
SQLite cache, full LLM logs, and prompt payloads remain local.
