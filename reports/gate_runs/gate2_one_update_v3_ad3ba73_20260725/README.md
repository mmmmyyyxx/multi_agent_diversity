# Gate 2 Public Summary

- Source commit: `ad3ba73a89cc579bae906c570a677c72600d955c`
- Local source artifact: `runs_gate2_one_update_v3_ad3ba73_20260725`
- Method: `member_aware_peer_state_v3`
- Setting: `shared_member_aware_full`
- Task: `disambiguation_qa`
- Seed: `42`
- Epochs / update interval: `1 / 75`
- Split sizes: `75 / 50 / 125`
- Solver max tokens: `1800`

## Solver

- Unique requests: `250`
- First-pass valid: `249`
- Recovered invalid: `1`
- Terminal invalid: `0`
- Eventual valid rate: `1.0`

## Update Funnel

- Eligible targets: agents `0..4`
- Selected target: agent `2`
- Teacher calls: `2`
- Critic calls: `2`
- Critic approvals: `0`
- Critic semantic rejections: `2`
- Student calls: `0`
- Stage A evaluated: `0`
- Stage B evaluated: `0`
- Feasible / accepted candidates: `0 / 0`

Terminal failure: `critic_semantic_rejection_exhausted`.

Critic rejection classes were `evidence_mismatch` followed by
`actionable_specificity`. This is an operational TCS failure, not a candidate
efficacy result. Raw responses, SQLite cache, and full LLM logs remain local.
