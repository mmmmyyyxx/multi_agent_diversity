# Qwen 3.7 Flash qualification

This report records the secret-free aggregate outcome of the 2026-08-05 model
qualification. Raw API responses, prompts, questions, caches, credentials, and
absolute paths are intentionally excluded.

## Configuration

- Model for Solver, Teacher, Critic, and Student: `qwen3.7-flash-2026-07-15`
- Provider protocol: Alibaba Cloud Model Studio OpenAI-compatible Chat Completions
- Thinking mode: explicitly disabled
- Method: `member_aware_peer_state_v10`
- Checkpoint: `19`
- Seed: `44`
- Qualification split sizes per task: 75 optimization, 50 validation identity,
  and 125 test rows
- Starting source commit: `3943a29dd41919a907bfa85379a17d56ffc863ea`
- Source state: dirty; this is qualification evidence, not a formal matched result

## Gates

The four-role transport smoke passed after the mutable-prompt boundary was
strengthened. Solver output parsed, all TCS responses passed their strict JSON
schemas, the Critic rejected an unsafe live plan, two Student candidates passed
the shared contract validator, and no response was truncated.

The frozen-prompt performance qualification failed the required Solver
eventual-valid threshold:

| Task | Test vote accuracy | First-pass valid | Eventual valid | Terminal-invalid unique requests | Gate |
|---|---:|---:|---:|---:|---|
| `disambiguation_qa` | 72.8% | 97.0% | 100.0% | 0/200 | pass |
| `geometric_shapes` | 76.0% | 90.0% | 95.5% | 9/200 | fail |
| `ruin_names` | 88.0% | 94.0% | 99.0% | 2/200 | fail |

All dataset-overlap and comparison-cache gates passed. Failures were caused by
visible Solver reasoning reaching the fixed 1800-token limit before the strict
terminal line. Two temporary interface variants and one user-message reminder
also failed qualification and were reverted.

The 8-update S5 pilot and formal matched-setting experiments were not started.
