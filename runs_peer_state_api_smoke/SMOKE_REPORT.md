# Real API Connectivity Smoke

## Protocol

- Commit: `1a809b373bf079d586ee6e8858626f9f9f26af8f`
- Task: `disambiguation_qa`
- Settings: `shared_baseline`, `shared_peer_state_full`
- Seed: `42`
- Epochs/updates: `1 / 1`
- Train/validation/test: `8 / 8 / 8`
- Candidates per parent: `1`
- Stage B budget: `1`
- Solver concurrency: `1`
- Resume: disabled
- Hard budget: `200` calls and `300000` tokens per run

All three roles used `deepseek-chat` through the configured OpenAI-compatible
endpoint. The run made no permanent failed API attempts.

## Results

| Setting | Calls | Tokens | Cache hits/misses | Test valid | Vote accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared_baseline | 8 | 4,823 | 32 / 8 | 5 / 8 | 0.375 |
| shared_peer_state_full | 30 | 25,605 | 136 / 24 | 3 / 8 | 0.125 |

The cache counts confirm that five agents sharing the same prompt reuse one
prompt-question rollout.

## Verdict

This smoke did not pass the minimum acceptance criteria:

- Teacher and Critic were each called three times, but no Critic proposal was
  approved.
- Student was therefore not called.
- No candidate reached Stage A or Stage B.
- Strict solver parsing classified 3/8 baseline outputs and 5/8 full-run test
  outputs as `out_of_domain_answer`.
- A valid checkpoint was observed after the update, but the run completed and
  deleted it before the attempted interruption. Real checkpoint resume was not
  demonstrated.
- The requested settings produce only the B4
  `ResponsibilityProposalContext`; B1 and B3 runtime context isolation was not
  exercised by this smoke.

Do not use this run as evidence of method quality or as approval to start a
formal pilot. It is retained for API, cache, budget, artifact, and failure-funnel
audit.
