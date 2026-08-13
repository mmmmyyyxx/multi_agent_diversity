# V16 Module2 compute-matched efficacy experiment

This is the formal train-only efficacy comparison after the Seed52 development
pilot. Module2 semantics are frozen before outcomes from Seeds 53, 54, and 55.
Seed52 is excluded.

The three arms all use v15 Module1, Top-2 branches, two source candidates per
branch, the same initialization, train split, models, decoding, fixed-peer
rollout, Common-Safe acceptance, ranking, and eight-update horizon.

- `G-Matched`: Module1 plus generic peer-state evolution. Every valid generic
  source candidate receives one loss-blind revision slot. The revision sees
  only parent and source prompts; it never sees responsibility or actual loss
  evidence.
- `R-M20`: Module1 plus the frozen M20 responsibility-conditioned proposal,
  without a second revision.
- `R-M2F`: Module1 plus M20 and the frozen conditional candidate-specific M2F
  repair.

Compute is matched by opportunity ceilings, not by forcing divergent
trajectories to make identical actual calls. Every branch has two source slots
and at most two second-stage slots; every update has at most four source and
four second-stage candidates. G-Matched always spends its available loss-blind
second-stage slot after each valid source. R-M2F spends the corresponding slot
only when the frozen eligibility rule fires. R-M20 isolates the proposal-only
mechanism. Every run has the same hard ceiling of 8000 successful provider calls
and 3,000,000 total tokens. Reaching either ceiling invalidates the run rather
than changing ranking or acceptance.

Primary outcomes are final train `VoteAcc`, `g_min`, and `g_sum`. Primary
contrasts are paired by seed: G-Matched vs R-M20, R-M20 vs R-M2F, and G-Matched
vs R-M2F. Mechanism and cost measures are secondary. Oracle, correctness
correlation, effective ensemble size, pivotal/unique exchange, CriticalNet,
and specialization migration are analysis-only and never affect generation,
acceptance, ranking, stopping, or selection.

Validation and test evaluation are disabled. Raw prompts, examples, answers,
responses, caches, checkpoints, endpoints, credentials, and local paths remain
local and must not be published.
