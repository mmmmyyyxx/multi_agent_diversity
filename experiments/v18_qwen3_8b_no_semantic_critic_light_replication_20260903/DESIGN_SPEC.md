# V18 Qwen3-8B No-Semantic-Critic Light Replication

## Purpose

This is a one-seed, two-arm model-transfer replication of the completed V18
Canonical versus No-Semantic-Critic online experiment. It asks whether the
previous front-end throughput and accepted-update transfer observations remain
visible after changing only the model allocation:

- five task Solvers: `qwen3-8b`, thinking disabled;
- Teacher and Student: `qwen3.7-flash`;
- canonical semantic Critic: `qwen3.7-flash`;
- the C arm makes no semantic-Critic API call and retains the frozen
  deterministic hard-safety gate.

This experiment does not promote either arm into the canonical method. It does
not compare `qwen3-8b` causally with `qwen3-14b`; the historical model result is
context only.

## Frozen design

- Task: BBH `disambiguation_qa`.
- Seed: `71`, selected before any new result.
- Train/validation: the existing frozen 75/50 split.
- Test: prohibited; zero test rows loaded for evaluation and zero test calls.
- Arms, in fixed order: `A_CANONICAL`, then `C_NO_SEMANTIC_CRITIC`.
- Horizon: exactly 8 updates per arm unless the pre-existing
  `no_actionable_responsibility` early stop occurs.
- Five members, equal-weight plurality, tie-as-abstain.
- Proposal memory off, two target branches, two source candidates per target,
  one loss-blind generic revision for each valid source, Stage-B budget 2.
- Same within-seed initialization, task split, target selector,
  responsibility, Student contract, fixed-peer rollout, Common-Safe,
  candidate ranking, max-one commit, and cache semantics across arms.
- Validation is replayed only after each trajectory is frozen. The initial
  state and every changed state are evaluated to attribute accepted commits.
  Validation never affects training, acceptance, ranking, or selection.

No additional seed, update, arm, retry for efficacy, or result-contingent
extension is permitted.

## Frozen analyses

The post-run work is zero API and applies the same rules to both arms.

1. Front-end funnel: Teacher plans, semantic-Critic/hard-gate decisions,
   Student reach, strict-valid candidates, feasible candidates, and commits.
2. Accepted-transition transfer: validation target/Vote/Oracle deltas, Vote
   gains/losses/net, gain persistence, loss provenance, and a telescoping
   identity from initial to final validation Vote.
3. C-arm feasible-pool audit: all feasible candidates at each accepted update,
   selected winner, train target/Vote/loss evidence, and whether any unselected
   candidate train-Pareto-dominates the winner. Unobserved alternative
   validation outcomes remain explicitly unidentifiable.

No candidate receives counterfactual validation in this experiment. The
separate 9-candidate validation-only replay remains a later decision and is not
authorized by this design.

## Frozen labels

Front-end label:

- `FRONTEND_THROUGHPUT_REPLICATED` iff C has strictly more Student reaches and
  feasible candidates than A, and at least as many commits.
- otherwise `FRONTEND_THROUGHPUT_NOT_REPLICATED`.

Transfer label:

- `NO_C_COMMITS` if C accepts no update;
- `TRANSFER_INSTABILITY_OBSERVED` if at least one C commit has validation Vote
  delta less than or equal to zero;
- otherwise `POSITIVE_TRANSFER_ONLY_OBSERVED`.

Ranking label:

- `TRAIN_SIDE_RANKING_SIGNAL_PRESENT` if at least one feasible alternative
  train-Pareto-dominates its selected winner on target gain, Vote net, and Vote
  loss;
- otherwise `RANKING_NOT_IMPLICATED_BY_AVAILABLE_TRAIN_EVIDENCE`.

The ranking-versus-feasible-set causal question remains unresolved whenever
unselected alternatives lack frozen validation observations. Absence of a
negative commit within this short horizon is reported as unexercised, not as
evidence that transfer risk disappeared.

## Stop conditions

Any source-identity, model-identity, initialization, cache, persistence,
validation-timing, test-isolation, parser/infrastructure, or frozen-budget
violation causes `HOLD` without retry. Efficacy does not change the run or add
data.

## Authorization state

Phase A implementation and preflight are zero API. Real API execution remains
disabled until the user explicitly authorizes it for this task and the tracked
manifest is amended and re-frozen before any call.
