# Seed78 Primary-Responsibility Scheduler A/B Protocol

## Status and scope

This is a single-seed prospective mechanism pilot. Seed `78` is the runtime
random seed; it reuses the previously frozen Seed75 cross-fit data identity:

- Optimize100: fold A plus fold B;
- Shadow50: fold C;
- Validation50: the external validation split;
- Test50: prohibited, zero calls.

The experiment is not a formal multi-seed efficacy result. It isolates one
Layer-2 target-allocation change and may not be pooled into prior Seed75/76/77
evidence as though it were preregistered replication.

## Arms and only intervention

- `A_VOTE_ALIGNED_HIERARCHICAL`: the frozen hierarchical scheduler
  `direct_flip > near_margin > pure_coverage`, with deterministic round-robin
  fill when fewer than two scored members are available.
- `B_PRIMARY_RESPONSIBILITY_REALIZABILITY`: select exactly two distinct members
  by descending
  `max(4 * D_i, 2 * N_i, C_i) / (1 + f_i)`, with deterministic round-robin
  zero-score fill. `f_i` is the persistent count of completed target attempts
  since that member's last successful write-back.

The target scheduler is the only arm difference. Both arms always run two
target branches so target policy is not confounded with branch compute.

## Shared execution contract

- five equal-weight members, plurality vote, tie-as-abstain;
- Solver `qwen3-8b`, thinking disabled, `COMMON_SOLVER_CONTRACT_V1`;
- reflection/optimizer model `qwen3.7-flash`;
- the same initial P0 prompt team and initial Solver profiles;
- the same official frozen GEPA backend and configuration;
- fresh local GEPA state per target branch;
- local metric budget 36 per target, sufficient for parent plus two independent
  mutation evaluations over TeamMiniBatch12;
- TeamMiniBatch12: responsibility 4, coalition 4, preservation 4, with
  deterministic fill when a group is smaller;
- at most two full-Optimize promotions per target branch;
- unchanged Common-Safe selection, ranking and max-one write-back;
- winner-only Shadow gate;
- at most 32 opportunities per arm;
- early stop after six consecutive opportunities without a Shadow-approved
  commit;
- no Proposal Memory, M2F, semantic-Critic change, GEPA change, or method tuning.

The two arms start from identical frozen profiles. A completes before B, but B
receives a copy of the initial cache made before A begins; candidate-search
cache entries never cross arms.

## Data roles and timing

Shadow50 is optimization-time safety data. It is used only for winner-only
write-back gating and is not an unbiased generalization endpoint.

Validation50 is evaluated exactly once per final arm, only after both A and B
training trajectories have terminated and their final teams are frozen. It is
never used for proposal generation, target allocation, ranking, retry,
write-back or early stopping.

Test50 remains inaccessible and has zero calls.

## Frozen observations and decision

Mechanism observations are target counts by member, commit counts by member,
target-to-commit rates, target/commit distribution mismatch, maximum unresolved
target streak, calls, tokens, and tokens per commit. Final descriptive outcomes
are Validation50 Vote, MeanMember, Vote-minus-MeanMember, Oracle, and the full
`G=0..5` histogram.

The frozen labels are defined in `classifier_definition.json`. No parameter or
protocol may be adjusted after observing results. Any protocol, persistence,
source, model, split, timing, isolation or Test50 violation yields `HOLD` with
no automatic retry.

Protocol document SHA256 (canonical protocol payload):
`854879a50c08817e1d1a074b8a9320c034ebd363c80940f2fd73a2b7b3cc9d2d`.
