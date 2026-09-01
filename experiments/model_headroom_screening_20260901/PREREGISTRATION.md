# Task Model Headroom Screening Preregistration

## Question

Which of two task-agent backbones has stable baseline competence and sufficient
prompt-optimization headroom under the unchanged canonical Generic evolution
protocol?

## Frozen inventory

- Task: BBH `disambiguation_qa`
- Models: `qwen2.5-7b-instruct`, `qwen3-8b`
- Seeds: 62, 63, 64
- Arms: `shared_static_reference`, `shared_generic_evolution`
- Agents: 5, equal-weight plurality, tie abstains
- Train/validation: 75/50
- Test evaluation: prohibited
- Initial prompt: canonical shared-identical prompt
- Generic budget: 32 updates, one target branch, two candidates, Stage B budget 2
- Optimizer/Teacher/Critic models: `qwen3-14b` for both task backbones
- Proposal Memory: off
- Final active state: automatic; no validation selection
- Backbone execution order is counterbalanced by seed: 62 uses A then B, 63
  uses B then A, and 64 uses A then B. Within every backbone/seed run Static
  precedes Generic because Static supplies the frozen comparison initialization.

No Full, Module1, M20, M2F, selector modification, Common-Safe modification,
data change, extra seed, or test evaluation is permitted.

## Validation protocol

Training is completed with validation and test evaluation counts equal to zero.
The final checkpoint is then evaluated exactly once on validation in an external
read-only evaluator. The evaluation cannot mutate the checkpoint, prompt team,
training directory, or selection state. Validation results cannot affect any
training decision.

## Metrics

For each model and seed:

- Static/Generic validation VoteAcc and OracleAcc;
- Generic minus Static VoteAcc;
- Static and Generic Oracle-minus-Vote gaps;
- per-member validation accuracies;
- train plus validation terminal-invalid/output-failure rates;
- Generic accepted-update count.

## Frozen stability rule

Serious parsing/output instability is true if any infrastructure failure occurs
or the aggregate terminal-invalid response rate exceeds 1%. First-attempt
invalidity is reported but does not itself fail the rule when bounded recovery
produces a valid terminal response.

## Frozen model-selection rule

A model passes only when all are true:

```text
mean Static VoteAcc <= 0.65
mean Generic-Static VoteDelta >= 0.04
mean Generic Oracle-Vote gap >= 0.08
Generic > Static on at least 2 of 3 seeds
no serious parsing/output instability
```

If exactly one model passes, select it. If both pass, choose larger mean uplift;
values within 0.01 are close, then choose the larger Generic Oracle-Vote gap;
if those are also within 0.01, choose the lower Static VoteAcc. An exact tie
after all frozen criteria yields HOLD. If neither passes, yield HOLD. Efficacy
cannot stop execution or alter the seed inventory.

```text
FULL_METHOD_NOT_RUN=true
TEST_ACCESSED=false
```
