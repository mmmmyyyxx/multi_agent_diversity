# Seed77 common-contract end-to-end P1 protocol

This prospective single-seed mechanism diagnostic reruns only Seed77 Diversity
P1. The optimization split, Shadow split, initial P0 decision procedure,
vote-aligned target scheduler, semantic Critic, Common-Safe write-back,
candidate budgets, and early-stop semantics are unchanged from the frozen
Seed76/77 confirmatory experiment.

The sole intended intervention is that every Solver call made during P0
initialization, Optimize rollout, Shadow gating, candidate evaluation, and
post-freeze ExternalValidation50 evaluation uses
`COMMON_SOLVER_CONTRACT_V1`. Teacher, Critic, Student, and Evaluator remain
`qwen3.7-flash`; Solver remains `qwen3-8b` with thinking disabled.

ExternalValidation50 is evaluated only after the P1 trajectory reaches a valid
terminal state. P0 and final P1 are each evaluated once. Test50 is prohibited.
The run neither reconstructs nor substitutes for the lost Seed75 frozen state.
