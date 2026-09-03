# D0-D5 Diversity Matrix design freeze

This development experiment tests whether member-level responsibility produces useful, task-relevant complementarity in a five-member equal-weight plurality ensemble.

The six arms are frozen as follows:

| Arm | Allocation | Proposal | Scheduled budget |
|---|---|---|---:|
| D0 | none | Static | 0 updates |
| D1 | canonical Generic-S0 | canonical Generic | 32 updates, 1 target x 2 candidates |
| D2 | eligible round-robin | Generic | 32 updates, 2 targets x 2 candidates |
| D3 | W1 | Generic | 32 updates, 2 targets x 2 candidates |
| D4 | eligible round-robin | responsibility-conditioned C pipeline | 32 updates, 2 targets x 2 candidates |
| D5 | W1 | responsibility-conditioned C pipeline | 32 updates, 2 targets x 2 candidates |

D2-D5 use the same Common-Safe acceptance, ranking, max-one commit, source-candidate budget, and one loss-blind Generic revision opportunity per strict-valid source. Invalid outputs consume their scheduled opportunity; attempted and evaluable rows need not be equal.

The responsibility-conditioned C pipeline is Teacher-Clean, deterministic hard gate, Student, then fixed-peer rollout. It has no semantic Critic veto. M2F is disabled with no trigger. D1-D3 retain their canonical Generic pipeline behavior.

The Solver is `qwen3-14b`; Teacher, Student, canonical Critic, and evaluator roles use `qwen3.7-flash`; thinking is false. There are five agents, plurality ties abstain, Proposal Memory is off, and the fixed data are BBH `disambiguation_qa` Train75 and Validation50. Test is prohibited: zero rows loaded and zero calls.

Fresh seeds are selected before execution by the deterministic registry scan rule and then frozen. Every arm within a seed starts from the same frozen initial team and isolated cache clone. Validation is evaluated exactly once per frozen final state, only after all training cells finish, and is never used for selection or write-back.

The primary endpoint is final Validation Vote accuracy. The primary factorial contrast is D4-D2. Other frozen contrasts are D3-D2, D5-D3, `(D5-D3)-(D4-D2)`, D5-D1 (not compute matched), D1-D0, and D5-D0. Each three-seed contrast is classified as `CONSISTENT_POSITIVE`, `MAJORITY_POSITIVE`, `MIXED`, `NEUTRAL`, or `NEGATIVE` using the frozen deterministic rules in the implementation.

No outcome can change seeds, horizon, arm order, selector, proposal pipeline, Common-Safe, ranking, retry semantics, or analysis definitions. A protocol or infrastructure failure yields HOLD; efficacy does not alter execution.
