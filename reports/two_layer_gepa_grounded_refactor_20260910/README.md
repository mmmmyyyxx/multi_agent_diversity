
# Two-layer GEPA-grounded refactor

The zero-API refactor introduces a replaceable Local Prompt Optimizer and a
backend-agnostic Responsibility-Guided Team Search controller. Responsibilities
formerly co-located in `system.py` are represented by Layer 1 proposal search
and Layer 2 responsibility/task construction/team evaluation/selection/commit.
The canonical v15 runtime remains unchanged.

GEPA is genuine, not GEPA-like: the implementation invokes official frozen
GEPA `v0.1.1` at `b4dbb55b7601dac448cdb836d5a401ca7d9eb920`, uses its persistent candidate
population, instance Pareto parent selection, reflection mutation, minibatches,
strict improvement, callbacks, result state, and parent lineage. It returns up
to four local-frontier candidates; the Team Controller is unaware of GEPA
internals and chooses the team winner separately.

The same Team Controller accepts Stub, Legacy TCS, or GEPA without source
changes. Local optimizers import neither responsibility scheduling nor voting,
Shadow, or the training system. Local single-member evaluation and team
fixed-peer evaluation have distinct interfaces, ledger phases, and costs.

Remaining transitional debt is the production binding from canonical runtime
objects into the new facade. It is intentionally not performed here so this
architecture-only task cannot alter v15 scientific behavior. The historical
experimental RG-GEPA files remain untouched. The opt-in architecture satisfies
`GEPA_GROUNDED=True`; canonical promotion is not implied.

The next minimal API pilot is a frozen paired local-backend comparison on the
same parents, targets, responsibility assignments, TeamMiniBatch12, full-team
policy, Common-Safe selector, solver contract, and split: Legacy TCS versus
official GEPA. No memory or objective change should be included.

API calls: 0. Test50 calls: 0. Focused tests: 168 passed. Full
tests: 977 passed with 1 unrelated known historical
artifact failure(s).
