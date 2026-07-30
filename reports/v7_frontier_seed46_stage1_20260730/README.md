# Seed46 frontier responsibility Stage-1 mechanism pilots

This directory is a sanitized, compact record of the three valid 8-update
mechanism pilots run under the frozen-initialization frontier protocol.

## Scope

All three treatments used the same frozen initial train state, five identical
initial prompts, the same Seed46 split and Solver request identity, independent
mutable Solver caches, 75 training examples, two candidates per update, Stage B
budget two, proposal memory off, no validation selection, and no test
evaluation. Each final active state follows exactly eight updates.

| Treatment | Responsibility and scheduler | Catch-up | Valid result |
| --- | --- | --- | --- |
| A: v6-owner | one deterministic legal owner per residual; owner portfolio scheduler | off | pass |
| B: v7-frontier-core | multi-member repair frontier; repair value, uplift deficit, age | off | pass |
| C: v7-full | same as B | fallback_v1 | pass |

## Mechanism conclusion

All hard lifecycle, isolation, assignment-legality, and max-wait checks passed
for the three valid runs. B and C exercised genuinely overlapping frontiers;
A preserved a unique owner per residual. C never entered its catch-up lane in
this short trajectory. It therefore established no borrowed-residual failure,
but also supplies no evidence of catch-up efficacy.

These are development mechanism checks only. They do not compare test
performance or establish efficacy/generalization. The first interrupted C
launch and other incomplete launch attempts are excluded rather than resumed
or compared.

`stage1_summary.json` contains the machine-readable sanitized facts used for
this conclusion. Raw prompts, questions, gold labels, model responses, SQLite
caches, checkpoints, endpoint details, and local paths remain untracked.
