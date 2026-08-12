# M2E Scoped Behavioral Patch Probe

This report contains sanitized, analysis-ready evidence from the preregistered
fixed-parent comparison of byte-current M20 and M2E scoped behavioral patches.
The execution used the same eight frozen parent/target cases, two requested
candidates per cell, a shared M20 evaluator, and no prompt-team commits.

The protocol gate passed all 16 cells. It found no source, case-identity,
budget, parent-state, optimizer-state, Module1, common-safe, or scoped-patch
mechanism mismatch, and no validation or test calls.

The frozen classifier result is `TARGETING_LOST`. M2E reduced total measured
collateral loss (96 versus 129; paired 6 wins and 2 losses), but retained only
14 of M20's 42 responsibility repairs, below the frozen 80% threshold. M2E
produced 3 feasible candidates versus 12 for M20. Two M2E branches ended in
Critic semantic rejection; these are included in the preregistered branch-level
comparison and are not infrastructure failures.

The collateral decrease must not be read alone as a successful mechanism:
M2E evaluated only 7 candidates versus M20's 16 because the constrained
generation pipeline often did not reach valid candidates. The defensible
conclusion is that this scoped representation reduced collateral exposure but
lost too much targeting signal. It is not a promotion candidate.

Runtime caches, prompts, questions, answers, model outputs, endpoints, and call
logs are intentionally excluded.
