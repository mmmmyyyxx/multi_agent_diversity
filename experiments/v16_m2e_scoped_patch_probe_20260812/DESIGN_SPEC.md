# M2E Scoped Behavioral Patch Fixed-Parent Probe

This noncanonical diagnostic reuses the exact eight parent/target identities
from the gate-PASS Generic-vs-M20 Study B. Both arms are generated fresh under
one new execution source freeze. Odd cases run M20 then M2E; even cases reverse
the order. Each cell requests two candidates using qwen3-14b with thinking
disabled.

M20 is byte-current v15. M2E receives the identical SingleLane responsibility
context but changes only Teacher/Student output representation. Teacher returns
`trigger_condition` and `localized_behavior`; Student returns two scoped-patch
objects. Program code deterministically constructs each candidate as the exact
parent prompt bytes followed by one versioned conditional-refinement template.

Module1, target, parent profiles, responsibility hashes, retry ceilings,
Critic hard-blocker semantics, candidate budget, common evaluator, Stage A/B,
Common-Safe geometry, solver namespace, and no-commit isolation are shared.
Validation and test access are prohibited.

Every M2E candidate must have a byte-identical parent prefix, nonempty trigger
and behavior hashes, and no unconditional trigger markers. A mechanism,
infrastructure, source-identity, budget, mutation, commit, validation, or test
violation makes the protocol gate fail and blocks result interpretation.

The frozen descriptive decision is:

- targeting is retained when total M2E responsibility repair is at least 80%
  of fresh M20 responsibility repair;
- collateral is reduced when total M2E non-responsibility loss is lower and
  M2E wins more paired cases than it loses on that count;
- `SUPPORTED` requires both;
- otherwise report `TARGETING_LOST` or `COLLATERAL_NOT_REDUCED`.

Responsibility repair, non-responsibility loss, stable loss, pivotal loss,
target regression, F geometry, and common-safe feasibility remain separately
reported. No scalar is introduced into runtime selection.
