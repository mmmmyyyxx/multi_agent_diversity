# Current MATH measurement and admission semantics

Scope: the current MATH binding, not every benchmark. This document clarifies
existing behavior; it changes no runtime identity, scoring or admission rule.
Executable identities remain in `current_contract.py`, benchmark `protocols.py`
and the [current specification](CURRENT_SPEC.md). Historical bindings retain
their original source. [Global principles](SCIENTIFIC_PRINCIPLES.md) govern claims.

Output validity uses flexible gold-blind extraction and the pinned mathematical
domain. Correctness uses binary pinned mathematical equivalence. First valid
output stops bounded recovery even if wrong; exhausted invalid outputs remain
incorrect in the fixed denominator and have no vote. Distinguish valid-wrong
to correct, invalid to valid-wrong, invalid to correct and valid to invalid.
Only the first is direct observed mathematical repair; invalid to correct combines
presentation and task success and does not isolate reasoning improvement.

Five members use equal-weight equivalence plurality. Oracle is any-member
correctness; Vote scores the actual selected answer. Correct coverage can increase
without Vote improving. Neither textual diversity nor increased Oracle alone is
the frozen team endpoint. Fixed Peers retain their own member request lanes;
identical prompts do not collapse five responses into one. Legal exact request
reuse is confined to its frozen cache policy and attempt.

The Full competence floor is each target's actual initial binary correct count
on the entire frozen Optimize support. It is not a per-question invariant or an
independent estimate of general competence. Current Full admission requires:

```text
candidate_target >= initial_target
candidate_Vote >= incumbent_Vote
candidate_target > incumbent_target OR candidate_Vote > incumbent_Vote
```

Other members are fixed. Target decline above the initial floor can pass with
positive team progress; target-only improvement can pass with unchanged Vote.
Finite collateral losses are allowed subject to the bound Probe, Full and Shadow
contracts. Under current MATH invalid-as-incorrect semantics, invalidity is
observed and affects correctness/vote eligibility; invalid delta is not a
separate zero-regression Full or Shadow veto. Other benchmark policies can differ.

Local Mutation and SearchValidation are Optimize search observations. Current
V2.5 advances every safe, contract-legal, nonduplicate evaluated edit even with
neutral or negative local effects. Assigned 0-to-1 repair and independent Probe
risk checks control Full promotion. Full remains Optimize evidence. Only its
selected winner reaches Shadow, whose current gate requires Vote nonregression
and target loss no worse than two on its frozen support. Shadow pass can therefore
coexist with no positive target replication. Commit follows all gates atomically.

`team_epoch_no_commit_v1` observes complete eligible-member epochs at the same
committed state; a commit resets its patience. Consecutive no-commit opportunities
and consecutive no-deployed-Vote-improvement opportunities are separate diagnostic
series, with no stopping read point. A target-only commit resets the first while
extending the second. Neither is a replacement for the epoch stopper.

Models, initial Arm, seeds, memberships, budgets, generation limits and operational
caps belong to the frozen experiment. Development Shadow is adaptive, and
Optimize/Shadow gains provide no independent Validation/Test generalization.
