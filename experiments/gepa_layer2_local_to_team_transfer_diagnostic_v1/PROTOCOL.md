# Online local-to-team transfer diagnostic v1 — preregistration

Status: `READY_FOR_AUTHORIZATION`, not authorized and not executed. This is a
new prospective diagnostic, not a continuation or retry of either post-refactor
canary. The old `PROTOCOL_DRAFT.md` is retained as design history; this file is
the frozen scientific protocol. Preparation, fake-provider checks, and cost
arithmetic use zero real API calls. No Validation50 or Test50 access is allowed.

## Question and interpretation limit

In one fresh online GEPA–Layer2 trajectory, what happens to strictly accepted
local GEPA mutations when placed in the current five-member team? Does
TeamMiniBatch filter a candidate whose **read-only** Full Optimize100 result is
positive? This is a mechanism diagnostic over at most five adaptive mutations,
not an IID transfer-rate estimate or method-efficacy trial. No confidence
interval or population claim is permitted.

## Frozen identity and state sequence

- Seed 81; `FRESH_DETERMINISTIC_INITIALIZATION_V1`.
- BBH `disambiguation_qa`; Optimize100 = frozen fold A+B, Shadow50 = fold C.
  Shadow50 is optimization-time safety data. Validation50 and Test50: 0 calls.
- Solver `qwen3-8b`, thinking false, `COMMON_SOLVER_CONTRACT_V1`; reflection
  `qwen3.7-flash`; provider profile `lwj`; exact endpoint fingerprint and data
  hashes are bound in the private preparation bundle, never published raw.
- Official pinned GEPA v0.1.1 and its frozen source hash; local metric-call
  budget 36, local validation 12, reflection minibatch 3, unchanged supported
  search configuration. No GEPA search-core edits.
- Exactly one production Layer2 Top-1 target per opportunity, using the
  unchanged primary-responsibility/persistent-realizability scheduler. The
  assignment, evidence packet, and candidate replacement use the **actual
  current committed parent team** at opportunity start. A successor parent
  exists only after a genuine ordinary commit. No hand-picked parent or
  candidate, no parent restoration between opportunities.

The Layer1 accepted-event count must equal the valid unique local candidates
returned across the backend boundary. With the frozen 36-call/12-val budget,
official GEPA can accept at most one child per opportunity. If an internal
accepted child is absent from the returned frontier, or an unexpected extra
candidate appears, abort with evidence: never silently change the sampling
denominator. The sample is every accepted mutation encountered before the
stopping condition, not only winners or commits.

## Stop rules and fail-closed ceilings

Stop at the first completed opportunity after any of:

1. five locally accepted mutations (`TARGET_REACHED`);
2. ten Layer2 opportunities;
3. twenty reflection proposals, or before the next complete 36-call local
   opportunity if its frozen maximum of four proposals could exceed twenty.

Fewer than five is `TARGET_NOT_REACHED`; no extra seed, parent, opportunity,
retry, or extension may be added post hoc. Global ceilings are 1,200 successful
provider calls and 4,800 physical attempts across Solver and Reflection. These
are emergency safeguards, not spending targets. A ceiling encountered during
an incomplete evaluation fails closed with a preserved incomplete attempt;
partial candidate measurements are never claimed as complete Full results.
There is no experiment-level auto-resume or retry. Frozen request-level
transport retry semantics remain unchanged.

## Mandatory dual observation and online isolation

For each local accept, run the unchanged TeamMiniBatch12 (`4/4/4`, unchanged
promotion and max-two rule). In parallel in *decision semantics*, measure Full
Optimize100 for **every** accepted candidate. For a MiniBatch-promoted candidate,
reuse its ordinary Full evaluation. For a non-promoted candidate, execute a
separate `diagnostic_full_eval` stage, with its Full result held only in
diagnostic metadata. The diagnostic result is never attached to a candidate
record offered to the selector. It cannot promote the candidate, change
Common-Safe eligibility, trigger Shadow, or cause a commit. The immutable
online selector sees only normally promoted records. Only the ordinary
winner may reach unchanged Shadow and atomic write-back.

Read-only Common-Safe computed from diagnostic Full is labelled
`DIAGNOSTIC_COMMON_SAFE`, distinct from ordinary Common-Safe. A MiniBatch fail
with positive Full Vote is a descriptive false negative, **not** an override.
Diagnostic and ordinary evaluations share only the pre-existing exact-request
cache; no diagnostic score or gate result is fed into GEPA, scheduler, or
subsequent target allocation.

## Required evidence and labels

Each accepted candidate row records: parent team hash, candidate hash/ID,
target/lane, pre-candidate target pivotal count `P_i`, total pivotality,
single-member vote-changeable Optimize100 case count, local parent/child
Full-validation scores and their delta, the distinct strict-acceptance
minibatch delta, local newly fixed/broken and preservation loss, MiniBatch
Vote/target/Oracle and promotion, Full target/Vote/Oracle, correct-Vote
gains/losses, pivotal gains/losses, ordinary Common-Safe/Shadow/commit status,
stage costs, and sanitized failure reasons. No prompt, question, answer,
model output, credential, endpoint, or absolute path enters a tracked report.

`single-member vote-changeable` means that with the same four peer votes,
the case's Vote correctness differs between at least two legal option votes
or abstention by the target. `P_i` counts parent-correct target votes whose
removal makes a correct parent Vote wrong. Both are calculated before candidate
generation. For Full Vote delta zero, use `STRUCTURALLY_VOTE_CENSORED` when
changeable count is zero, else `RESPONSIVE_BUT_VOTE_NEUTRAL`. Additional
possibly overlapping descriptors are `LOCAL_POSITIVE_COVERAGE_GAIN_ONLY`,
`LOCAL_POSITIVE_TEAM_NEGATIVE`, `LOCAL_POSITIVE_PASSES_MINIBATCH`, and
`LOCAL_POSITIVE_BREAKS_PEER_SUPPORT` when directly supported by case counts.
No label affects online selection.

## Zero-API cost audit

The frozen local budget arithmetic is `12` seed metric calls, `6` per
proposal, and `12` extra for an accepted child's Full local evaluation. Thus
at most one accepted child or four rejected proposals per opportunity;
ten opportunities contribute at most `360` local Solver example calls and at
most `20` Reflection proposals under the global stop rule. Initialization is
`5×100 = 500` Solver rows. At most five accepted mutations add `5×12 = 60`
MiniBatch logical rows and `5×100 = 500` Full logical rows; exact-request
reuse makes their joint unique candidate-profile upper bound `500` successful
Solver calls. Ordinary winner-only Shadow can logically reach up to
`5×(5+1)×50 = 1,500` rows without assuming cross-parent cache reuse.

Therefore the unconstrained envelope is at most `500+360+20+500+1500 =
2,880` successful provider calls. The hard global ceiling truncates this to
1,200; **five complete candidates are not guaranteed within that ceiling**.
The published outcome must distinguish `TARGET_NOT_REACHED` from an emergency
incomplete/aborted run. Physical attempts are separately capped at 4,800;
this is not a claim that every role uses exactly four request-level retries.
Provider-reported input/output tokens and cache hits are recorded, not inferred
from these row counts.

## Governance

The source commit, normalized source hashes, protocol SHA, private split
hashes, provider endpoint fingerprint, model roles, attempt ID and run identity
are fixed by `scripts/prepare_online_transfer_diagnostic.py` in a fresh ignored
prep root. The unified `scripts/run_experiment.py` is the only real execution
entrypoint and must fail closed before provider construction on any mismatch.
The preparation bundle starts `authorized=false`, with phase `diagnostic` and
roles `solver, reflection`; a later **separate, attempt-specific** user grant
is required to run. Fresh formal run root only; no historical artifact may be
resumed or overwritten. Existing canary reports remain immutable.
