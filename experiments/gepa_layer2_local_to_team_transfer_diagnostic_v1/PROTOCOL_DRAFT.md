# Local-to-team transfer diagnostic — draft, not preregistered

Status: `DRAFT_ONLY`; `READY_TO_RUN=false`; API / Validation50 / Test50 calls from this draft: `0 / 0 / 0`. This document is not an executable handoff or an amendment to either completed post-refactor canary. It does not authorize an API run, a new seed, a retry, a method change, or a changed GEPA/TeamMiniBatch/Common-Safe/Shadow threshold.

## Question and sampling unit

Among *locally accepted, strictly positive* real-GEPA mutations from a prospectively frozen set of Layer-2 parent states, what happens when each replaces only its target member while peers remain fixed? The unit is the accepted mutation, not a reflection attempt or a team commit. All accepted mutations encountered under the frozen search schedule enter the diagnostic; no post-hoc candidate selection.

Proposed stopping form: collect an exact preregistered number of locally accepted mutations **or** hit exact preregistered proposal, opportunity and provider ceilings, whichever occurs first. The suggested range of 5–10 is not a frozen value. Failed-to-reach-target sample sizes must be reported, not topped up by ad hoc parents or retries.

## Candidate-level record

For each locally accepted mutation, record only hashed identifiers and sanitized counts in publishable artifacts:

```text
parent team/state hash; candidate hash; target member; responsibility lane
local parent score; candidate score; strict delta
local newly fixed; local newly broken; preservation losses
TeamMiniBatch12: vote gain/loss/net, target-member delta,
  Oracle/coverage delta, coalition/support changes, gate result and reason
Full Optimize100: stage status; target-member, Vote and Oracle deltas;
  new/lost coverage, correct-vote and pivotal case counts
Common-Safe; Shadow; commit: PASS / FAIL / NOT_REACHED and frozen reasons
provider calls, tokens, cache hits and failed attempts by stage
```

The ordinary pipeline remains unchanged. A diagnostic-only Full replay of every locally accepted candidate, including TeamMiniBatch failures, is **recommended but not yet authorized or frozen**: the earlier five-candidate replay showed that pass-conditioned Full data cannot identify false-negative filtering. If the later protocol instead measures Full only after MiniBatch pass, Full transfer among rejected candidates must be labelled unobserved rather than inferred from MiniBatch.

## Analysis labels, not new gates

Use stage-specific, potentially overlapping descriptors; do not force one mutually exclusive label or use them for online acceptance:

- `LOCAL_POSITIVE_BUT_TEAM_NEUTRAL`: measured team Vote delta is zero on the named stage.
- `LOCAL_POSITIVE_TEAM_NEGATIVE`: measured team Vote delta is negative on the named stage.
- `LOCAL_POSITIVE_COVERAGE_GAIN_ONLY`: Oracle coverage increases while team Vote delta is zero on the same measured cases.
- `LOCAL_POSITIVE_BREAKS_PEER_SUPPORT`: candidate loses correct target-member votes on frozen cases where peer votes provided useful coalition support; report case counts and whether any team Vote loss followed.
- `LOCAL_POSITIVE_PASSES_MINIBATCH`: unchanged frozen TeamMiniBatch promotion rule passes.

Separate newly fixed from newly broken cases and distinguish `Oracle+ / Vote0` from `Vote-`. These are descriptive mechanisms, not evidence that the gate is too strict or that the local objective is systematically misaligned. A small adaptive sample is not IID and cannot support population-rate confidence intervals.

## Required freeze before any execution

Freeze exact parent/state acquisition and order; seed and split hashes; models, provider and solver contract; GEPA version and budget; local metric-call budget; TeamMiniBatch/Full/Common-Safe/Shadow semantics; the integer accepted-mutation target; proposal/opportunity/provider ceilings; cache and request retry rules; candidate-level persistence; diagnostic Full policy; Validation/Test access; source commit, preregistration, run identity and fresh root. Run a zero-API deterministic path and governance preflight. Only then request separate, attempt-specific API authorization. No API or additional candidate-generation run may start from this draft.
