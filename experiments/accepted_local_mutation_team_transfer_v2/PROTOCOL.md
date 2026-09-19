# Accepted-Mutation Team-Transfer Replay v2

## Amendment and frozen scientific state

This zero-API amendment supersedes the unexecuted v1 protocol because v1 made
Full evaluation conditional on TeamMiniBatch12. That censoring mixed the
primary local-to-team transfer question with the quality of a progressive
filter. The v1 protocol, manifest, preparation report, and private bundle are
immutable historical evidence and remain unexecuted.

The state entering v2 remains:

```text
LOCAL_EMPIRICAL_PATH_CONFIRMED
LOCAL_STRICT_IMPROVEMENT_CONFIRMED_WITHIN_SINGLE_STATE
TEAM_TRANSFER_NOT_EVALUATED
```

The five accepted mutations, common baseline state, candidate hashes, local
telemetry, model, seed, splits, TeamMiniBatch composition and threshold,
Common-Safe rule, Shadow rule, and solver contract are unchanged.

## Primary question and five independent pairs

Given a prompt mutation already confirmed as a strict local GEPA improvement,
does replacing only its target member improve the fixed team's performance?

Every one of the five frozen accepted mutations is included, without ranking,
selection, editing, regeneration, or new GEPA search. For mutation `j` targeting
member `i`:

```text
control:   T0 = (P0, ..., Pi,  ..., P4)
treatment: Tj = (P0, ..., P'i, ..., P4)
```

All five treatments start from the same baseline team hash. They are replayed
independently; no treatment changes the state seen by another treatment.

## Mandatory Full and diagnostic stages

Every candidate runs both TeamMiniBatch12 and Full Optimize100:

```text
                         +-> TeamMiniBatch12 diagnostic simulated gate
five frozen candidates -|
                         +-> mandatory Full Optimize100 (5/5)
                                           |
                                           +-> Common-Safe / Shadow diagnostics
```

TeamMiniBatch uses the unchanged 4 responsibility + 4 coalition + 4
preservation rows and unchanged promotion rule. Its label in this experiment is
`TEAM_MINIBATCH_DIAGNOSTIC_GATE_V1`. Pass or fail cannot suppress, alter, or
condition Full evaluation.

Full evaluation is mandatory for all five candidates. A scientifically
evaluable completion requires `full_evaluated_candidates == 5`; otherwise the
run aborts or is `TEAM_TRANSFER_REPLAY_NOT_EVALUABLE`. Full execution cannot
depend on MiniBatch, Common-Safe, or Shadow outcomes.

Common-Safe is computed after Full with its existing fixed-peer constraints.
Shadow retains its existing eligibility rule and is reached only after a
Common-Safe pass. These are secondary admissibility diagnostics and never
change the primary Full result. No candidate is committed.

## Primary estimand and decomposition

For every candidate:

```text
FullTeamVoteDelta_j = Vote(Tj) - Vote(T0)
```

Its Full classification depends only on that delta:

```text
delta > 0  -> TEAM_POSITIVE
delta = 0  -> TEAM_EQUAL
delta < 0  -> TEAM_NEGATIVE
```

The five-row transfer table joins local delta, local newly fixed/broken,
preservation loss, target-member Full delta, Full team vote delta, oracle
coverage delta, correct-vote gains/losses, hashed newly team-correct and newly
team-wrong cases, pivotal gains/losses, and collateral losses. Non-target
member behavior delta is zero by construction.

The primary summary is `team_positive_count / 5`, with equal and negative
counts reported separately. The five mutations are not IID samples; no
binomial, Wilson, population, or cross-state inference is allowed.

## TeamMiniBatch gate audit

Mandatory Full results permit a frozen descriptive 2x2 audit:

```text
MiniBatch pass + Full positive      -> correct_promotion
MiniBatch fail + Full positive      -> false_negative_filtering
MiniBatch pass + Full non-positive  -> false_positive_promotion
MiniBatch fail + Full non-positive  -> correct_filtering
```

Only raw counts are reported. TeamMiniBatch is not tuned after observation.

## Interpretation states

Interpretation states are descriptive and may coexist:

```text
at least 2/5 TEAM_POSITIVE
  -> LOCAL_IMPROVEMENT_CAN_TRANSFER_TO_TEAM_GAIN_WITHIN_THIS_STATE

at least 3/5 TEAM_EQUAL
  -> LOCAL_IMPROVEMENT_OFTEN_FAILS_TO_CHANGE_TEAM_OUTCOME

at least 2/5 TEAM_NEGATIVE
  -> LOCAL_TEAM_OBJECTIVE_MISALIGNMENT_OBSERVED

at least two of TEAM_POSITIVE / TEAM_EQUAL / TEAM_NEGATIVE are present
  -> MEMBER_SPECIFIC_TRANSFER_HETEROGENEITY
```

If fewer than five Full results exist, the sole state is
`TEAM_TRANSFER_REPLAY_NOT_EVALUABLE`.

## Isolation and caching

Optimize100 contains TeamMiniBatch and Full. The run-scoped exact-request cache
is unchanged: 60 MiniBatch logical calls populate the cache, the same 60 calls
inside Full are hits, and the remaining 440 Full calls reach the provider. The
frozen Phase-A baseline Optimize evidence is reused and is not resampled.

If Shadow is reached, one 50-row control profile is shared across all eligible
candidates through the existing run-scoped evidence, followed by one 50-row
candidate profile per eligible candidate. Baseline provider calls, cache hits,
and candidate calls are reported by stage.

Validation50 and Test50 are inaccessible and have call ceilings of zero.
Scheduler selection, target allocation, candidate competition, persistent
realizability, GEPA, Reflection, and team write-back are disabled.

## Frozen cost envelope

```text
mandatory logical Solver rows:
  TeamMiniBatch: 5 x 12  = 60
  Full:          5 x 100 = 500
  total                    560

mandatory unique successful provider calls after exact-cache reuse:
  Optimize candidate profiles = 500

optional Shadow successful provider calls:
  shared control = 0 or 50
  candidates     = 0 to 250

normal mandatory provider floor = 500
hard successful-provider ceiling = 800
hard transport-attempt ceiling = 3200
hard completion-token ceiling = 800 x 1800 = 1,440,000
```

Prompt-token and currency estimates are not frozen because no repository
tokenizer or stable provider price is part of this experiment identity. Actual
provider usage is recorded if a later authorized execution occurs.

The run stops after five Full results plus eligible secondary diagnostics, or
aborts on any provider, budget, identity, split, lifecycle, or integrity
failure. The lifecycle writer must terminate at `EXECUTION_COMPLETE` or
`EXECUTION_ABORTED`.

Preparation and preflight make zero API calls. Execution requires a new clean
source/handoff freeze and separate explicit authorization for this exact v2.
