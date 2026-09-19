# Accepted-Mutation Team-Transfer Replay v1

## Scientific state before this experiment

The completed `local_gepa_acceptance_rate_pilot_phase_b_v2` freezes the
following claims:

```text
LOCAL_EMPIRICAL_PATH_CONFIRMED
LOCAL_STRICT_IMPROVEMENT_CONFIRMED_WITHIN_SINGLE_STATE
TEAM_TRANSFER_NOT_EVALUATED
```

The 5/10 pooled local acceptance rate is descriptive evidence from one frozen
baseline state, four member tasks, and adaptive GEPA search. It is not an IID
or cross-state acceptance-rate estimate.

## Question

Do the five local mutations accepted by official GEPA in Phase B transfer to
team-level gains when each is independently substituted into the same frozen
five-member baseline team?

This is a read-only measurement/replay experiment. It performs no GEPA search,
Reflection, scheduler choice, target allocation, candidate selection, prompt
generation, team write-back, or persistent-realizability update.

## Frozen candidates and counterfactual pairs

All five accepted Phase-B mutations are included. No candidate may be omitted,
regenerated, edited, or replaced. Each record freezes:

```text
baseline team hash
target member
root parent hash
accepted candidate hash
local delta
newly fixed / newly broken
source proposal index
source Phase-B run identity
```

For mutation `j` targeting member `i`, control and treatment are:

```text
control:   T    = (P0, ..., Pi,  ..., P4)
treatment: T'j  = (P0, ..., P'i, ..., P4)
```

Every treatment starts from the same control team. No treatment observes or
inherits another treatment's prompt, profile, gate decision, or team state.
The other four prompts and their Optimize profiles are byte-for-byte frozen.

## Data and isolation

- TeamMiniBatch12 and Full use only the frozen Optimize100 state and evidence.
- TeamMiniBatch12 is the already frozen 4 responsibility + 4 coalition + 4
  preservation set for that target task, with no backfill.
- Full means all 100 Optimize rows.
- Shadow means the frozen Shadow50 (`fold_c`) and is only reached after a
  Common-Safe pass.
- Validation50 and Test50 calls are zero.
- Test data is inaccessible.

The Phase-A baseline Optimize realization is reused as the control profile. It
is not regenerated and incurs no new provider calls. Shadow control requests
use one run-scoped exact-request cache shared across the five replay pairs.

## Progressive evaluation

Each accepted mutation is processed independently in deterministic frozen
order `(target_member, source_proposal_index, candidate_hash)`:

```text
TeamMiniBatch12 -> Full Optimize100 -> Common-Safe -> Shadow50
```

TeamMiniBatch promotion uses the existing frozen rule: it must not be
catastrophic and must have at least one positive responsibility, target, vote,
broad, or coalition signal. A failure records later stages as `NOT_REACHED`.

Full records the complete Optimize counterfactual. `full_status=PASS` means the
full evaluation completed; `full_positive_signal` separately means target
accuracy or team vote count strictly improved. Common-Safe then applies the
unchanged fixed-peer constraints. Because each replay contains one candidate,
a Common-Safe pass is the sole Shadow-eligible candidate for that replay.

Shadow uses the unchanged winner-only shadow gate. Regardless of Shadow result,
the candidate is never committed.

## Required decomposition

For every candidate the result records stage status and, when Full is reached:

```text
local_delta
target_member_full_delta
team_vote_delta
oracle_coverage_delta
team_correct_votes_gain / loss
new_team_correct_cases / lost_team_correct_cases (hashes only)
pivotal_flip_gain / loss
unique_correct_gain / loss
collateral_loss outside assigned responsibility (hashes only)
non_target_member_behavior_delta (= 0 by construction)
TeamMiniBatch / Full / Common-Safe / Shadow status
```

`collateral_loss` is an Optimize row outside the frozen assigned-responsibility
set on which the incumbent target member is correct and the candidate target
member becomes incorrect or invalid.

## Frozen classifiers

The primary unit is the five-case mapping, not a pooled IID estimate.

```text
incomplete or integrity failure
    -> TEAM_TRANSFER_REPLAY_NOT_EVALUABLE

at least two Full-evaluated candidates with team_vote_delta > 0
    -> LOCAL_IMPROVEMENT_CAN_TRANSFER_TO_TEAM_GAIN

exactly one Full-evaluated candidate with team_vote_delta > 0
    -> SINGLE_TEAM_GAIN_OBSERVED_REPLICATION_NEEDED

zero Full team gains but at least one TeamMiniBatch pass
    -> NO_FULL_TEAM_GAIN_WITH_SOME_MINIBATCH_SIGNAL

zero Full team gains and zero TeamMiniBatch passes
    -> LOCAL_TEAM_OBJECTIVE_MISALIGNMENT_SIGNAL
```

These labels are limited to this single baseline state. Scheduler efficacy,
cross-state generalization, Validation performance, and Test performance remain
unevaluated.

## Budget and stopping

The experiment has five replay pairs and stops after all five reach a frozen
terminal stage or on any operational/integrity failure. Worst-case successful
Solver provider requests are:

```text
Optimize candidate profiles: 5 x 100 = 500
Shadow control profile shared once: 50
Shadow candidate profiles: 5 x 50 = 250
total hard successful-provider ceiling: 800
```

TeamMiniBatch calls are reused if a candidate reaches Full. Transport retries
follow `COMMON_SOLVER_CONTRACT_V1`, at most four attempts per unique request.
Solver request concurrency is frozen at eight.
Reflection, GEPA, write-back, persistent realizability, Validation50, and
Test50 all have hard ceiling zero.

The experiment may not start until a clean frozen execution handoff exists and
the user separately authorizes the exact team-evaluation API scope. Preparation
and preflight are zero-API.
