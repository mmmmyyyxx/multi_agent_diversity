# Transition Reachability Audit V1

## Scope

- repository: `mmmmyyyxx/multi_agent_diversity`
- audited parent SHA: `c921c395a75320f02081eeae517be7a5f975d248`
- amendment branch: `fix/target-or-team-progress-v3`
- API / Solver / Gradient / Pattern / Validation / Test calls: **0**
- purpose: repair the transition reachability contradiction exposed by the
  byte-identical five-member initial condition, then audit adjacent current
  control-flow assumptions for the same class of logical inconsistency.

Historical V2.1 manifests, bindings, reports and transition identity
`initial_competence_team_gain_v2` remain immutable replay evidence.

## Established contradiction

The current MATH V1_2 initial condition uses five byte-identical prompts and the
completed Seed81 Pilot observed the same 22 correct rows for every member and
Oracle=22. With fixed peers and single-member replacement, a newly correct
answer on one of the 38 all-wrong rows supplies at most one correct vote against
four fixed wrong peers (or a tie if all wrong classes split). Therefore the
V2.1 transition requirement

```
target >= immutable initial floor
AND
team Vote strictly improves
```

cannot accept the first update from that parent. No first commit means no changed
parent can be reached. The zero-commit outcome is therefore not evidence that
search failed to find an admissible first transition.

This is consistent with the earlier historical
`accepted_local_mutation_team_transfer_v2` result, which already recorded
0/5 TEAM_POSITIVE and identified four identical fixed peers as structurally
locking plurality against one-member replacement.

## V2.2 transition

Current semantics are versioned as
`initial_competence_target_or_team_progress_v3`:

```
target >= immutable_initial_target
AND team_vote >= parent_team_vote
AND (target > parent_target OR team_vote > parent_team_vote)
AND terminal_invalid_guard
```

Thus two progress paths are legal:

1. **TEAM**: team Vote strictly improves; target may decline relative to the
   incumbent only down to its immutable initial floor.
2. **TARGET**: team Vote is non-regressing and target competence strictly
   improves.

A candidate such as the manual P2 diagnostic (target 22 -> 25, team Vote
22 -> 22) is therefore reachable, while a target-positive / Vote-negative
candidate remains rejected.

## Coupled inconsistencies found and repaired

### 1. Winner ranking omitted target competence

V2.1 ranked feasible Full candidates by team Vote and secondary safety/soft
metrics. Once Vote-neutral target improvements become feasible, two candidates
with equal Vote (for example target 23 vs 25) could otherwise be ordered by
soft Vote or prompt hash. V2.2 keeps team Vote primary and inserts target
competence second in the deployment winner key.

### 2. Pilot bound proof depended on strict Vote progress

The current Pilot bound explicitly proved at most N commits from strict integer
team Vote improvement. Target-only commits invalidate that proof. Reusing it
would create an operational ceiling that could truncate a scientifically legal
V2.2 trajectory.

V2.2 therefore fails closed with
`TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN` until a new finite bound is
derived and frozen. No numerical replacement was guessed in this amendment.

### 3. Current execution entrypoint still pointed at V2.1 binding identities

The current preexecution helper previously defaulted to a V2.1 offline profile,
and the current MATH binding entrypoint accepted the historical V2.1 Pattern
binding identity before later policy checks rejected it. V2.2 now has its own
method/transition/execution-binding identities. The current entrypoint fails
closed until a fresh V2.2 binding exists; V2.1 stays behind historical replay.

### 4. Allocation telemetry assumed every commit had positive team gain

V2.2 commits may have `realized_team_gain == 0`. Current allocation telemetry
now records `realized_target_gain` and
`realized_progress_path in {TEAM, TARGET, TARGET_AND_TEAM, NONE}` instead of
using team gain alone to describe accepted progress.

## Adjacent logic checked

- **TeamProbe promotion:** no structural target-only deadlock found. Promotion
  already treats positive target/responsibility/broad deltas as signal, so a
  Vote-neutral target-positive candidate can reach Full. Sampling quality and
  false-negative rate remain empirical questions.
- **Full evaluator:** the active provider delegates safety to the bound
  transition policy, so the V3 change propagates to Full feasibility without a
  second hidden strict-Vote gate.
- **Global stopping:** no structural contradiction found. A commit resets the
  parent epoch; two complete no-commit epochs still define scientific
  saturation. The unresolved issue is the operational upper-bound proof, not
  the stopping rule itself.
- **Rolling Risk Memory:** current success is based on completed atomic commit,
  not positive team delta, so target-only commits are representable. Historical
  V2.1 memory/tests that assert positive team delta remain replay semantics.
- **Shadow:** no reachability deadlock found. It requires nonnegative Shadow
  Vote and bounds target loss by two. For a target-only Full winner this means
  the target improvement itself need not replicate on Shadow. That is an
  explicit safety-vs-transfer design choice, not a logical contradiction; it
  should not be silently changed without a separate scientific decision.

## Current readiness

V2.2 implementation semantics are staged, but real execution is **HOLD**.
Before any new Pilot:

1. derive and freeze a valid target-or-team progress provider/opportunity bound;
2. create a fresh V2.2 execution binding and preexecution source identity;
3. run zero-API/current-suite conformance on that exact source;
4. obtain a fresh attempt-specific authorization.

The completed V2.1 Seed81 Pilot remains valid historical evidence of candidate
generation behavior, but its zero commits and strict-Vote Full rejection cannot
be interpreted as evidence against first-step transition efficacy from the
symmetric parent.
