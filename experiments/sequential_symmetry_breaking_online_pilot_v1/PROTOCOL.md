# Sequential Symmetry-Breaking Online Pilot v1

## Question

Can the frozen online Layer-2 scheduler, official Level-B GEPA backend, and
Common-Safe write-back autonomously move the exact homogeneous Seed-78 state
through safe intermediate commits to a plurality-responsive state and then to
an Optimize100 vote gain?

This is a single-state prospective mechanism pilot. It is not a scheduler A/B,
an IID acceptance-rate estimate, a generalization claim, or a held-out result.

## Frozen starting state

- Source state: `faa0fc81ebe71a686554355b9c5946a1765fa3913e026ab65e59f07ae01052dd`.
- All five members use the decision procedure with SHA256
  `549bc93c03f703faf5aa1bd56b557135fb6e65d0cf6c055b8fad6a15e7c87a63`.
- The 100 categorical parent outputs are reconstructed from the Phase-B parent
  task file with SHA256
  `1b27cce0731acfd0ee2fb9f34fe454caf97d3d1721be8b80d432a9bace94de27`.
- Their task IDs, member IDs, prompt hash, and source-state identity are bound by
  the Phase-B manifest with SHA256
  `32933872b0dc2ed2d7fcdaf23b5c368e5320b0861204ad908f7a20cf732a1147`.
- The five-member identity claim is bound to the v2 baseline redundancy audit
  with SHA256
  `bc9e6f43202d5f296c328cb49226d6a6f3573c4d2f973e835d0eb888643b2f05`.
- Reconstruction makes zero provider calls and must prove that the four stored
  member tasks contain the same ordered questions, gold labels, parent outputs,
  and parent prompt.
- Execution fails closed unless the baseline has five identical profiles and
  `sum_i P_i = 0`.

## Frozen online pipeline

Every opportunity uses the existing primary-responsibility scheduler:

```text
TargetScore_i = max(4 D_i, 2 N_i, C_i) / (1 + f_i)
```

It selects exactly two target branches. Each branch uses the unchanged official
GEPA Level-B adapter with metric budget 36, local validation size 12, and the
current component-specific reflection evidence. The controller then applies
the unchanged primary-lane TeamMiniBatch12, Full Optimize100 evaluation,
Common-Safe selection, winner-only Shadow50, and max-one atomic write-back.
Persistent realizability changes only through the existing online binding.

The experiment does not modify GEPA search, proposal or reflection semantics,
the scheduler, TeamMiniBatch, Common-Safe, Shadow, plurality, or write-back.

## Prospective endpoints

The baseline is state 0. After every completed opportunity, the runner records
the realized categorical team state, Optimize100 vote count, all five `P_i`
values, target/lane decisions, funnel, commit identity, realizability
transitions, and cost.

- `T_commit_opportunity`: first opportunity with a safe committed mutation.
- `T_pivotal_opportunity`: first opportunity whose realized state has
  `sum_i P_i > 0`.
- `T_pivotal_commit_index`: number of safe commits in that state.
- `T_vote_opportunity`: first opportunity whose realized Optimize100 vote count
  exceeds the frozen baseline.
- `T_vote_commit_index`: number of safe commits in that state.

These endpoints are descriptive within one frozen state. Candidate outcomes do
not alter budgets, target rules, thresholds, or stopping except for the
preregistered first-vote event.

## Stopping and budget

Stop at the first of:

1. first Optimize100 vote improvement;
2. four safe commits;
3. six consecutive completed opportunities without a commit;
4. eight completed opportunities.

Each opportunity has two GEPA branches and metric budget 36 per branch. The
budget arithmetic permits at most one accepted GEPA child per branch. A
conservative per-opportunity Solver ceiling is 596 logical rows: 72 local, 24
TeamMiniBatch, 200 Full, and 300 winner-only Shadow. Across eight opportunities
this is 4,768 Solver calls. At most 64 successful Reflection calls are allowed.
The combined successful-provider ceiling is 4,832 and the four-attempt
transport ceiling is 19,328. The runner reserves every transport attempt before
the request leaves the process.

There is no sixth commit, quota extension, adaptive budget, automatic retry of
the experiment, or resume into the same run root.

## Split isolation

- Optimize100: responsibility, GEPA local search/validation, TeamMiniBatch, and
  Full evaluation.
- Shadow50: winner-only safety gate.
- Validation50: zero access.
- Test50: zero access.

No held-out endpoint is evaluated or used for selection.

## Frozen classifier

After integrity passes:

- `ALGORITHMIC_SYMMETRY_BREAKING_WITH_VOTE_GAIN_OBSERVED` when ordered
  `T_commit <= T_pivotal <= T_vote` exists.
- `PLURALITY_RESPONSIVE_STATE_REACHED_WITHOUT_VOTE_GAIN` when `T_pivotal`
  exists but `T_vote` does not.
- `SAFE_COMMITS_WITHOUT_PLURALITY_RESPONSIVENESS` when at least one safe commit
  occurs but `T_pivotal` does not.
- `NO_SAFE_COMMIT_OBSERVED` when no safe commit occurs.
- Any baseline, integrity, budget, access, or endpoint-order violation is HOLD.

This pilot can establish algorithmic reachability only within the frozen state.
It cannot establish scheduler superiority, cross-state replication, or
held-out generalization.
