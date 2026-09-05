# Seed75 Vote-Aligned Generic Shadow Pilot

Status: `COMPLETE`, official corrected audit `PASS`, classifier
`NO_CLEAR_SIGNAL`. This is descriptive one-seed pilot evidence, not a confirmed
multi-seed result.

P0 is canonical round-robin D2 Generic. P1 changes only target scheduling to
the deterministic hierarchy `direct_flip > near_margin > pure_coverage`, with
round robin fallback. Solver is `qwen3-8b` with thinking disabled;
Teacher/Critic/Student/Evaluator are `qwen3.7-flash`.

## Completion-semantics repair

P0 scientifically reached the frozen early-stop rule at its 21st update
opportunity (zero-based index 20): the final six opportunities had no
Shadow-approved commit. The original generic CLI incorrectly required all 32
planned opportunities before marking training complete. The preserved P0
checkpoint and update registry were deterministically audited and remained
byte-for-byte unchanged. A separate derived completion marker records
`completed_by_early_stop`; no P0 update was rerun and no fabricated updates
were added.

The completion repair changed engineering state semantics, not the scientific
protocol. A pre-update P1 identity-gate attempt made no model calls and is
preserved outside the valid P1 cell. It exposed that the legacy initialization
hash included execution provenance. An explicit transition manifest allowed
only execution commit and task-manifest provenance to change; dataset hashes,
prompt hashes, solver identity, member baselines, probe identity, and the full
initial G/H/M outcome remained equal. P1 then ran once from this matched state.

The first post-run audit is preserved as an original HOLD because it compared
the provenance-coupled initialization hash. The corrected auditor compares a
scientific-state signature while independently checking the frozen execution
commit. The corrected gate is PASS with no protocol blocker.

## Trajectory completion

| Arm | Planned maximum | Executed | Termination | Final no-commit streak | Shadow-approved commits |
|---|---:|---:|---|---:|---:|
| P0 | 32 | 21 | `completed_by_early_stop` | 6 | 9 |
| P1 | 32 | 20 | `completed_by_early_stop` | 6 | 7 |

P0's 11 and P1's 12 remaining opportunities are genuinely unexecuted. They
are not missing data and are not counted as completed updates.

## Final frozen-state results

| Arm | Split | VoteAcc | MeanMemberAcc | Vote−MeanMember | OracleAcc |
|---|---|---:|---:|---:|---:|
| P0 | Optimize100 | 0.790 | 0.728 | 0.062 | 0.980 |
| P1 | Optimize100 | 0.760 | 0.712 | 0.048 | 0.980 |
| P0 | Shadow50 | 0.660 | 0.636 | 0.024 | 0.880 |
| P1 | Shadow50 | 0.600 | 0.600 | 0.000 | 0.860 |
| P0 | Validation50 | 0.640 | 0.652 | -0.012 | 0.980 |
| P1 | Validation50 | 0.700 | 0.636 | 0.064 | 0.800 |

Validation P1−P0 deltas are:

- VoteAcc: `+0.060`
- MeanMemberAcc: `-0.016`
- EnsembleGain (`Vote−MeanMember`): `+0.076`
- OracleAcc: `-0.180`

Validation member accuracy is P0 `[0.68, 0.62, 0.74, 0.50, 0.72]` and P1
`[0.62, 0.60, 0.58, 0.68, 0.70]`. P0 mean/min/max/std are
`0.652/0.500/0.740/0.086`; P1 values are
`0.636/0.580/0.700/0.046`.

Validation support depth (`G0..G5`) is:

| Arm | G0 | G1 | G2 | G3 | G4 | G5 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 1 | 14 | 3 | 3 | 11 | 18 |
| P1 | 10 | 3 | 2 | 8 | 7 | 20 |

P1 substantially reduces singleton-only correctness and slightly increases
`G>=3` support (35 versus 32), which is consistent with improved plurality
conversion. However, the increase in `G0` from 1 to 10 and Oracle loss of 0.18
show a large broad-coverage regression. The Vote gain therefore does not meet
the frozen positive-signal rule, which allowed at most 0.01 MeanMember loss and
required preserved broad competence. The correct local classifier is
`NO_CLEAR_SIGNAL`, not a positive efficacy claim.

## Scheduler telemetry

P1 allocated 36 target slots to `direct_flip`, 3 to `pure_coverage`, 1 to
fallback round robin, and 0 to `near_margin`. Shadow-approved commits by those
lanes were respectively 4, 1, 1, and 0. Lane commit attribution counts the
Optimize winner's selected lane; it does not turn the two-target update into
two commits.

## Transfer gaps

Vote gaps (`Optimize−Shadow`, `Optimize−Validation`, `Shadow−Validation`) are
P0 `(0.130, 0.150, 0.020)` and P1 `(0.160, 0.060, -0.100)`. P1 is weaker on
Shadow50 but stronger on Validation50, so the one-seed direction is
split-sensitive and must not trigger Seeds76/77 automatically.

## Isolation and API accounting

- Final evaluations occurred only after both trajectories were terminal.
- Final Shadow50 reused frozen cached solver outcomes; each Validation50 added
  250 logical solver outcomes.
- Current repair task added zero P0 model calls.
- P1 training recorded 6,757 provider attempts: 6,756 successful and one
  failed operational attempt.
- P1 Validation50 recorded 252 successful provider responses for 250 logical
  solver outcomes; P0 Validation50 recorded 250.
- `NEW_TEST_CALLS = 0`; Test50 was not loaded, evaluated, selected on, or used
  in analysis.
- Seeds76/77 were not run.
