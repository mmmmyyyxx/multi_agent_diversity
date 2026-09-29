# Formal GEPA saturation comparison V3 (execution gated)

V3 supersedes the unexecuted V2 freeze. It does not reuse V2 execution
artifacts. The treatment is the active GEPA + Layer2 V4 method; the control
is GEPA native. They use the same frozen official GEPA search core and common
Solver contract, but the native and Layer2 search units are not equated.

Six separate logical cells are frozen in seed order 80, 81, 82, with
GEPA_NATIVE before GEPA_LAYER2_V4 per seed. Each cell starts from the same
deterministic five-member initialization and the frozen Optimize100/Shadow50
split. Solver is qwen3-8b without thinking. Reflection is qwen3.7-flash.
Provider profile is lwj with no fallback. Root focus/anchor are empty.

The treatment uses raw pre-routing overlapping responsibility, D/N/C and
V=max(4D,2N,C), feasibility masking, V/(1+f) ranking, bounded 36-item
evidence, TeamMiniBatch 4/4/4 with identical M_eval IDs, fixed-peer Full,
Common-Safe, winner-only Shadow and atomic commit. Team epochs are scoped to
one parent state. Eligibility is V>0 and feasible; a successful commit closes
the current epoch, resets outer patience and recomputes eligibility on the
successor. Two complete no-commit epochs satisfy team saturation. Empty
eligibility stops before any opportunity-stage provider call.

Native uses official GEPA Optimize-only feed and local no-update patience 3.
Neither arm has a scientific hard budget. The frozen emergency ceilings are
100000 provider attempts/successes, 100000 optimizer steps, 10000 team epochs
and 86400 seconds. Emergency termination is never scientific saturation.

Validation50 and Test50 are prohibited (zero calls). Formal execution remains
gated until the separate real Seed81 V4 diagnostic is scientifically valid,
the pilot phase is closed, and a new attempt-specific API authorization is
given. This zero-API preparation consumes no authorization and does not run
either real arm.

## Attempt2 pilot closure and final-state representation

The six attempt1 preparations remain immutable historical preregistrations and
are `SUPERSEDED_BEFORE_EXECUTION`. Attempt2 binds the Seed81 V4 attempt3 pilot
closure in `PILOT_CLOSURE.json`: scientific validity is `VALID`, efficacy is
`NOT_EVALUABLE`, and the pilot is `CLOSED_VALID_INCONCLUSIVE`. This satisfies
the diagnostic prerequisite without changing the scientific method or
authorizing a Formal API call.

After search, Native's first returned official GEPA candidate is represented
as a homogeneous five-member team. If no candidate is returned, its initial
five-member team remains final. Layer2's final five committed prompts remain
in member order. Each cell privately persists the five prompts and their
hashes in `final_team_materialization.json`; this representation does not
feed back into search. A completed cell's raw artifacts are inventoried before
the read-only `formal_trajectory_trace.jsonl` is derived.

The separate `POST_FREEZE_VALIDATION50_EVALUATION.md` fixes the later Stage 0
generalization endpoint. It requires new explicit authorization after all
six searches are frozen. Test50 remains sealed throughout Stage 0.

The initial attempt2 prep at execution source `504dadb4024a0c69cc2f8aac6e6d400f77f3a680`
is superseded before execution because its post-freeze policy mislabeled
adaptive Shadow50/fold_c as Validation50. No Formal real call or Formal efficacy
observation preceded this correction. The fresh attempt2 prep retains the same
search method and six cell names but binds the independent `validation` 50
question-hash set from `anti_overfitting_split_v1/split_manifest.json` as the
post-freeze endpoint. Fold_c remains Shadow50, and Test50 stays sealed.

## Attempt3 execution-only repair

The corrected attempt2 campaign is closed as an invalid execution campaign.
Seed80 Native consumed its one-time authorization and made 1137 successful
provider calls, but its lifecycle ended `ABORTED` after a post-search JSON
read-back type mismatch. Its scientific validity is
`INVALID_EXECUTION_CONFORMANCE`; efficacy is `NOT_ASSESSED`. The other five
attempt2 cells are `SUPERSEDED_BEFORE_EXECUTION`. The published incident is
bound to commit `c29bdceca25d63b93edba9eaf57bff33b9ad51c3`.

Attempt3 keeps the same six seeds and arms, scientific search method, models,
Optimize100, Shadow50, and independent post-freeze Validation50 policy. Its
execution wrapper first converts the completed result to the strict JSON value
that disk read-back will produce, then verifies the persisted summary against
that value. Tuples become JSON arrays; unsupported values and non-finite
numbers fail. This repair occurs after search and does not feed into candidate
generation, selection, stopping, team materialization, or any online decision.

All six attempt3 cells require new attempt-specific authorization. This
preparation authorizes no real API calls or held-out evaluation.
