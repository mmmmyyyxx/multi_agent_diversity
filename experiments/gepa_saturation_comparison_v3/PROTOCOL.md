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
