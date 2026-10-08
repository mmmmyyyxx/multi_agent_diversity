# V2.2 A4 / Seed81 Pilot attempt — output truncation

PILOT_COMPLETED=NO. Runner status=EXECUTION_ABORTED. The first Pattern clustering
response reached the frozen1800-token output cap, returned finish_reason=length,
and aborted before the first opportunity. This is NOT_EVALUABLE_AS_COMPLETE_PILOT.
No scientific saturation or valid zero-commit Pilot result was obtained.

## A. Operational ceiling

K=64; attempt charged-token ceiling=24M; external protected margin=12,585,228.
Historical attempt6 cost832,198 tokens over ten opportunities. Fourfold planning
inflation and a2M internal reserve give floor((24M-2M)/(4*832,198/10))=66;64 was
frozen. Historical averages select K only and are not worst-case proofs.
Per opportunity the strict maxima are880 physical Solver,180 Gradient,one
cluster,six reflections; bootstrap at most2,000 physical Solver calls. Totals:
58,320 Solver,11,520 Gradient,64 cluster,384 reflection;70,288 successful provider
calls;1,476,048 transports. Exact full ReservationV2 admission jointly limits
charged attempt usage and reservation peak to24M. It may truncate before K and
does not guarantee maximum-cost64-step completion. Scientific stopping remains
team_epoch_no_commit_v1. This actual abort reached neither K nor the attempt
token ceiling; it was a separate provider-output contract failure.

## B. Final freeze

Source=4b02b734e7e5306bd5f0192464b1cf0bb6a95562; binding=MATH_V2_2_EXECUTION_BINDING_V1.
Attempt/cache=math_v2_2_gradient_pattern_A4_seed81_pilot_attempt1. preexecution_freeze.json records binding,manifest,
startup and authorization hashes. The exact single-use scope was consumed and
is permanently closed. Historical authorization and scientific outputs were
not reused. No scientific policy or frozen output cap changed.

## C. Tests

Final current suite: 1583 passed,0 failed,2 skipped,
1468 historical/private cases deselected. Network attempts=0. The first full
suite's one stale governance-test failure and94-passing repair verification are
preserved separately in tests.json. Historical replay PASS is not claimed.

## D. Actual execution

Opportunities=0/64; proposals=0; Fulls=0; TARGET=0; TEAM=0; TARGET_AND_TEAM=0.
Initial Vote/Oracle=22/22; member scores=[22,22,22,22,22]. Last durable scientific
state is that same initial state: no opportunity or deployed mutation occurred.
The five identical prompt hashes used independent member realization lanes.
The300 logical initial evaluations used330 physical Solver calls;40 raw invalid
responses exhausted recovery on10 logical evaluations, which scored incorrect.

All38 Gradient records passed on their first generation;one bounded numeric
warning was retained, with no contract rejection. The single cluster response
used1800 output tokens, exactly its request cap; no cluster JSON parsing,
partition completion,Layer1,Full,Shadow or transition was reached. Output text
and source examples remain private. There is no Full-outcome basis for blaming
promotion,floor,Vote regression or Shadow, and the prior strict-Vote-only rule
is not the current transition contract.

Successful provider calls=369 (Solver330,Gradient38,cluster1,reflection0).
Transports=369; transport failures=0. Task charged tokens=109,444;
cumulative=3,524,216; remaining=36,475,784; in-flight reservations=0.
Validation=0; Test=0. No extra diagnostic,LLM judge,rerevaluation or rerun.

## E. Scientific evidence

Commit finiteness and source/scope/accounting integrity: ESTABLISHED.
Observed Gradient contract compliance38/38: SUPPORTED.
Complete Pilot execution: NOT SUPPORTED.
Real TARGET reachability,later TEAM conversion,Gradient semantic usefulness,
overall efficacy,component causality and generalization: INCONCLUSIVE.
The blocker is first-cluster provider output truncation, before scientific
transition evaluation. No negative method-efficacy claim follows from this abort.

## F. Publication and closure

369 immutable response receipts,300 sealed resolved-cache entries and the ledger
replay passed owner zero-API audit. All683 actual raw files remained unchanged.
The frozen manifest is preserved. owner_abort_closure_receipt.json is explicitly
an owner abort record; it does not replace or claim a runner completion receipt.
The failure registry records the observed cap incident. Search remains closed;
there is no authorization for another attempt or a generation-limit amendment.
The containing result publication commit carries the sanitized report; remote
main verification is performed after publishing. See analysis_bundle_index.json.
