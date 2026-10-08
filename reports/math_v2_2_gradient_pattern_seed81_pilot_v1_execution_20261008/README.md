# V2.2 A4 fixed-horizon Pilot

Zero-API closure passed. Exact source, manifest and single-use authorization
are frozen; preexecution_freeze.json contains the receipt hashes. No real API
call has been made for this attempt. Execution source=4b02b734e7e5306bd5f0192464b1cf0bb6a95562.

K=64 and attempt charged-token ceiling=24,000,000. The preexecution cumulative
ledger is 3,414,772 charged, 36,585,228 remaining, zero reserved. The task protects
12,585,228 tokens outside the attempt ceiling. Historical attempt6 used 832,198
tokens across ten opportunities. Fourfold historical cost inflation and a 2M
internal planning reserve give floor((24M-2M)/(4*832,198/10))=66; K is frozen at64.
Historical costs select K only and are not worst-case proofs.

The strict per-opportunity limits are 220 logical Solver calls, multiplied by4
for bounded invalid recovery, 180 physical Gradient calls, one cluster and six
reflections. Bootstrap has at most500 logical Solver calls. Totals are58,320
Solver,11,520 Gradient,64 cluster and384 reflection successes;70,288 provider
successes and1,476,048 transports. Shadow Solver calls are a subset, at most11,040.
Exact ReservationV2 is admitted before transport only if charged attempt usage
plus the complete reservation fits24M. Charges and reservation peak therefore
cannot exceed24M; the coupled admission bound can truncate before64 opportunities.
It does not guarantee completion of a maximum-cost64-step trajectory.

Scientific stopping remains team_epoch_no_commit_v1. Only SATURATION_REACHED or
NO_FEASIBLE_OPPORTUNITY constitute scientific completion. Resource ceilings are
INCOMPLETE_OPERATIONAL_TRUNCATION, PILOT_COMPLETED=NO; no resume, enlarged K or
second attempt is authorized. Validation/Test calls remain zero.

Final current suite: 1583 passed,0 failed,2 skipped,
1468 historical/private cases deselected. The first full run's single stale
governance-test failure is preserved in tests.json;94 governance/operational
checks passed after the repair. Historical replay PASS is not claimed.

Current execution has zero legacy dependencies. Frozen provider-visible fields
are unchanged, initial members have identical prompt hashes and five independent
realization lanes, and Memory starts empty. Prior reports and attempt6 raw
evidence remain unchanged. Reports publish only hashes,counters,categories and
metrics. See tests.json,closure_summary.json and the frozen resource derivation
in experiments/protocols/math_v2_2_gradient_pattern_seed81_pilot_v1/.
