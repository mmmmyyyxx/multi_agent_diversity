# sequential_symmetry_breaking_online_pilot_v2

This is the fresh executable successor to the immutable v1 preregistration at
`80442264a8086f3fe407e1551ae2ecc8448f8676`. The scientific method remains anchored at
`a85e31bea2ab28f62abb31337b91f9895b11ae37`; the execution harness is `351405d898ed1add9e641c87c8d0567912019866`.

Execution-only changes: explicit `provider_profile=lwj`, exact models, and
`FRESH_DETERMINISTIC_INITIALIZATION_V1`. No historical private parent, candidate freeze, or
process-local cache is required. Root focus and anchor are empty. Initialization
Solver usage is accounted separately from optimization usage.

Validation50 calls: 0. Test50 calls: 0. No implicit provider/model fallback.

The mechanism pilot preserves T_commit, T_pivotal, T_vote and the complete
target -> GEPA -> TeamMiniBatch -> Full -> Common-Safe -> Shadow -> atomic
write-back path. Its small opportunity ceiling is distinct from saturation.
