# gepa_saturation_comparison_v2

This is the fresh executable successor to the immutable v1 preregistration at
`80442264a8086f3fe407e1551ae2ecc8448f8676`. The scientific method remains anchored at
`a85e31bea2ab28f62abb31337b91f9895b11ae37`; the execution harness is `351405d898ed1add9e641c87c8d0567912019866`.

Execution-only changes: explicit `provider_profile=lwj`, exact models, and
`FRESH_DETERMINISTIC_INITIALIZATION_V1`. No historical private parent, candidate freeze, or
process-local cache is required. Root focus and anchor are empty. Initialization
Solver usage is accounted separately from optimization usage.

Validation50 calls: 0. Test50 calls: 0. No implicit provider/model fallback.

Formal arms are GEPA_NATIVE_SATURATION and GEPA_LAYER2_SATURATION for seeds
[80, 81, 82]. Local patience is 3; team patience
is 2. The study remains execution-gated on the canary
and sequential prerequisite statuses and requires a later one-time authorization.
