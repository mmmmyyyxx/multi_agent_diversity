# Current model and provider routing

`OPENLUX_SOLVER_LWJ_OPTIMIZER_V1` is the sole current deployment policy.
Solver, including initial profiling, local evaluation, Probe, Full, Shadow and
authorized final evaluation, uses `gpt-4o-mini` on the `openlux` provider profile.
Reflection, per-example Gradient and Pattern clustering retain `qwen3.7-flash`
on `lwj`. Both clients share the existing broker, ledger, bounds and retry policy.
Unknown roles or model-route mismatches are rejected; provider fallback is forbidden.

Current bindings use `MATH_ROLE_ROUTED_GENERATION_RECOVERY_BINDING_V5` and
`ROLE_ROUTED_RECOVERY_MEMBER_LANE_CACHE_V6`. Full routing policy, model names,
environment variable names and `OPENAI_SOLVER_DECODING_POLICY_V3` enter frozen
run, scope and request identities. Solver retains temperature 0.2, top_p 0.8,
zero presence/frequency penalties and bounded 3600/6144 output capacity; its
wire excludes enable_thinking, top_k and min_p. Optimizer wire is unchanged.

Credentials and deployment URLs are supplied through private deployment
configuration. OpenLux environment values take precedence over the ignored
local configuration; guarded zero-API execution cannot read that configuration.
Neither credentials nor deployment URLs enter published manifests or reports.
Historical manifests, results and source identities remain immutable.
Model replacement invalidates old response caches and competence measurements.
Fresh preregistration and exact API authorization are required for a paid run.
This amendment does not authorize a Canary, Pilot, Validation or Test execution.
