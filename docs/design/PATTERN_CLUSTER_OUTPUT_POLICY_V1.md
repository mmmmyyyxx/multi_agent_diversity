# Cluster-specific generation ceiling

`PATTERN_CLUSTER_GENERATION_POLICY_V1` is an explicit fresh V2.2 Pilot
operational binding, retained by the current composition where explicitly bound.
Only `pattern_cluster` uses max_completion_tokens=8192
and accounting output ceiling=8202 (existing ten-token measurement tolerance).
Reflection and per-example Gradient retain the exact V3 policy at 1800/1810.
Sampling, model and nonthinking controls are unchanged. Old bindings without
the role override retain their original request bytes and caps.

The override enters method, manifest, provider/request, startup, authorization
and source identities. Reservation V2 reserves the entire transmitted role cap
plus tolerance before each transport. K=64, the 24M attempt ceiling and
40M cumulative authorization describe the original V2.2 experiment only.
Current horizons and budgets come from the frozen experiment; they are admission
bounds, not convergence claims or guarantees that every maximum-cost path fits.

The global clustering prompt, complete Gradient set, partition schema, aliases,
support IDs and same-F Pattern selection remain unchanged. No cluster-count
restriction or content truncation is introduced. Context admission uses the
actual role output ceiling. Any length/max_tokens/max_output_tokens finish
reason is rejected before partition parsing, even with apparently complete JSON.

The previous truncated attempt remains closed and immutable. Operational
recovery requires a new individually frozen attempt and exact single-use scope
under a new exact explicit user authorization. Scientifically valid
unfavorable results never authorize regeneration, retry or algorithm changes.
Validation requires its independent authorized phase; Test remains sealed.
