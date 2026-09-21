# Unified backend four-mode protocol

This protocol freezes backend choice and controller/data-flow choice as
independent configuration fields in one codebase:

| `optimizer_backend` | `optimization_mode` | Mode |
|---|---|---|
| `gepa` | `native` | GEPA_NATIVE |
| `gepa` | `layer2` | GEPA_LAYER2 |
| `mars` | `native` | MARS_NATIVE |
| `mars` | `layer2` | MARS_LAYER2 |

Native controls retain the selected backend's documented Optimize-only data
flow. Layer2 treatments consume the one immutable
`ResponsibilityEvidencePacket`; the backend cannot select examples outside it.

Every run freezes the seed, solver and optimizer models, budget, data-split
manifest, initial state, code SHA and configuration hash. All modes use the
common run schema in `infrastructure/unified_backend_run.schema.json`.

This preparation uses zero provider calls and does not access Validation50 or
Test50. `READY_TO_RUN=false` until a future experiment receives its own explicit
authorization and complete preregistration.
