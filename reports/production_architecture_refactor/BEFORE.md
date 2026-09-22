# Production architecture before refactor

Baseline: `9285822296ae17ef830354ca8688ee0595744f08`.

This inventory describes the code reachable from the current GEPA saturation,
GEPA Layer-2 canary, and sequential Layer-2 entrypoints, plus the programmatic
four-mode backend runtime. Historical reports and run artifacts are excluded.

## Active production surface

The reachable production surface contains 30 Python files and 10,154 physical
lines. The three current executable entrypoints are:

1. `scripts/run_gepa_saturation_comparison_v2.py`
2. `scripts/run_gepa_layer2_real_canary_v2.py`
3. `scripts/run_sequential_symmetry_breaking_online_pilot_v2.py`

MARS is registered programmatically for native and Layer-2 use, but has no
equivalent single production CLI. New experiments therefore choose among
historically named runners instead of one current entrypoint.

The main active package files are `native_feed.py`, `saturation.py`, the
`local_optimizers/` backend and GEPA/MARS modules, `team_search/` controller,
evidence, admission and system adapters, and the centralized `governance/`
identity/authorization/lifecycle modules.

## Actual dependency graph

```text
GEPA_NATIVE
  runner/config -> BackendRuntimeConfig -> NativeFeedRequestBuilder
  -> GEPANativeFeedOptimizer -> official pinned GEPA -> LocalOptimizationResult

GEPA_LAYER2
  historically named runner -> Config + execution runtime
  -> primary responsibility scheduler/binding -> TeamSearchController
  -> Layer2EvidenceRequestBuilder -> GEPALayer2EvidenceOptimizer
  -> official pinned GEPA -> TeamMiniBatch -> Full -> Common-Safe
  -> Shadow -> commit

MARS_NATIVE
  programmatic registry -> NativeFeedRequestBuilder
  -> MARSNativeFeedOptimizer -> Planner/Teacher/Critic/Student/Target
  -> LocalOptimizationResult

MARS_LAYER2
  programmatic registry -> shared Layer2EvidenceRequestBuilder
  -> MARSLayer2EvidenceOptimizer -> Planner/Teacher/Critic/Student/Target
  -> shared TeamSearchController/admission when a caller supplies it

Execution for current real GEPA paths
  experiment runner -> governance/startup_identity + authorization
  -> team_search/execution_runtime provider/client/ledger
  -> lifecycle and sanitized reporting
```

## Duplicated or ambiguous ownership

- Four backend/mode implementation registrations duplicate mode dispatch.
- There is no single orchestration engine for native and Layer-2 execution.
- Stopping is centralized in `saturation.py`, but runners/backends independently
  decide how to bind and drive it.
- Scientific `Config`, `BackendRuntimeConfig`, `SaturationConfig`, and runner
  protocol dictionaries form four overlapping configuration representations.
- Real runners reconstruct config, providers, identity and orchestration with
  experiment-specific functions instead of translating into one typed spec.
- GEPA adaptation is split across adapter, optimizer, native/layer2 wrappers,
  callbacks and experiment runner wiring. MARS native/layer2 behavior is kept
  in one large 787-line file.
- Execution identity is already centralized and should be retained rather than
  replaced.

## Historical runtime leakage

| Dependency | Classification | Evidence |
|---|---|---|
| `run_seed78_primary_responsibility_ab.py` imported through sequential v1/v2 | `ACCIDENTAL_RUNTIME_DEPENDENCY` | Current sequential v2 imports v1; v1 imports Seed78 runner helpers. |
| hard-coded `seed78_update...` lineage IDs in `team_search/system_runtime.py` | `ACCIDENTAL_RUNTIME_DEPENDENCY` | Reachable team evaluation path uses a historical seed label. |
| `legacy_tcs.py` and legacy registry entry | `COMPATIBILITY_WRAPPER` | Not required by the four current modes. |
| canonical v15 modules and version constants | `CURRENT_PRODUCTION_LOGIC` | Normative historical canonical runtime remains supported. |
| old experiment scripts and reports not imported by current paths | `HISTORICAL_ONLY` | Retained for reproducibility; not refactor targets. |

## Problem statement

The repository already has the right Layer-1/Layer-2 scientific boundaries,
one immutable evidence packet, one team controller, one admission sequence and
one governance subsystem. The missing production abstraction is a typed
experiment contract and one engine that orthogonally selects backend, scope and
stopping regime. Current runners expose implementation assembly and historical
dependencies that should instead be compatibility translation only.
