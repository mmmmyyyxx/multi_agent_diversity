# Production architecture after refactor

Base: `9285822296ae17ef830354ca8688ee0595744f08`.

## Outcome

The current production surface now has one public orchestration API,
`run_experiment(spec, runtime, inputs, services)`, and one CLI composition
root, `scripts/run_experiment.py`. Backend, optimization scope, and stopping
regime are orthogonal typed fields rather than runner identities.

```text
ExperimentEngine
  native  -> LocalOptimizerBackend -> GEPA or MARS native search
  layer2  -> shared TeamSearchController -> immutable evidence packet
          -> LocalOptimizerBackend -> shared team admission/write-back

Execution services
  -> one provider construction boundary
  -> existing centralized identity/authorization/lifecycle/accounting
```

`GEPABackend` and `MARSBackend` are the two production Layer-1 facades. They
translate the explicit `LocalOptimizationRequest` and `RuntimeContext` to the
existing frozen backend seams; they do not reimplement either search core.
The shared Layer-2 package imports neither facade and owns target selection,
responsibility/evidence, TeamMiniBatch, Full, Common-Safe, Shadow, and atomic
write-back.

Historical runners and all historical artifacts remain available for exact
reproduction, but none is reachable from the new production entrypoint. In
particular, `run_seed78_primary_responsibility_ab.py` is no longer a current
production dependency. No historical runner or report was deleted.

## Before / after metrics

The production graph uses the same explicit counting boundary as `BEFORE.md`:
current orchestration, backend internals that remain reachable, Layer 2,
stopping, execution governance, and the current CLI; reports/runs and
historical-only reproduction entrypoints are excluded.

| Metric | Before | After |
|---|---:|---:|
| Active production Python files | 30 | 30 |
| Active production physical LOC | 10,154 | 7,538 |
| Current production entrypoints | 3 | 1 |
| Shared experiment orchestration implementations | 0 | 1 |
| Production Layer-2 controller implementations | 1 | 1 |
| GEPA mode-specific production facades | 2 | 1 |
| MARS mode-specific production facades | 2 | 1 |
| Shared stopping subsystems | 1 | 1 |
| Execution identity/governance subsystems | 1 | 1 |
| Historical experiment-specific files reachable from production | 6 | 0 |
| Overlapping production config/spec representations | 4 | 2 |

The active file count did not increase: the unified engine, two backend
facades, provider boundary, and one CLI replace experiment-runner code in the
current graph while backend search internals remain intentionally retained.
There were no source deletions in this conservative first pass.

## Complexity removed

- mode selection no longer requires four backend/mode registrations;
- new experiments no longer choose a historical runner by experiment name;
- production Layer 1 no longer receives the historical giant `Config` bag;
- seed ownership is explicit in `RuntimeContext` and checked at each boundary;
- the reachable team-evaluation lineage no longer embeds `seed78`;
- provider-client construction has one boundary;
- currently frozen, unexecuted experiments are recorded as superseded rather
  than silently retargeted to the new source.

## Behavioral and governance validation

- Eight deterministic golden traces (GEPA/MARS x native/Layer2 x
  fixed-budget/saturation) are byte-identical before and after.
- The focused production/governance suite passed: `174 passed`.
- `compileall` passed for the package and scripts.
- The manifest-only production preflight passed with provider, Validation,
  and Test counts all zero.
- Pinned GEPA identity replay passed at v0.1.1,
  `b4dbb55b7601dac448cdb836d5a401ca7d9eb920`, source SHA-256
  `84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8`.
- The full clean-worktree baseline was `1193 passed, 1 skipped, 35 failed,
  20 errors`; after the refactor it was `1210 passed, 1 skipped, 35 failed,
  20 errors`. The non-passing test set is identical and consists of existing
  missing private historical artifacts/hash fixtures; no new failure class was
  introduced.
- Golden SHA replay, changed-file sanitization, endpoint/key scan, and
  `git diff --check` passed.
- Real provider attempts: `0`; Validation calls: `0`; Test calls: `0`.

## Acceptance summary

| Requirement | Result |
|---|---|
| Unified experiment engine | YES |
| One production GEPA facade | YES |
| One production MARS facade | YES |
| One Layer-2 controller | YES |
| One team-admission pipeline | YES |
| One stopping subsystem | YES |
| One execution/governance subsystem | YES |
| Giant historical `Config` required by production Layer 1 | NO |
| Seed-specific runner required by current production path | NO |
| Golden semantic traces | PASS |

```text
SCIENTIFIC_METHOD_CHANGED = NO
```

Any future real execution requires a fresh freeze, preregistration, run
identity, and authorization. This refactor does not authorize an experiment.
