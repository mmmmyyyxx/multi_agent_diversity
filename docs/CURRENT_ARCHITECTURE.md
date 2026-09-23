# Current Production Architecture

This document describes the current execution architecture. Scientific
invariants remain normative in `docs/design/CURRENT_SPEC.md`; frozen runtime
identities and numeric constants remain authoritative in `versions.py`.
The active two-layer implementation is opt-in. The v15 method identity remains
reserved for historical canonical replay and must not be attached to a new
GEPA/MARS Layer-2 run.

## Production module map

```text
scripts/run_experiment.py
    manifest parser and gated CLI composition root
                     |
                     v
multi_dataset_diverse_rl/experiment.py
    ExperimentSpec
    RuntimeContext
    LocalOptimizationRequest
    ExperimentEngine / run_experiment
            |                         |
            | native                  | Layer2
            v                         v
local_optimizers/              team_search/controller.py
production_backends.py         responsibility/evidence/admission/write-back
  GEPABackend                          |
  MARSBackend                          v
            \-----------------> LocalOptimizerBackend

saturation.py
    shared fixed-budget/saturation state and stopping events

provider_factory.py + governance/
    provider construction, identity, authorization, lifecycle and accounting
```

The only public orchestration API for new experiments is:

```python
await run_experiment(spec, runtime, inputs, services)
```

The engine is exercised by deterministic offline fixtures. The current CLI
does not yet bind the existing startup identity, source freeze, explicit API
authorization, and lifecycle transaction to a concrete provider composition.
Its preflight therefore returns `HOLD`, and `--execute` fails before importing
any manifest-supplied factory. A real post-refactor canary needs a new frozen
execution adapter, preregistration, run identity, and explicit authorization.

Existing experiment-named runners are historical reproduction or compatibility
surfaces. They are not dependencies of the production engine. A historical
freeze never moves to the new engine: execution after an architecture change
requires a new freeze, preregistration, run identity and authorization.

## Orthogonal experiment configuration

`ExperimentSpec` selects three independent dimensions:

```text
backend             GEPA | MARS
optimization_scope  NATIVE | LAYER2
stopping_regime     FIXED_BUDGET | SATURATION
```

The four scientific modes are configurations rather than orchestration forks:

| Backend | Scope | Mode |
|---|---|---|
| GEPA | Native | `GEPA_NATIVE` |
| GEPA | Layer2 | `GEPA_LAYER2` |
| MARS | Native | `MARS_NATIVE` |
| MARS | Layer2 | `MARS_LAYER2` |

`ExperimentSpec` contains scientific choices. `RuntimeContext` contains the
run seed, provider/model binding, run/authorization/cache/ledger identities.
Layer 1 receives neither the historical monolithic `Config` nor environment
variables. Compatibility execution code may translate a legacy `Config` into
the typed context before entering production Layer 1.

## Layer 1 contract

Both backend adapters implement one small protocol:

```python
LocalOptimizerBackend.optimize(LocalOptimizationRequest, RuntimeContext)
    -> LocalOptimizationResult
```

The request freezes target member, parent prompt, scope, native or Layer2
problem, run seed, local RNG seed, local stopping provenance and request
provenance. Missing or mismatched fields produce `ExperimentContractError` or
`BackendContractError`, never an implicit `Config` `AttributeError`.

`GEPABackend` is the only production GEPA façade. It dispatches to the existing
pinned native or Layer2 GEPA adapter. Official GEPA v0.1.1 search mechanics,
population, Pareto logic, parent selection, reflection and local acceptance are
unchanged. `MARSBackend` similarly dispatches to the existing native or
packet-owned MARS search while preserving Planner/Teacher/Critic/Student/Target
semantics.

Layer 1 owns local search only. It cannot select team members, compute team
responsibility, update realizability, apply Common-Safe or Shadow, or write a
prompt into the team.

## Layer 2 contract

Layer 2 owns:

- target selection and primary responsibility;
- persistent realizability and the frozen evidence schedule;
- one immutable `ResponsibilityEvidencePacket`;
- TeamMiniBatch, Full, Common-Safe and Shadow evaluation;
- competitive selection and atomic write-back;
- latest committed transition focus/anchor state.

Responsibility is current team evidence. Focus is the latest committed
parent-correct to child-wrong set; anchor is parent-wrong to child-correct.
Roots have empty focus and anchor. Uncommitted mutations cannot change either.
Local evaluation examples are frozen by Layer 2. GEPA and MARS Layer2 modes
consume the same packet schema and cannot fetch or backfill examples.

The production Layer2 modules depend only on the `LocalOptimizerBackend`
boundary. They do not import GEPA or MARS implementations. Given identical
local results, both backends produce identical team evaluation, admission and
write-back behavior.

## Native and Layer2 execution

Native mode invokes the selected backend with its existing backend-owned
Optimize-only data semantics. It does not create a fake team controller.

Layer2 mode runs opportunities through the shared `TeamSearchController`:

```text
target/responsibility
  -> frozen packet
  -> backend local result
  -> TeamMiniBatch
  -> Full
  -> Common-Safe selection
  -> Shadow
  -> commit or reject
```

Backend-specific logic ends at `LocalOptimizationResult`.

## Stopping

`saturation.py` is the one stopping subsystem. Fixed-budget runs stop on their
preregistered scientific unit limit. Saturation runs retain local patience 3
and team patience 2. Backends report their existing complete optimization unit:
GEPA native epoch, GEPA Layer2 evidence epoch, MARS native round or MARS Layer2
round. Layer 2 reports complete team epochs. Provider, optimizer, team-epoch and
wall ceilings remain operational aborts and never masquerade as convergence.

## Execution boundary

`ProviderClientFactory` is the sole current client-construction boundary.
Credential environment names and endpoints remain in execution code and never
enter scientific requests. The existing governance package remains the source
of truth for startup identity, preregistration, authorization, lifecycle and
provider-boundary checks. Ledger/cost facts remain execution-owned.

Telemetry ownership is single-source:

- Layer 1: proposal, lineage and local objective events;
- Layer 2: responsibility, packet, team deltas, admission and commit events;
- execution: provider attempts, tokens, lifecycle and cost accounting.

## Adding an experiment or backend

A new GEPA or MARS experiment changes only its manifest/configuration and
execution inputs. It does not add a runner, controller, backend adapter or
stopping implementation. The common CLI loads the scientific/runtime sections
and invokes the central engine through one execution composition function.

A future SEPO backend would implement `LocalOptimizerBackend` and be selected
by an intentionally versioned backend extension. Layer 2, evidence, admission,
stopping and execution governance would remain unchanged. SEPO is not
implemented by this architecture refactor.
