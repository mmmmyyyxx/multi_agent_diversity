---
archive_status: HISTORICAL_REPLAY_ONLY
source_document: docs/CURRENT_ARCHITECTURE.md
source_commit: ec993b13a252699f19b93348b01570ffd9bea879
source_section: FULL_DOCUMENT
---

# Current Production Architecture

This document describes the opt-in Unified Team Prompt Search architecture.
Scientific invariants remain normative in `docs/design/CURRENT_SPEC.md`;
runtime identities and numeric constants remain authoritative in `versions.py`.
The canonical v15 and the former two-layer modes remain historical replay
paths with their original identities.

## Current unified production graph

```text
run_experiment(SearchMethodConfig, RuntimeContext, UnifiedExperimentInputs,
               UnifiedExperimentServices)
    -> UnifiedSearchOrchestrator
    -> BenchmarkAdapter + AggregationPolicy
    -> TeamStateSnapshot + StateAnalyzer
    -> OpportunityBuilder(feasibility, target, role evidence)
    -> SearchEngine
    -> CandidateEvaluationPipeline(TeamProbe, Full)
    -> TransitionPolicy + AdaptiveValidationGate
    -> TeamStateCommitter + HistoryState/MemoryProvider
    -> GlobalStopPolicy
```

The current BBH composition lives in `search/current_bbh_runtime.py`. It uses
raw overlapping plurality responsibility, V4 4/4/4 evidence and a derived
GEPA search adapter. Low-level fixed-peer BBH evaluation and atomic system
write-back are reused at a compatibility boundary to preserve semantics.
The old controller and task builder are absent from the new method's control
flow. `SearchMethodConfig` identifies every active policy; a future GEPA
change receives a new method identity. The official GEPA baseline remains
pinned for its historical control.

Plurality is active for current BBH. Equal-status LLM aggregation is available
for inference with public input only; no LLM-aggregation responsibility policy
is frozen, so those optimization tasks fail closed. New real-API execution is
also held before provider construction until a separate governed prep and
authorization exist.

## Historical two-layer production map (replay only)

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
Layer 2 freezes local `M_eval` from the same TeamMiniBatch12 example IDs before
candidate generation. Local GEPA acceptance compares target-member scores;
later TeamMiniBatch promotion compares fixed-peer team metrics. Team-level
outcomes cannot change the local accepted frontier, candidate IDs or scores.
Preservation prioritizes latest accepted target-transition changes OR current
target pivotality to correct plurality, then smaller positive margin, greater
valid-answer disagreement, SHA256(example ID), and raw ID only on hash collision.
This is a deterministic proxy, not a learned vulnerability model. GEPA and MARS Layer2 modes
consume the same packet schema and cannot fetch or backfill examples.

The production responsibility graph permits the same team residual to be
legally assigned to several members. Primary responsibility chooses who is
optimized, without exclusive ownership. TeamMiniBatch12 draws four
primary-lane repair, four team-correct preservation, and four global team-hard
residual rows; team-hard rows need not be unassigned. Equal target scores use
ascending member ID. The former coalition quota and seed/RR tie ordering are
superseded and occur only in archived experiment definitions.

The V4 Layer2 diagnostic keeps full pre-routing raw-legal responsibility for
`D/N/C/V` and target scores. Before selection, Layer2 masks members that cannot
form the unchanged 4/4/4 TeamMiniBatch or fit four repair role items after
reserving exact latest-transition focus/anchor within 36 nominal packet slots.
The mask does not alter `V/(1+f)`; skipped members do not change `f`. Layer2
then supplies a deterministic bounded search curriculum to GEPA. The full
responsibility universe, scheduled packet and actually delivered batches are
separate audit objects; `M_eval` remains the full frozen TeamMiniBatch12 IDs.

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
<!-- END_VERBATIM_ARCHIVE_BODY -->
