# Repository Map

Read AGENTS.md, CURRENT_SPEC.md, current_frontier.yaml, the relevant benchmark
contract and parent manifest, then the failure registry before changing code.

| Path | Role |
|---|---|
| multi_dataset_diverse_rl/search/current_composition.py | Sole current scientific builder |
| multi_dataset_diverse_rl/search/current_layer1.py | Current optimizer/engine/config; generic bounded primitives |
| multi_dataset_diverse_rl/search/legacy/ | Explicit historical search/Pattern/Memory replay |
| multi_dataset_diverse_rl/benchmarks/legacy/ | Historical binding implementations; current binding is flat |
| multi_dataset_diverse_rl/governance/legacy/ | Historical execution/governance tools |
| multi_dataset_diverse_rl/benchmarks/ | Benchmark scientific/data contracts and pinned evaluator provenance |
| multi_dataset_diverse_rl/benchmarks/math_visible_trajectory.py | V6 private profile/provenance and bounded Optimize feedback; shared scoring is unchanged |
| multi_dataset_diverse_rl/benchmarks/math_evidence_binding.py | Sole executable V23 binding and fresh single-arm preparation |
| docs/archive/runtime/a4_pre_v23_only_7342d85/ | Non-importable, hash-indexed original source blobs; original checkout required for replay |
| multi_dataset_diverse_rl/data_preparation/ | Isolated full-source derivation; never an adaptive runtime dependency |
| multi_dataset_diverse_rl/current_contract.py | Closed current identities; no historical dispatch |
| multi_dataset_diverse_rl/versions.py | Historical registry and preserved identity values |
| multi_dataset_diverse_rl/team_search/ | Mixed: shared schemas/evidence/physics are compatibility dependencies; old controllers are historical |
| multi_dataset_diverse_rl/governance/ | Authorization, manifest, registry and offline governance support |
| multi_dataset_diverse_rl/governance/canary_review.py | Read-only continuous Canary evidence barriers; no provider or selection authority |
| multi_dataset_diverse_rl/search/solver_execution.py | Frozen bounded Solver batches and truncation-only capacity; no optimization decision authority |
| multi_dataset_diverse_rl/persistence/solver_evidence_reuse.py | Read-only verified initial realization prefixes, original receipt/ledger references and new-attempt reconstruction |
| experiments/registry.yaml | Experiment metadata authority, explicit eras/kinds |
| experiments/lineage.yaml | Sole lineage authority |
| experiments/manifests/ | Permanent current and historical freeze paths; historical identities not migrated |
| experiments/execution_bindings/ | Versioned model, provider, budget, mechanism and split runtime bindings |
| experiments/schema/ and templates/ | New Unified manifest governance v2 |
| reports/ | Immutable evidence, generated INDEX.md/index.json views |
| docs/archive/ | Historical design/paper text, provenance and invariant index |
| scripts/run_experiment.py | Sole current experiment entrypoint |
| scripts/replay_experiment.py | Explicit historical execution entrypoint |
| scripts/ | Current tooling, historical replay and one-off audits; see tooling_classification.json |
| tests/ | Current contracts, replay and private-artifact replay; see suite_classification.json |
| runs*/ | Ignored local runtime, caches, prep and verification artifacts |

Existing low-level candidate selection, responsibility, vote state, evidence
types and pure transition value types are retained compatibility physics. GEPA controllers and old factories do not enter the current graph. Import guards
forbid historical controllers, experiment runners and MARS controllers from
the current graph. Source closure records actual local import dependencies.

## Explicit test suites

CURRENT_SUITE:
`python tests/formal_zero_api_runner.py tests --suite=current`

FULL_HISTORICAL_SUITE:
`git checkout <original-frozen-source>` then use that checkout's guarded historical replay command.

The current suite excludes explicitly classified retired replay modules before
importing them, including private-artifact modules. `tests/suite_classification.json`
is the complete inventory; the regression report records migration reasons and
actual results. Excluded tests are never counted as PASS. Current V23 wire,
Solver/scoring/recovery, independent evidence, admission, Memory and fresh
accounting checks are retained or ported. Historical source and evidence remain
at original Git commits; current HEAD cannot execute retired replay constructors.

V2.1 flat Gradient replay lives in the explicit `*_v21.py` legacy modules.
`legacy_current_contract_v21.py` freezes their identity imports. The receipt
helpers `gradient_contract_receipt`, `numeric_admissibility_contract`,
`numeric_calibration_contract`, `operational_pilot_contract`,
`partition_completion_contract` and `gradient_recovery_contract` reconstruct
historical V2.1 receipts only. They are absent from the current dependency
graph. The invariant/provenance index is
[the V2.1 replay index](archive/specs/unified_v2_1_replay_index.json).
