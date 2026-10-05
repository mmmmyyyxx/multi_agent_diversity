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
| multi_dataset_diverse_rl/data_preparation/ | Isolated full-source derivation; never an adaptive runtime dependency |
| multi_dataset_diverse_rl/current_contract.py | Closed current identities; no historical dispatch |
| multi_dataset_diverse_rl/versions.py | Historical registry and preserved identity values |
| multi_dataset_diverse_rl/team_search/ | Mixed: shared schemas/evidence/physics are compatibility dependencies; old controllers are historical |
| multi_dataset_diverse_rl/governance/ | Authorization, manifest, registry and offline governance support |
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
types and GEPA adapter imports are retained compatibility physics. Import guards
forbid historical controllers, experiment runners and MARS controllers from
the current graph. Source closure records actual local import dependencies.

## Explicit test suites

CURRENT_SUITE:
`python tests/formal_zero_api_runner.py tests --suite=current`

FULL_HISTORICAL_SUITE:
`python tests/formal_zero_api_runner.py tests --suite=full-historical`

The current suite excludes the named private-artifact modules and reports their
count. Full historical runs them when private frozen assets exist, otherwise
skips them explicitly. Historical source remains at its reproduction paths;
archiving documentation does not imply HEAD replay has private assets.
