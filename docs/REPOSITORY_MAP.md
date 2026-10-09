# Repository Map

Current authority: AGENTS.md, docs/design/CURRENT_SPEC.md and
[Responsibility Fallback V2.5](design/RESPONSIBILITY_FALLBACK_REPAIR_V25.md).

| Path | Role |
|---|---|
| multi_dataset_diverse_rl/search/system_prompt.py | Sole structured member prompt state and block identity |
| multi_dataset_diverse_rl/search/current_composition.py | Sole production graph |
| multi_dataset_diverse_rl/current_contract.py | See package-root current_contract.py for closed identity imports |
| multi_dataset_diverse_rl/benchmarks/math_structured_binding.py | Fresh V2.5 freeze and composition |
| multi_dataset_diverse_rl/benchmarks/math_structured_answer.py | Frozen gold-blind extraction |
| multi_dataset_diverse_rl/benchmarks/math_response_evidence.py | Private ordinary content with optional written steps |
| multi_dataset_diverse_rl/governance/ | Authorization, source/startup freeze, accounting and registries |
| multi_dataset_diverse_rl/persistence/ | Durable receipts, attempt-isolated cache and journals |
| multi_dataset_diverse_rl/team_search/ | Shared value types and evidence physics; no current historical controller |
| experiments/manifests/ | Permanent immutable historical and fresh current experiment contracts |
| experiments/execution_bindings/ | Current V2.4 bindings and non-executable historical evidence |
| reports/ | Immutable scientific evidence; current cleanup inventory and tombstone |
| scripts/run_experiment.py | Sole current runner |
| tests/suite_classification.json | Current, excluded and Git-only historical test inventory |
| runs/ | Ignored private runtime artifacts, caches and verification logs |

Retired runtime and compatibility code is recovered from its pinned Git source.
Raw evidence, billing journals, manifests, scientific reports and unknown unique
artifacts remain protected. The current closure includes no retired namespace.
Run the current suite with `python tests/formal_zero_api_runner.py tests --suite=current`.
Historical/private-artifact tests require the original frozen source and assets;
excluded or retired modules are not claimed to pass in the current checkout.
