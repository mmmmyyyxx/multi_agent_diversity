# Repository Map

Current authority: AGENTS.md, docs/design/CURRENT_SPEC.md and
[Responsibility Fallback V2.5](design/RESPONSIBILITY_FALLBACK_REPAIR_V25.md) with
[flexible extraction V3](design/FLEXIBLE_ANSWER_EXTRACTION_V3.md).
The current Layer1 authority is [V2.9 population search](design/VOTE_GUIDED_PARENT_POPULATION_V29.md).
Current WHO/Pattern responsibility is defined by
[V2.10 exclusive weighted errors](design/WEIGHTED_ERROR_RESPONSIBILITY_V210.md).

| Path | Role |
|---|---|
| multi_dataset_diverse_rl/search/system_prompt.py | Sole structured member prompt state and block identity |
| multi_dataset_diverse_rl/search/current_composition.py | Sole production graph |
| multi_dataset_diverse_rl/search/parent_population.py | Bounded per-member fitness and independent sampling |
| multi_dataset_diverse_rl/search/population_evaluation.py | Complete Optimize fitness and joint final assembly |
| multi_dataset_diverse_rl/search/population_checkpoint.py | Atomic complete-boundary state recovery; no API grant |
| multi_dataset_diverse_rl/search/population_memory.py | Scoped single-step measured edit experience |
| multi_dataset_diverse_rl/search/weighted_responsibility.py | Exclusive D/N/C/R3/R4 labels and shared summed weights |
| multi_dataset_diverse_rl/search/optimizer_output_contract.py | Strict optimizer types and source-free bounded retry feedback |
| multi_dataset_diverse_rl/current_contract.py | See package-root current_contract.py for closed identity imports |
| multi_dataset_diverse_rl/benchmarks/math_structured_binding.py | Fresh V2.5 freeze and composition |
| multi_dataset_diverse_rl/benchmarks/math_flexible_answer.py | Current gold-blind flexible extraction |
| multi_dataset_diverse_rl/benchmarks/math_structured_answer.py | Frozen V2 baseline replay parser; not current extraction |
| multi_dataset_diverse_rl/benchmarks/math_response_evidence.py | Private ordinary content with optional written steps |
| multi_dataset_diverse_rl/governance/ | Authorization, source/startup freeze, accounting and registries |
| multi_dataset_diverse_rl/benchmarks/math_formal_scope.py | Versioned formal resource bound and metadata-only memberships |
| multi_dataset_diverse_rl/governance/cost_accounting.py | Supplier tariff validation and integer monetary reservations |
| multi_dataset_diverse_rl/governance/canary_progression.py | Engineering-only Canary prerequisite for frozen formal search |
| multi_dataset_diverse_rl/governance/postfreeze_validation.py | Separate single-use post-search Solver-only Validation capability |
| multi_dataset_diverse_rl/governance/paired_validation_analysis.py | Frozen paired team measurements; no search read point |
| multi_dataset_diverse_rl/persistence/ | Durable receipts, attempt-isolated cache and journals |
| multi_dataset_diverse_rl/team_search/ | Shared value types and evidence physics; no current historical controller |
| experiments/manifests/ | Permanent immutable historical and fresh current experiment contracts |
| experiments/execution_bindings/ | Current versioned bindings and non-executable historical evidence |
| multi_dataset_diverse_rl/diagnostics/pilot_observations.py | Offline opportunity streaks and Gradient failure/cost observations; no runtime read point |
| multi_dataset_diverse_rl/diagnostics/gradient_threshold_replay.py | Offline two-field counterfactual isolation and local retry accounting; no runtime policy |
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

Current diagnostic limits and identity amendment: [Gradient 400 V2](design/GRADIENT_DIAGNOSTIC_LIMITS_400_V2.md).

Current readable evidence and retained parallel timing contract: [Direct readable V2](design/DIRECT_READABLE_OPTIMIZER_EVIDENCE_V2.md).
The interval codec `search/optimizer_encoding.py` is an offline historical analysis utility, outside current runtime; its original execution source is commit `74e2e0b7818b4cc307e6bcf0aa7d713558113f91`.
