# Repository Map

Current authority: AGENTS.md, docs/design/CURRENT_SPEC.md and the
[sole Hendrycks MATH correctness contract](math_scoring_cevolve_hendrycks_v1.md).
Search retains [V2.12 count responsibility](design/COUNT_RESPONSIBILITY_V212.md)
and [V2.9 population search](design/VOTE_GUIDED_PARENT_POPULATION_V29.md).

| Path | Role |
|---|---|
| multi_dataset_diverse_rl/math_scoring/ | Sole current MATH correctness: pinned Hendrycks extraction and finite normalization, versioned as cevolve_hendrycks_math_v1 |
| multi_dataset_diverse_rl/search/system_prompt.py | Sole structured member prompt state and block identity |
| multi_dataset_diverse_rl/research/single_prompt*.py | Explicit independent 20-round mechanism study; outside current team composition |
| multi_dataset_diverse_rl/research/ablations/ | Isolated independent single-Prompt kernel and adapters importing formal modules; repeat-scoped, offline preparation only |
| scripts/ablations/study.py | Thin isolated draft/synthetic CLI; cannot authorize real execution |
| experiments/ablations/pattern/default.json | Editable seed42 preparation defaults, outside frozen historical manifests |
| multi_dataset_diverse_rl/search/current_composition.py | Sole production graph |
| multi_dataset_diverse_rl/search/parent_population.py | Bounded per-member fitness and independent sampling |
| multi_dataset_diverse_rl/search/population_evaluation.py | Complete Optimize fitness and joint final assembly |
| multi_dataset_diverse_rl/search/population_checkpoint.py | Atomic complete-boundary state recovery; no API grant |
| multi_dataset_diverse_rl/search/population_memory.py | Scoped single-step measured edit experience |
| multi_dataset_diverse_rl/search/weighted_responsibility.py | Current count responsibility; historical module name retained |
| multi_dataset_diverse_rl/search/optimizer_output_contract.py | Strict optimizer types and source-free bounded retry feedback |
| multi_dataset_diverse_rl/current_contract.py | See package-root current_contract.py for closed identity imports |
| multi_dataset_diverse_rl/benchmarks/math_structured_binding.py | Fresh current MATH freeze and composition |
| multi_dataset_diverse_rl/benchmarks/math_flexible_answer.py | Current gold-blind Hendrycks boxed extraction adapter |
| multi_dataset_diverse_rl/benchmarks/math_structured_answer.py | Thin current extraction adapter; predecessor source is archived |
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

V2.11 current auxiliary group diagnosis and verified Reflection projection live in
`search/group_diagnosis.py` and `search/reflector_view.py`.
[Scientific contract](design/GROUP_DIAGNOSED_REFLECTION_V211.md); historical
V2.10 source is recoverable at `d7de59818da609cff6b03c6ebb334fa79011aedd`.


Current MATH correctness: `multi_dataset_diverse_rl/math_scoring/` is the sole
`cevolve_hendrycks_math_v1` implementation. Benchmark and single-prompt adapters
delegate to it. In current execution MathVerify is aggregation-only; retained offline
data-census utilities are historical provenance analysis, not correctness judges. Retired scoring source and tests
are inventoried in `docs/archive/scoring/pre_hendrycks_20261021/`; they cannot
execute as current. Historical V1 reports describe the earlier opt-in audit.
