# Current Architecture

The sole active research architecture is Unified Team Prompt Search.
Scientific authority: [CURRENT_SPEC](design/CURRENT_SPEC.md).

| Component | Current responsibility |
|---|---|
| UnifiedSearchOrchestrator | Owns the complete production control flow |
| BenchmarkAdapter | Public input, parser, scorer and capability declaration |
| TeamStateSnapshot | Immutable prompts, predictions and evaluated state |
| ResponsibilityAnalyzer / OpportunityBuilder | Diagnosis, feasibility, target and evidence |
| SearchEngine | Candidate exploration; current replaceable derived GEPA |
| AggregationPolicy | Benchmark-selected aggregation and request firewall |
| CandidateEvaluationPipeline | Progressive TeamProbe and Full evidence |
| TransitionPolicy / AdaptiveValidationGate | Winner selection and gate |
| TeamStateCommitter | Atomic deployment |
| HistoryState / MemoryProvider | Structured history; memory defaults to null |
| GlobalStopPolicy | Frozen scientific stopping and operational ceilings |

Implementation lives in `multi_dataset_diverse_rl/search/`, with benchmark
contracts in `multi_dataset_diverse_rl/benchmarks/`. Shared typed compatibility
physics is explicitly retained; historical controllers never own this graph.
The sole current composition CLI is `scripts/run_experiment.py`.

See [repository map](REPOSITORY_MAP.md) for current/historical classifications,
[frontier](../experiments/current_frontier.yaml) for readiness and
[historical architecture](archive/architecture/baseline_architecture.md) for replay.

## Current V2 method state

The active opt-in method is `unified_team_prompt_search_v2`. Team admission is
independent of strict GEPA local survival; evidence sizes are variable. Pattern
and deterministic private/shared Memory are implemented, default disabled.
Use `SearchMethodConfig.v2()` and the V2 composition; the V1 constructor and
adapters remain reproduction paths. See the normative CURRENT_SPEC for policy
boundaries. Mechanism correctness is tested offline; efficacy is unverified.
Benchmark blockers remain unchanged; no new real execution is ready or authorized.
