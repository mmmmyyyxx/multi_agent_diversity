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

## Current V2.1 method state

The active opt-in method is `unified_team_prompt_search_v2_1`.
Use `SearchMethodConfig.v2_1()` and the shared binary production composition.
Pattern selects one mechanism after WHO, with explicit all-residual coverage
metrics. Deployment requires strict team gain above an immutable initial member
competence floor. Memory stores grounded strategy experience, default disabled.
V1/V2 factories and frozen bindings remain replay identities; current experiments
need a fresh V2.1 binding, preexecution freeze and explicit authorization.
See CURRENT_SPEC for precise rules; fake conformance does not establish efficacy.
Current research benchmarks are MATH, IFBench and HotpotQA. BBH composition
is retained for historical development/replay and structural tests. Canonical
data freeze and experiment-facing split readers are separate benchmark modules;
their access policies precede raw-row resolution. MATH composition uses the
shared `binary_composition.py` graph and benchmark-owned `math_execution.py`
binding. `private_gate.py` owns gate content and `unified_execution.py` owns
source/startup identity, canary phase and single-use authorization. Reference
eligibility preparation is isolated in `data_preparation/`, outside runtime.
Readiness is checked from the permanent versioned manifest; real calls require
a separate explicit authorization. IFBench responsibility and HotpotQA
retrieval remain blockers. Efficacy is unverified.
