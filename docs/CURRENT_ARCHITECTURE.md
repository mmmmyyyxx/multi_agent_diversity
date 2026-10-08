# Current Architecture

The sole active research architecture is Unified Team Prompt Search.
Scientific authority: [CURRENT_SPEC](design/CURRENT_SPEC.md).

| Component | Current responsibility |
|---|---|
| UnifiedSearchOrchestrator | Owns the complete production control flow |
| BenchmarkAdapter | Public input, parser, scorer and capability declaration |
| TeamStateSnapshot | Immutable prompts, predictions and evaluated state |
| ResponsibilityAnalyzer / OpportunityBuilder | Diagnosis, feasibility, target and evidence |
| SearchEngine | Current bounded Gradient Layer1 candidate exploration |
| AggregationPolicy | Benchmark-selected aggregation and request firewall |
| CandidateEvaluationPipeline | Progressive TeamProbe and Full evidence |
| TransitionPolicy / AdaptiveValidationGate | Winner selection and gate |
| TeamStateCommitter | Atomic deployment |
| HistoryState / MemoryProvider | Separate history and bounded Rolling Risk Memory |
| GlobalStopPolicy | Frozen scientific stopping and operational ceilings |

Implementation lives in `multi_dataset_diverse_rl/search/`, with benchmark
contracts in `multi_dataset_diverse_rl/benchmarks/`. Shared typed compatibility
physics is explicitly retained; historical controllers never own this graph.
The sole current composition CLI is `scripts/run_experiment.py`.

See [repository map](REPOSITORY_MAP.md) for current/historical classifications,
[frontier](../experiments/current_frontier.yaml) for readiness and
[historical architecture](archive/architecture/baseline_architecture.md) for replay.

## Current V2.3 method state

V2.3 is the sole active scientific method; V2.2 and paired execution are permanently retired.
The production entrypoint is `scripts/run_experiment.py`, using `MATHEvidenceBinding`,
the complete current policy bundle and `build_current_team_prompt_search`.
Missing or historical method, evidence and trajectory identities fail closed.

Five qwen3-8b members use V6 ordinary visible mathematical steps and one final
answer line, with thinking disabled. qwen3.7-flash supplies independent
trajectory/reference Gradients, gradients-only clustering and conservative
Layer1 edits. Equal-weight equivalence plurality and fixed peers remain.
Mutation, SearchValidation and TeamProbe are disjoint Optimize memberships.
Layer1 keeps six generations, 42 metric calls and four exports; at most two
candidates reach Full. The immutable initial competence floor, nonregressing
Full Vote, strict target OR team progress and winner-only Shadow gate remain.
Measured initial competence and edit effects enter bounded Memory; uncommitted
candidate coverage cannot replace committed competence. Team-epoch no-commit
patience remains two. Operational horizon/budget stops do not prove saturation.

Real-model adherence and efficacy remain unverified. Fresh source, manifest,
request, attempt, cache and ledger identities precede exact single-use API
approval. Historical authorizations grant no new access. Validation is not
authorized and Test remains sealed. Current execution uses MATH; IFBench and
HotpotQA contracts remain maintained, with their execution blockers intact.
