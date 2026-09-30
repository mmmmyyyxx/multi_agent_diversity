# Unified Team Prompt Search refactor, zero API

`BASE_SHA=a639e4402c1e314c19e5e5f782c414f4da5cf4f6`.
This is an architecture migration, not a new efficacy result. The tracked
worktree was clean at start; an unrelated untracked retry report was left
untouched. No real API, Validation50, or Test50 call was made.

## Method and control flow

The old Layer1/Layer2 division was retired as the new method's ownership
boundary because responsibility, evidence, GEPA search, local validation,
team evaluation, admission, history, and stopping form one adaptive loop. The
new flow is `BenchmarkAdapter -> TeamStateSnapshot -> StateAnalyzer ->
OpportunityBuilder -> SearchEngine -> CandidateEvaluationPipeline ->
TransitionPolicy -> AdaptiveValidationGate -> Committer -> History/Memory ->
GlobalStopPolicy`. `run_experiment` dispatches the new method to this loop.
Historical Formal V3 retains its old executable path and exact identity.

GEPA is a replaceable, versioned SearchEngine. It is **not** required to stay
immutable in future main-method versions: sampler, validation, reflection,
archive, acceptance, and candidate selection may change only under a new
scientific identity. The pinned official GEPA v0.1.1 baseline remains a
separate historical control. Current BBH search keeps strict GEPA local
acceptance for behavioral equivalence. Candidate search proposes and locally
compares prompts; Common-Safe separately selects a team transition after
TeamProbe and Full, with winner-only Shadow before atomic commit.

Current D/N/C is in `search.current_bbh.PluralityResponsibilityAnalyzer`.
Target ranking is a separate policy using raw `V=max(4D,2N,C)` and
`V/(1+f)` after feasibility masking. `EvidenceView` separates mutation,
search validation, TeamProbe, Full scope, and adaptive gate scope. Search
validation and TeamProbe may use the same Optimize IDs under current V4, but
they remain different roles. Pattern analysis enters `StateAnalyzer`; memory
enters the opportunity context and observes accepted transitions. Both are
null today. Structured history contains failure counts, accepted transitions,
and lineage; it is not an LLM memory store.

## Aggregation

Plurality gives one vote to each valid member and abstains on a top-count tie.
The fake LLM aggregation path sends public problem text and five equal-status
member outputs to an independent `aggregator` role. The aggregation model is
read from `RuntimeContext.optimizer_model`, matching the optimizer model
identity while retaining a separate role, request/cache identity, and
accounting stage. The request has no gold, reward, correctness vector, or
evaluation result. The benchmark adapter parses the aggregator's final text;
only then does evaluation use gold. Aggregator calls and tokens are recorded
separately from member Solver calls in row-level accounting.

BBH supplies plurality and boolean correctness capabilities. Free-form tasks
may use LLM aggregation for inference, but plurality-specific D/N/C is never
applied to them: search fails closed with `SCIENTIFIC_DECISION_REQUIRED` until
an aggregation-aware responsibility policy is preregistered. No Pattern or
Memory mechanism was activated.

## Migration evidence and scope

The zero-API golden fixture compares the historical controller with the new
current BBH composition. It matches raw responsibility, target, evidence IDs,
the complete GEPA packet hash and schedule, fake official GEPA calls and
returned prompts, TeamProbe/Full/Shadow order, commit, and final team hash.
The Shadow rejection path matches; the successor packet after an accepted
transition also matches. The full saturation trajectory was not replayed.

The new graph reaches low-level pinned BBH evaluator and atomic committer
through a compatibility adapter. It does not call the old controller, old
task builder, or historical seed runners. The CLI's new real-API route remains
closed pending a separately frozen execution prep and authorization.

See `scientific_conflict_inventory.md` for the unresolved search, evidence,
objective, stopping, history, pattern, and aggregation choices. The method
did not change any of those rules based on observed results.

`UNIFIED_ARCHITECTURE_READY = YES` for the zero-API architecture migration.
Real-API readiness remains `HOLD_PRE_PROVIDER` until a new frozen execution
prep and explicit authorization exist. The unfiltered historical test suite
still requires private frozen artifacts absent from this worktree; those
artifact-dependent failures are listed in `test_summary.json`.
