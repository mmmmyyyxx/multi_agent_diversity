# Phase-B single-state multi-member Local GEPA pilot v2

Status: preregistered zero-API freeze. API authorization is pending and
`READY_TO_RUN=false`. Phase A is complete and immutable. This protocol neither
reacquires parents nor executes GEPA.

## Scientific scope

This is a single-baseline-state, multi-member, Layer-1 GEPA acceptance-rate
pilot. It describes official GEPA local search behavior on four member-specific
`LocalOptimizationTask` objects within one fixed team state. The shared source
state is `faa0fc81ebe71a686554355b9c5946a1765fa3913e026ab65e59f07ae01052dd`;
the frozen targets are members 1, 2, 3 and 4. All have the `coverage` primary
lane. Exact task, prompt, search, local-validation and context hashes are bound
by `PARENT_FREEZE.json` and replayed against the private Phase-A task bundle
before any future provider construction.

Allowed conclusions concern member/task heterogeneity inside this one baseline
state. This pilot does not estimate cross-state GEPA effectiveness, a
population-wide acceptance probability, scheduler effectiveness or team-level
transfer. `TEAM_TRANSFER_NOT_EVALUATED` remains true for every outcome.

Phase A remains separate: 100 successful acquisition Solver calls, 19,665
prompt tokens, 6,448 completion tokens, 26,113 total tokens, five eligible
parents, zero Reflection/GEPA/held-out calls, and deterministic selection of
members 1-4. These calls and tokens never enter a Phase-B proposal denominator
or Layer-1 optimization cost.

## Immutable parent selection

Do not reacquire, replace or reselect a parent. The fifth eligible task,
`seed78_update0_member0`, is unselected because all five candidates tie on one
source state, four distinct targets and one primary lane, and the preregistered
minimum stable subset-SHA256 tie-break selected members 1-4. It remains recorded
in the selection audit and may only enter a separately preregistered replication.
Phase-B outcomes cannot change this choice.

Before any future provider call, verify the Phase-A source report manifest,
private task-bundle hash, exact four IDs, source-state equality, member IDs,
parent prompt hashes, primary lanes, task payload hashes, ordered search and
local-validation identity/payload hashes, optimization-context hashes and task
seeds. Any mismatch fails closed.

## Official GEPA and capacity

Run pinned official GEPA v0.1.1 through the supported Level-B adapter. Preserve
the official search core, Pareto population and parent selection, reflection
template and dataset, component contract, sampling, scoring, strict acceptance,
skip-perfect behavior and evolved-candidate lineage. Never reset to the root
after a proposal. Telemetry observes the actual selected parent hash and depth;
it does not select or accept candidates.

Each task has a quota of eight actual official `on_proposal_end` events: four
parents times eight events equals 32 planned proposal events. Loop iterations
are not the quota. The supported event stopper enforces `proposal_end <= 8`
independently per task. Normal completion requires exactly eight. At most 16
skipped iterations are allowed per task. Metric capacity is frozen at 205 rows
per task: 12 seed-validation rows + 8*(2*3 sampled rows + 12 possible accepted
full-validation rows) + 16*3 skip rows + one boundary headroom row. A pre-batch
guard prevents overshoot. A perfect-skip path, provider/protocol/persistence
failure, or budget boundary can yield fewer than eight events; classify it as
`PARENT_PROPOSAL_QUOTA_INCOMPLETE`, preserve evidence, do not replace, resume or
adapt it, and do not declare a completed four-parent pilot estimate.

## Outcomes and telemetry

The primary pooled descriptive estimand is total official accepted mutations
divided by contract-valid changed proposals reaching empirical Solver
evaluation. Duplicate changed events remain; unchanged or contract-invalid
events do not. Cache-backed exact-request evidence counts as Solver-reached but
is separately marked from provider calls. Report the same rate for every member
task and report the fraction of the four parents with at least one acceptance.

The 32 adaptively generated outcomes are not IID Bernoulli trials: eight events
share each task, later proposals may use evolved parents, and all four tasks
share one team state. No inferential confidence interval appears in the main
conclusion. If Wilson 95% is retained in machine output, label it only
`NAIVE_IID_REFERENCE_ONLY` and keep it secondary.

Every Solver-reached proposal records only sanitized identities and numeric
diagnostics: proposal index, task/member ID, selected-parent hash, proposal hash,
parent/proposal lineage depth, sampled parent/proposal local scores and delta,
newly fixed/broken totals and responsibility/coalition/preservation components,
edit similarity and token counts, dominant reflection-failure group/lane and
concentration, sampled-minibatch ID overlap, and official acceptance. Never
persist prompt/proposal text, question text, gold/model answers, traces or
reflection content in public artifacts.

Every empirical proposal is classified separately as `STRICT_POSITIVE`,
`EXACT_EQUAL` or `STRICT_NEGATIVE`. An exact-equal proposal is further classified
as `BEHAVIORAL_NO_OP` when fixed=broken=0, or
`REPAIR_PRESERVATION_CANCELLATION` when fixed=broken>0. Equality is never merged
with degradation.

For each task report attempts, contract-valid, Solver-reached, positive/equal/
negative, accepted, total fixed/broken, preservation losses, mean/median edit
similarity, unique failure patterns, dominant-pattern concentration, maximum
lineage depth and accepted-generation count. Pooled summaries cannot replace
these four task summaries.

## Interpretation matrix

- Acceptance across multiple parents:
  `LOCAL_GEPA_STRICT_IMPROVEMENT_OBSERVED_ACROSS_MULTIPLE_MEMBER_TASKS`.
- Acceptance concentrated in one parent: `STRONG_MEMBER_TASK_HETEROGENEITY`.
- Mostly exact-equal behavioral no-ops: `LOW_BEHAVIORAL_EFFECT_OF_PROPOSALS`.
- Fixed and broken both common: `LOCAL_REPAIR_PRESERVATION_TRADEOFF`.
- High similarity with concentrated patterns: `SEARCH_EXPLORATION_CONCENTRATION`.
- Diverse edits/patterns with near-zero deltas:
  `LOCAL_OBJECTIVE_OR_FEEDBACK_LIMITATION`.

These are diagnostic labels, not automatic method changes. Observed results
must not change the GEPA core, reflection template/dataset, proposal contract,
local-validation size, scheduler, TeamMiniBatch or Common-Safe semantics. Any
such change needs a separate analysis and preregistration.

## Isolation, costs and freeze

Phase B ends at Layer-1 local optimization. TeamMiniBatch evaluation, full-team
evaluation, Common-Safe, Shadow, write-back and persistent-realizability updates
are all zero. Search and local validation use only the frozen Optimize evidence;
Validation50 and Test50 calls are zero.

Phase-B cost is reported separately as Reflection cost and local Solver cost.
Hard ceilings are 32 Reflection logical calls, 820 local metric rows, 3,280
Solver transport attempts under the four-attempt contract, 1,476,000 successful
Solver output tokens at 1,800 per metric row, and 57,600 Reflection output tokens
at 1,800 per call. Input-token and currency maxima are not asserted without a
provider tokenizer/price schedule. Failed transport-attempt token usage may be
unreported and must be disclosed. The Phase-A 26,113 tokens remain a third,
separate accounting category.

The formal run root is absent. No API call is authorized by this freeze.
Execution requires a new explicit Phase-B authorization plus a clean committed
source freeze and executable Sol-to-Luna handoff. Until then:

```text
Phase B preregistered
exact 4 parents frozen
proposal quota and cost ceilings frozen
API calls = 0
Validation50/Test50 = 0
authorization = pending
READY_TO_RUN = false
```
