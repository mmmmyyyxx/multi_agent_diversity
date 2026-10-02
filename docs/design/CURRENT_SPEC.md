# Current Implementation Specification

The sole active research architecture is Unified Team Prompt Search.
This document is the normative active scientific method specification.
Runtime identities come from `multi_dataset_diverse_rl/versions.py`; benchmark
contracts come from `multi_dataset_diverse_rl/benchmarks/`; experiment access
and authorization come from the frozen manifest. Reports are evidence, never
design authority. Historical specifications and invariant IDs are preserved in
`docs/archive/specs/historical_current_spec.md` and its invariant index.

## Active Unified Team Prompt Search V2 (opt-in)

- **INV-UNIFIED-FLOW-001**: One orchestrator owns snapshot, state analysis,
  opportunity construction, candidate search, progressive team evaluation,
  transition selection, adaptive validation, atomic commit, history update and
  global stopping. New method identity is a composition of versioned component
  identities, not a `native/layer2` mode.
- **INV-SEARCH-TRANSITION-001**: A search engine may use the supplied immutable
  opportunity and context to explore candidates. Team transition policy alone
  selects a candidate for write-back. This is a modularity boundary, not a
  restriction that team evidence can never enter future search research.
- **INV-GEPA-DERIVED-001**: V2 uses pinned GEPA public adapter, callback,
  logger and stopper seams. Its strict local improvement, Pareto/frontier,
  parent selection and search lineage remain owned by the untouched GEPA core.
  V1 migration and historical official/Formal baselines retain their identities.
- **INV-EVIDENCE-ROLE-001**: Mutation, search validation, TeamProbe, Full and
  adaptive-gate evidence have distinct role-bearing views or scopes. Historical BBH compatibility
  may use the same Optimize IDs for search validation and TeamProbe. Validation
  and Test never enter adaptive search without separate frozen authorization.
- **INV-RESP-CAPABILITY-001**: Current raw-overlap plurality responsibility,
  raw `V=max(4D,2N,C)` and target score `V/(1+f)` remain unchanged.
  V2 masks only members failing variable-evidence technical feasibility. A benchmark without current-responsibility capability
  fails closed; no plurality diagnostic is fabricated for LLM aggregation.
- **INV-AGGREGATION-001**: Plurality uses one equal vote per valid parsed member
  answer and abstains on top-count ties. The benchmark adapter owns parsing and
  scoring. LLM aggregation uses the optimizer model identity with a separate
  aggregator role, stage, cache key and accounting. Its request contains only
  public input, member outputs, public context and the immutable instruction;
  gold and evaluation results are not part of that interface. Its output is
  parsed again by the benchmark adapter before scoring.
- **INV-HISTORY-MEMORY-001**: Target/failure/commit counts, latest transition
  and lineage are structured optimizer history. Pattern and LLM memory ports
  default to null behavior. No held-out evaluation updates either.
- **INV-REPLAY-001**: The previous two-layer implementation remains available
  for historical reproduction and deterministic migration comparison. It is
  not the new method's ownership graph.

## Active search and stopping contract

V2 preserves strict local search survival, raw responsibility, target ranking,
progressive TeamProbe/Full, promotion budget two, Common-Safe transition,
winner-only Shadow, at most one atomic prompt commit, persistent realizability
and parent-scoped team epochs. It explicitly replaces candidate exposure and
fixed evidence quotas with separately versioned policies. Reflection minibatch
three, local metric budget 36, local no-update patience three and team no-commit
patience two remain unchanged. Public pre-iteration stopping reserves the next
complete pair plus possible Full local validation, so the metric ceiling cannot
overshoot. Resource exhaustion is not scientific convergence.
Only a successful atomic team commit resets team no-update patience. A
Vote-neutral safe commit still counts. Incomplete or emergency-aborted
opportunities do not advance scientific saturation. Operational ceilings
abort execution and must never be reported as convergence.

## V2 mechanism invariants

- **INV-SEARCH-TEAM-ADMISSION-DECOUPLING**: GEPA local survival controls search
  lineage only. Every changed, contract-valid, unique proposal actually reaching
  local Solver evaluation in the current immutable opportunity is eligible for
  outer team evaluation, independently of local delta or final frontier status.
  Accepted and rejected proposals share the TeamProbe -> promotion -> Full ->
  Common-Safe -> winner-only Shadow -> atomic commit pipeline. Team ranking and
  admission never use local scores or GEPA rank. Same-prompt exports appear once;
  accepted lineage metadata takes precedence. Proposal text exists in process
  memory only; callback and public evidence retain hashes, scores and status.
  V2 disables pinned backend persistence and its raw-text logger through supported
  API seams. Exposure adds no proposal or local Solver calls. Export capacity is
  `(metric_budget - validation_size) // (2 * reflection_minibatch_size)`; exceeding
  it fails closed. Proposal exposure, local survival and team commit are distinct
  telemetry. Global stopping reads actual survival, never candidate-pool presence.
- **INV-VARIABLE-EVIDENCE-001**: No 4/4/4 role quotas, exact 12 requirement or
  four-repair minimum apply to V2. Positive raw responsibility, Optimize-only
  mutation-eligible evidence, nonempty validation/probe and executable backend
  budget are necessary. Mutation rows meet the backend unique technical minimum
  three, using deterministic generic backfill only until that minimum. Validation
  may contain 1..capacity rows, with capacity `floor((36 - 4*3)/2)=12`: an
  operational maximum guaranteeing seed validation, two proposal minibatch pairs
  and one accepted full local validation. No fill to capacity occurs. Generic
  priority is primary-lane repair, transition focus, transition anchor, sensitive
  preservation, high-disagreement team-hard; ordinary lower-priority rows are
  technical backfill only. IDs are deduplicated and SHA256/ID ties are stable.
  Mutation and validation/probe are independently bounded and role qualified.
- **INV-PATTERN-DIAGNOSTIC-001**: Pattern diagnosis occurs after raw D/N/C/V,
  failure-discount ranking and target selection. It reads only the selected
  member's supplied Optimize evidence, parent identity and structured past
  history. Mechanisms, missing reasoning steps and corrective principles are
  distinguished from topics/entities. Support partitions target legal residuals;
  risk/counterexample IDs remain inside its Optimize universe. One focus is
  selected by primary-lane support, total support, confidence and stable pattern
  hash. DPR is dominant support / assigned residuals; entropy is normalized to
  [0,1]. Other mechanisms cannot fill mutation evidence except explicit shortage
  backfill to the technical minimum. Pattern may alter composition/reflection,
  never responsibility or target. Provider/model/prompt realization is not frozen;
  the structural prompt contract is frozen and non-null use needs explicit binding.
  Default `null_pattern_v1` adds no provider calls or optional context.
- **INV-LONG-TERM-MEMORY-001**: Successful atomic commits may write target-private
  success; deterministic TeamProbe catastrophe, Full/Common-Safe or Shadow
  rejection may write shared structural risk. Promotion-capacity losers do not
  become risk events. Operational/incomplete outcomes never teach memory. Memory
  stores only abstract fixed principle templates, mechanism hashes, risk codes,
  counts, member/update indices and source hashes; never raw questions, gold,
  outputs, reasoning, prompts or PII. Optimize/search trajectory and adaptive-gate
  structural rejection are allowed; held-out final Validation/Test never update
  memory. Prepare and validate an immutable MemoryDelta before team mutation;
  apply prepared tuples after the complete outcome without provider or I/O. The
  committer receives null memory and owns only atomic team transition. Reads see
  only target-private success and shared risk, ranked by pattern match, lane/risk
  relevance, recency and memory ID. Explicit top-k/context/storage limits enter
  method identity; no real-experiment values are preregistered here. Context
  enters optimizer reflection only, never solver problems, aggregation or scoring.
  Pattern and Memory toggle independently. Default `null_memory_v1` adds no
  context, writes or stateful reads. Non-null mechanisms require manifest-selected
  policies and explicit method identity; no CLI hidden flag may activate them.

V1 identity and reproduction use `SearchMethodConfig()`; V2 method composition
uses `SearchMethodConfig.v2()`. `build_v2_bbh_orchestrator` is a historical
development/replay and structural-test reference. Current multibench execution
composition requires a benchmark-specific preexecution freeze. V1 identities
`unified_team_prompt_search_v1`, `gepa_derived_v1`, `role_view_4_4_4_v1` and
`v4_exact_evidence_feasibility_v1` remain explicit compatibility contracts.
Mechanism correctness does not establish efficacy or real-execution readiness.

## Benchmark interface invariants

- **INV-BENCHMARK-CONTRACT-001**: Each adapter declares its public input,
  immutable output contract, parser, scorer, capabilities and protocol identity.
  Missing required capability fails closed before provider execution.
- **INV-BENCHMARK-FIREWALL-001**: Gold, labels, scorer state and held-out data
  are inaccessible to public solver and aggregator requests. Privacy pipelines
  retain their declared trust boundaries; exceptions fail closed.
- **INV-BENCHMARK-EVALUATOR-001**: External evaluator versions/resources and
  their provenance are frozen dependencies. No download or silent alternate
  scorer is permitted in evaluation. Benchmark protocol implementations and
  IDs are authoritative in the benchmark package and `versions.py`, rather
  than duplicated here.

## Current formal research suite and experiment data

The current formal research suite is **MATH, IFBench, HotpotQA**, in that future
phase order. BBH is a historical development/replay benchmark; its source,
manifests, reports and structural tests remain reproduction evidence. The
unexecuted BBH factorial proposals do not continue as new experiments.

- **INV-EXPERIMENT-SPLIT-001**: `benchmark_data_freeze_v1` is the canonical
  source/provenance layer. Its frozen memberships are immutable. The separate
  `benchmark_experiment_split_v1` derives Optimize150, Shadow300, Validation300
  and Test300 (IFBench Test294) without copying raw rows. Memberships, canonical
  manifest SHA and scientific contract SHA are distinct identities.
- **INV-EXPERIMENT-FIREWALL-001**: Optimize alone supplies responsibility,
  target, Pattern, Memory, GEPA and TeamProbe/Full evidence. Shadow is an adaptive
  gate with aggregate score/pass-fail feedback. Validation is post-freeze,
  read-only development evaluation; it never changes prompts, selection, stops,
  tuning or memory. Test remains sealed and requires separate explicit unlock.
  Search readers expose held-out metadata only, never raw held-out rows.
- **INV-EXPERIMENT-MODEL-001**: The versioned current experiment binding fixes
  all five members, A1/A2/A3/A4, seeds and adaptive re-evaluations to Solver
  `qwen3-8b`. Optimizer/GEPA reflection and Pattern use `qwen3.7-flash`. No
  benchmark/arm/seed may override Solver; compatibility CLI defaults are not
  experiment authority. Future execution must bind this policy in its identity.
- **INV-MULTIBENCH-READINESS-001**: MATH retains pinned equivalence plurality
  and binary responsibility; HotpotQA retains normalized plurality and binary
  responsibility with question-only retrieval. IFBench retains LLM equal raw
  responses and fails closed on unfrozen aggregation-aware responsibility.
  Its checker package/resource lock is verified offline in the bound environment.
  HotpotQA requires corpus, index, retriever and two-hop integration; supplied
  gold context is not a substitute. No benchmark-specific logic belongs in the
  orchestrator, Pattern/Memory cores or GEPA core.

All memberships precede any efficacy observation. Initial task-valid five-member
prompts, provider/resource policy, V2 production composition and authorization
require future benchmark-specific preexecution freeze. BBH prompts are not
inherited. Pattern/Memory are IMPLEMENTED_DEFAULT_OFF, efficacy unverified.

## MATH reference eligibility and versioned runtime binding

- **INV-MATH-REFERENCE-VALIDITY-001**: `MATH_REFERENCE_VALIDITY_V1` admits a
  correctness-dependent example only when the unchanged frozen deterministic
  source extractor returns a nonempty answer after stripping whitespace.
  Invalid references are data exclusions, never Solver errors. Eligibility is
  applied to all canonical source rows before the unchanged seed, proportional
  subject quotas, largest remainder, lexicographic ties and stable SHA ordering.
  MATH uses `benchmark_experiment_split_v1_1`; prior memberships and canonical
  source bytes remain immutable. Counts and source roles are unchanged.
  Full-source preparation runs in an isolated `DATA_PREPARATION_CONTEXT`,
  imports no optimizer/provider runtime and emits only metadata and hashes.
- **INV-MATH-RUNTIME-BINDING-001**: MATH production readiness comes from the
  explicit execution binding, checked canonical/split/initial-team hashes,
  evaluator pins, aggregation/responsibility identities and shared model policy.
  Static registry readiness alone cannot grant execution. The shared binary
  ports use the benchmark's actual aggregation and check agreement with the
  responsibility algebra. Common-Safe uses the existing monotone-safe S0-S2
  key; local scores never enter team ranking. Shadow cardinality is manifest
  driven, preserving its nonnegative Vote and target-loss-at-most-two guards.
  Shadow raw records/cache stay within a private gate capability. Successful
  commits write private success only; newly-broken counts cannot teach shared
  risk. A committed winner is excluded from rejection risk even if stale
  rejection diagnostics are present.
  The fixed GEPA evidence schedule replays complete epochs so perfect-score
  skips cannot exhaust it before frozen patience. Metric reserves and ceilings
  remain unchanged. Initial prompts are generic mutable reasoning only; the
  immutable output interface remains separate.
- **INV-MATH-OUTPUT-INTERFACE-002**: MATH Solver requests bind the versioned
  benchmark-owned immutable formatting interface in the system message for
  initial, local, candidate and committed prompts, independent of per-item
  wording. Mutable prompt identity excludes that interface; effective request
  identity and exact-request cache identity include its version and wording
  hash. Reflection evolves only mutable reasoning. The strict final-line
  parser, pinned equivalence evaluator and mutable contamination guard remain
  unchanged. Format noncompliance aborts without regeneration; fake and real
  providers share this parser path. Interface repairs establish request
  conformance, never real model adherence or scientific efficacy.
- Canary readiness is scoped to a preregistered first parent team epoch or its
  first atomic commit. This phase boundary does not redefine scientific
  saturation or patience. Its finite opportunity bound follows from integer
  positive raw V and failure-discount ranking; provider/transport bounds follow
  from metric/proposal/promotion/split capacities. A phase boundary or resource
  ceiling must never be reported as scientific convergence.
  Readiness requires a frozen source/config/startup identity and gives no API
  authorization. Only an exact single-use attempt authorization can construct
  the real provider. Validation remains unauthorized and Test sealed.

## Operational reservation accounting (explicit opt-in)

- **INV-ACCOUNTING-RESERVATION-002**: A versioned, explicitly authorized
  accounting policy may reserve the exact serialized provider-visible UTF-8
  request length plus its declared operational margin and transmitted output
  cap. This is an operational accounting bound, not provider-billing proof.
  Every physical transport, including retries, reserves durably before sending.
  Reliable usage releases the reservation and charges reported input/output;
  missing, invalid or untrusted usage charges the entire reservation. Charges
  never decrease. Recovery charges unresolved reservations in full. An OS lock,
  hash-chained journal and derived snapshot enforce single-owner accounting.
  Budget stops and output truncation are operational failures, never scientific
  convergence or valid outputs. Such failures cannot be swallowed by GEPA.
- **INV-VALIDATION-ACCOUNTING-PREP-002**: An explicitly authorized isolated
  ACCOUNTING_DATA_PREP_CONTEXT may project frozen Validation problems solely
  to serialized-request byte metadata. It exposes no content to search, calls
  no models, evaluates no correctness and emits no problems, gold or solutions.
  Search readers still reject held-out raw access. Validation's initial/final
  evaluation reserve is protected before each search transport; observed
  candidate prompt lengths can only increase its protected envelope. Test raw
  rows remain sealed. Actual Validation evaluation requires separate phase
  admission after an immutable SEARCH_COMPLETE_RECEIPT and cannot resume or
  write to search.

## Authority map

See AGENTS.md for the complete authority hierarchy; method.md is a conceptual
overview, CURRENT_ARCHITECTURE.md a code map, and reports/ immutable evidence.
Current implementation does not imply real-execution readiness or authorization.
