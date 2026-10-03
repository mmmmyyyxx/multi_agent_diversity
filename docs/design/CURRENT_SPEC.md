# Current Implementation Specification

The sole active research architecture is Unified Team Prompt Search.
This document is the normative active scientific method specification.
Runtime identities come from `multi_dataset_diverse_rl/versions.py`; benchmark
contracts come from `multi_dataset_diverse_rl/benchmarks/`; experiment access
and authorization come from the frozen manifest. Reports are evidence, never
design authority. Historical specifications and invariant IDs are preserved in
`docs/archive/specs/historical_current_spec.md` and its invariant index.

## Active Unified Team Prompt Search V2.1 (opt-in)

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
  V2.1 masks only members failing variable-evidence technical feasibility. A benchmark without current-responsibility capability
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

The user-authored [method semantic contract](UNIFIED_METHOD_SEMANTIC_CONTRACT.md)
defines the research meaning and claim boundaries. `SearchMethodConfig.v2_1()`
implements it with fresh component identities. V1 and V2 factories retain their
replay behavior; prior frozen MATH bindings remain V2 and confer no V2.1 access.

Responsibility remains resource allocation: raw overlapping D/N/C,
`V=max(4D,2N,C)` and target score `V/(1+f)` are unchanged. Target selection
uses all currently feasible members at every opportunity; it never removes a
previously selected member to enforce equal quotas or round-robin. Parent epochs
track exposure of the feasible set for stopping, not a once-per-member schedule.
Positive integer V with failure discount gives eventual exposure in a stationary
no-commit parent; after sufficient exposure, allocation may remain concentrated.

GEPA strict local survival, progressive TeamProbe/Full, promotion ceiling two,
winner-only Shadow, atomic single-member commit and parent-scoped epochs remain.
Reflection minibatch three, local metric budget36, local no-update patience three
and team no-commit patience two are unchanged. A successful commit resets team
patience; V2.1 commits require strict Full team gain. Resource/budget exhaustion,
incomplete outcomes and technical/conformance failures are not convergence.
Shadow retains its separately frozen nonnegative team and bounded target-loss
guard; the competence floor is measured on the complete Optimize scope.

## V2.1 mechanism invariants

- **INV-SEARCH-TEAM-ADMISSION-DECOUPLING**: GEPA survival controls search lineage
  only. All changed, valid, unique, Solver-evaluated proposals enter outer team
  evaluation, including rejected proposals and non-frontier candidates. Local
  scores never rank deployment. Supported GEPA seams disable raw-text logging
  and persistence; export adds no calls. Frozen metric reserves and dedup remain.
- **INV-VARIABLE-EVIDENCE-001**: No exact4/4/4 or12 quotas apply. Optimize-only
  mutation evidence meets the backend unique minimum three; validation/probe
  capacity remains `floor((36-4*3)/2)=12`, without filling to the maximum. With
  Pattern off, generic V2 variable evidence remains. With Pattern on, ONLY focus
  support supplies repair. Declared focus counterexamples/risks and genuine
  non-repair preservation/transition/team context supply boundary/safety. Another
  residual's repair cannot be relabeled as technical backfill. All boundary
  reflection records explicitly prohibit a second repair objective, even when
  their observed outcome is wrong. Insufficient legal unique evidence fails as
  `FOCUSED_BACKEND_MINIMUM_WITHOUT_LEGAL_BOUNDARIES`; no mixed-repair fallback.
- **INV-PATTERN-DIAGNOSTIC-001**: Diagnosis follows target selection. Pattern
  never chooses WHO or changes raw responsibility/failure discount. Every valid
  non-null diagnosis selects one focus by primary-lane overlap, support,
  confidence and stable structural identity, independent of concentration.
  Empty support partitions fail conformance, without generic fallback. Report
  `DPR_all=focus/all legal residuals`, `ConditionalDPR=focus/assigned residuals`,
  `Coverage=assigned/all legal residuals` and normalized entropy separately.
  Structural identity hashes normalized failure mechanism AND corrective
  principle, not provider cluster labels; identical normalized descriptions
  merge. This is deterministic description identity, not a paraphrase resolver.
  Provider/model/prompt/limits require a fresh explicit experiment binding.
- **INV-INITIAL-COMPETENCE-002**: Freeze the initial five-member Optimize
  scores and state identity from actual initialization. Candidate Full target
  competence must be at least its INITIAL score and team score must strictly
  improve; terminal-invalid delta must be measured and nonpositive. A member
  may decline relative to incumbent while staying above initial competence.
  Team-neutral/down or below-initial changes never deploy. Initial floors
  survive commit/rollback and cannot be rebased. Missing floors/guards fail
  closed. Safety tie-breaking never uses GEPA local scores.
- **INV-LONG-TERM-MEMORY-001**: Memory contains Situation–Action–Outcome–Lesson
  strategy experience, not a telemetry log. The deterministic closed action
  vocabulary recognizes added/removed reasoning checks from the ACTUAL parent
  to candidate edit. Unknown edits teach no invented generic principle. It
  projects mechanism descriptions into safe procedure concepts; stores no raw
  questions, gold, outputs, reasoning, prompts, entities or provider labels.
  Success means a completed atomic team deployment and belongs only to its
  target's private experience. Shared entries contain only measured structural
  rejection and the failed action to avoid, not another member's successful
  strategy. Promotion-capacity losers and operational/incomplete outcomes teach
  nothing. Outcome scope distinguishes Full from TeamProbe; absent fixed/broken
  counts remain unknown rather than fabricated zero. Lessons state applicability
  and single-observation limits, not causal or generalization proof. Prepare and
  validate before commit, apply immutable tuples after success; top-k/context/
  storage limits are explicit identities. Reads occur AFTER WHO/focus selection
  and enter reflection only. Provenance remains in storage/audit while bounded
  contexts carry the complete four-part experience. Pattern and Memory remain
  independently toggled, null by default.
- **INV-ALLOCATION-EVIDENCE-002**: Observation-only traces record D/N/C/V,
  failure counts, target scores, feasibility, exposure, initial/incumbent/child
  competence and realized team gain per opportunity. Opportunity-to-gain
  conversion and alignment can be audited from these facts. Distribution or
  concentration alone is not efficacy; counterfactual optimality and individual
  component causality require separately designed experiments. No observation
  changes online ranking, budgets or stopping.

The current questions are allocation efficiency, focused repair quality and
experience transfer efficiency. Task-aligned differentiation is an explanation
of team gains, not generic diversity or member accuracy maximization. Overall
V2.1 gains do not identify a single component's independent effect. No router,
weighted voting, aggregation change, hand-assigned roles or equal member budget
is introduced. IFBench responsibility and HotpotQA retrieval remain blockers.
Implementation and fake conformance do not imply efficacy or execution readiness.

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
  responsibility algebra. Historical V2 bindings use their frozen monotone-safe S0-S2
  key; new V2.1 bindings require the initial-competence/strict-team-gain policy.
  Local scores never enter team ranking. Shadow cardinality is manifest
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
  marker framing and mutable contamination guard remain unchanged. Historical
  interface-repair bindings preserve their pinned equivalence evaluator; the
  explicitly bound answer-domain V2 amendment below versions payload scoring.
  Format noncompliance aborts without regeneration; fake and real
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

## MATH answer-domain amendment V2 (explicit binding)

- **INV-MATH-ANSWER-DOMAIN-002**: New MATH execution selects
  `MATH_EXECUTION_BINDING_V1_4`, `MATH_ANSWER_DOMAIN_V2`,
  `MATH_PAYLOAD_PARSER_V2`, `MATH_EQUIVALENCE_V2` and the explicit
  `MATH_VERIFY_SETTINGS_V2`. The immutable Solver interface and its exact
  one nonempty final-marker line remain V2. Mathematical parsing receives only
  that payload, never reasoning or a guessed last number. Historical V1
  adapters, protocols, bindings and memberships remain replay contracts.
  Pinned math-verify 0.6.0, latex2sympy2_extended 1.0.9 and existing dependency
  pins remain unchanged. Correctness calls pinned verify with gold first;
  aggregation retains its existing finite symmetric-equivalence audit.
  Strict mode, normalization, rounding, precision and deadlines are explicit.
  A minimal container-family boundary prevents the pinned grader's unwanted
  interval/endpoint-set conversion and directional equation/scalar collapse.
  Equations remain mathematical relations with strict variable identity;
  scalar expressions remain a separate object family. No coordinate or
  equation solving is handwritten. Converter options are bound explicitly at
  the pinned public conversion seam; dependency files are unchanged.
- **INV-MATH-SCORABLE-REFERENCE-002**: Full canonical train7500/test5000
  references are prepared only in `EVALUATOR_CONTRACT_PREP_CONTEXT`. It projects
  source-authored extracted references and representation metadata, never
  problems, model predictions or scores to optimization components. Native
  parse and verify probes precede wrapper admission. Bare two-component
  parentheses are ambiguous between ordered tuples and open intervals;
  the pinned converter changes type according to endpoint values, and the
  source format supplies no type tag. V2 therefore marks this representation
  unscorable after native parsing rather than inventing its intended type.
  Unambiguous native tuples, matrices, sets, intervals, multiple values and
  expressions use the pinned semantics with common settings. Eligibility
  requires nonempty extraction, supported mathematical parsing and successful
  self-equivalence. Unsupported, exception and timeout references are data
  exclusions, never silently wrong examples. Apply eligibility over the whole
  source before the unchanged deterministic stratified algorithm; preserve
  existing role memberships if every existing reference is scorable, otherwise
  version the split. Counts remain150/300/300/300 and source roles unchanged.
  Every selected reference has a source-bound scorability receipt; actual
  Optimize/Shadow reference checks fail before dispatch. Search never opens
  Validation/Test for this audit. Test model calls remain forbidden.
  A correctly framed unparseable mathematical prediction is invalid/wrong;
  Historical bindings through V2.1 execution binding V2 still abort on marker failure without regeneration; the explicitly frozen prediction validity binding below supersedes that behavior only for fresh scopes. Only evaluator semantics
  change; responsibility, aggregation, scheduling, GEPA and admission do not.

## V2.1 MATH fresh binding

`MATH_V2_1_EXECUTION_BINDING_V1` binds the V2.1 factory directly to the
benchmark-wide `MATH_ANSWER_DOMAIN_V2` / `MATH_EQUIVALENCE_V2` amendment and
`MATH_SCORABLE_REFERENCE_V2` eligibility. Historical V2 execution bindings
are evidence only and are not consulted for admission. Evaluator settings,
source bytes, generic initial prompts, equal aggregation and numeric search
budgets remain frozen. Full-source reference preparation and any new split
freeze precede real execution; active reference defects abort before dispatch.

Initial member competence is the aggregation-independent binary correct count
on the complete frozen Optimize150 membership, using the same V2 evaluator as
candidate Full. Its ordered support hash, metric, evaluator and initial state
are persisted once, survive commits, and are never rebased. Allocation traces
include prior opportunity counts and prior responsibility-observation exposure
counts for every member. Commit gains carry the Full support identity; only
identical support/metric/evaluator deltas may be summed. Exposure telemetry has
no allocation or stopping read point. Canary retains its separately declared
first-parent-epoch/first-commit phase boundary; Pilot uses the unchanged global
scientific stopper. Neither phase forces round-robin.

The separately versioned immutable `MATH_SOLVER_INTERFACE_V3` requires one
literal `FINAL_ANSWER: <answer>` line and no visible reasoning or Markdown.
This output-only repair cannot be optimized by GEPA. The prior V2 interface
and strict marker/payload/equivalence parsers remain unchanged. Fresh bindings
freeze the exact interface hash; Solver, Validation and isolated accounting
preparation resolve the same interface bytes. An unknown interface or hash
mismatch fails before dispatch. Historical format failure still aborts with zero response
regenerations; fresh prediction-validity binding V3 records it as incorrect. Every interface change requires a fresh source, attempt,
authorization, cache namespace and process.

`MATH_SOLVER_INTERFACE_V4` preserves the V3 system bytes and adds an immutable
format-only suffix after the problem in the user message. The combined interface,
system bytes and suffix each have frozen hashes. Solver and Validation accounting
use the same benchmark-owned composer. Mutable member prompts remain byte-identical
and GEPA cannot change this suffix. The suffix contains no problem, reference,
example, solution, strategy or held-out information. V2/V3 historical request
compositions remain reproducible; unknown identities and altered hashes fail closed.

`MATH_SOLVER_INTERFACE_V5` preserves the V4 system/suffix bytes and versions
the Solver-only output envelope to 3600 tokens after an observed length abort.
Reflection and Pattern retain the 1800-token cap; GEPA metric/minibatch/promotion/
patience budgets, method semantics and total 30M authorization do not change.
The explicit `decoding.solver_max_output_tokens` has read points in Solver request
composition, isolated accounting preparation and protected Validation reservation.
V5 requires exactly this role envelope; prior interfaces retain their original
caps. Full initial/final Validation reservation must fit before admission and
remain protected during search. Historical bindings still abort on truncation without regeneration. Fresh
prediction-validity binding V3 retains strict parsing and records truncation as
incorrect; no scientific stopper changes.

## Authority map

See AGENTS.md for the complete authority hierarchy; method.md is a conceptual
overview, CURRENT_ARCHITECTURE.md a code map, and reports/ immutable evidence.
Current implementation does not imply real-execution readiness or authorization.

An explicit human authorization amendment may extend an existing
RESERVATION_V2 journal from 30M to 40M. The original authorization, hash chain
and every prior charge remain immutable. The extension is a new journal event;
new bindings opt into the effective ceiling, and historical bindings cannot
silently consume the enlarged authorization. No fresh replacement ledger is
permitted. Reservation physics and missing-usage charge rules remain V2.


### Solver decoding policy

`MATH_V2_1_EXECUTION_BINDING_V2` freezes `SOLVER_DECODING_POLICY_V1` for every
Solver/member request, including initial profiling, GEPA local, Optimize,
TeamProbe, Full, Shadow and both post-search Validation teams:
enable_thinking=false, temperature=0.7, top_p=0.8, top_k=20, min_p=0,
presence_penalty=0, frequency_penalty=0, max_output_tokens=3600.
The Chat Completions wire maps the output cap to `max_tokens`; extension fields
enable_thinking/top_k/min_p enter the final top-level body through extra_body.
This exact policy enters manifests, preexecution, authorization, request and
cache identities. Frozen older execution bindings retain their original bytes
and request semantics. Reflection/Pattern retain the existing optimizer policy.
In historical execution binding V2, no regeneration follows an invalid final
marker or length termination: a successfully dispatched Solver policy fails closed with
`STOP_SOLVER_DECODING_POLICY_INSUFFICIENT`. Provider rejection cannot silently
remove any frozen field. No decoding, prompt-interface, method, data or evaluator
adaptation follows performance or formatting outcomes.


### Default Optimizer/Reflection thinking policy

The user-approved `OPTIMIZER_REFLECTION_GENERATION_POLICY_V2` is the default
for newly constructed qwen3.7-flash Reflection/Optimizer and Pattern contracts.
It explicitly sends `enable_thinking=false`, `temperature=0.0`, and
`max_completion_tokens=1800`; `max_tokens` remains absent. The requested cap,
1810 accounting/verification ceiling, 10-token measurement tolerance and
fail-closed truncation semantics are unchanged. Solver decoding V1 already
disables thinking and remains unchanged.

New low-cost contract construction uses V2 by default and requires explicit
amendment authority. A frozen contract selects its own policy by versioned
identity; historical V1 continues to send `enable_thinking=true` and retains
its original reservation metadata. Disabling thinking without changing the
policy identity is rejected. The policy remains part of manifest, request,
cache and authorization identity. Historical bindings and reports are not
rewritten; a real continuation requires fresh preexecution and authorization.

### Optimizer non-thinking root-cause diagnostic

`OPTIMIZER_NONTHINKING_WIRE_WITNESS_V1` is an explicitly authorized, single-use,
optimizer-only operational diagnostic before a fresh low-cost Canary. It sends
one successful Reflection request using the exact recovered historical messages
and current explicit non-thinking policy. Only frozen transport failures may
retry. It never scores, archives, evolves, or reuses the resulting text in
Canary/Pilot/cache/Memory. It uses the original cumulative 40M journal and
protects the complete Validation100 first-attempt reserve.

Serialized HTTP body, actual reasoning usage/content length, fences, candidate
length, duplicate lines and repeated 8-grams are observed without changing GEPA
evidence, minibatch three, proposer template or accepted prompt limit3000.
Historical witness V1 used reported reasoning_tokens=0 and absent/empty
reasoning content. Fresh bindings may explicitly select
`OPTIMIZER_NONTHINKING_EQUIVALENT_EVIDENCE_V1`: Level A uses an actual zero;
Level B permits an absent token field when false is dispatched, the provider
accepts the request with stop, reasoning content is absent/empty, no retained
provider metadata indicates thinking, and the raw-HTTP metadata preservation
audit finds no dropped field. Absence is never relabeled zero. Missing optional
metadata alone is not a scientific blocker. A single diagnostic candidate
rejection is recorded and never admitted or reused, but does not gate a fresh
Canary. Scientific candidate validators remain unchanged.
Positive reasoning under explicit false fails closed as a
provider-control defect, distinct from cap exhaustion. Truncation remains a hard
failure in production; the diagnostic preserves truncated output for owner
classification without admitting it as a candidate. Generation amendments require
the user-authorized evidence criteria, a new policy/source/attempt and fresh
freeze; no Pilot may change its frozen policy after its first request.

### Historical Optimizer/Reflection generation wire amendment V1 (contract)

Historical frozen low-cost execution binding V2 opts into
`OPTIMIZER_REFLECTION_GENERATION_POLICY_V1` for qwen3.7-flash Reflection and
Pattern requests. It explicitly sends `enable_thinking=true`, the frozen
`temperature=0.0`, and `max_completion_tokens=1800`; `max_tokens` is absent.
The 1800 requested cap covers reasoning plus answer. The documented provider
measurement tolerance is 10 tokens, so Reservation V2 reserves the exact UTF-8
request input bound plus 1810 and verifies reported total output against 1810.
The 10-token tolerance is not sent as extra model generation budget.
`finish_reason=length` remains `OPERATIONAL_OUTPUT_TRUNCATION`, including when
reported usage fits the tolerance; output above 1810 remains
`OPERATIONAL_OUTPUT_CAP_NOT_ENFORCED`. All output usage, including reasoning,
is accounted; available provider usage details are retained privately.

The complete optimizer policy enters binding, manifest, preexecution, request,
cache and authorization identities. Legacy bindings retain their original wire
and accounting semantics. Solver decoding V1, GEPA budgets, method V2.1,
Low-Cost 12/60/40/100 memberships, terminal-invalid recovery and held-out
firewalls remain frozen. Real A1 Pattern calls remain zero.

The historical answer-only `max_tokens=1800` versus reported total 1814 is a
client cap/usage dimension mismatch, not evidence that the provider violated
its answer cap. Preserve the historical runtime stop and charge, with an
append-only classification correction; the failed Canary does not become PASS.

### MATH prediction validity policy

`MATH_V2_1_EXECUTION_BINDING_V3` explicitly freezes
`MATH_PREDICTION_VALIDITY_V1` and benchmark protocol V3. The method remains
Unified V2.1. All Solver decoding fields and immutable interface V5 bytes stay
unchanged. Reference scorable checks, payload mathematical semantics,
`MATH_EQUIVALENCE_V2`, `MATH_SCORABLE_REFERENCE_V2`, memberships, initial
team, Responsibility, allocation, failure discount, GEPA budgets, Pattern,
Memory, strict team gain and immutable initial floor stay frozen.

After successful transport, missing/empty/multiple final markers, other strict
framing violations, unsupported/unparseable payloads, and length termination
(even with an apparently valid marker) are INVALID_PREDICTION: validity false,
binary correctness zero, execution continues. No regeneration, fallback
extraction, output repair, dropped row or resampling occurs. Native prediction
parse exceptions/deadlines are invalid; worker initialization, dependency,
serialization or persistence faults remain hard operational failures.
Reference parse/unsupported/timeout failure remains a hard contract failure.

Private typed prediction results retain exact raw text and finish reason across
cache reuse, state snapshots and durable serialization. Classification is
benchmark-owned. Invalid predictions have no equivalence-class vote; all-invalid
teams have incorrect Vote/Oracle. Valid classes retain the frozen equal-weight
plurality and tie behavior. Wrongness alone enters existing Responsibility.
Initial and candidate competence use the complete Optimize membership; both
Validation teams retain all 300 rows per member. Invalid counts/rates/reasons
by phase/member are observations, with no allocation, promotion, transition,
Shadow, stopping or efficacy threshold read point. Existing score-based
TeamProbe/Full/Shadow guards remain unchanged. Legacy invalid-response hard
guards apply only to historical bindings; in V3 invalidity acts solely through
incorrectness. Transition still requires candidate member >= immutable initial
floor and candidate team > incumbent team, allowing specialization.

The policy enters execution binding, manifest, preexecution, authorization,
request/cache identities and Validation accounting metadata. Cached invalid
responses share their original realization; they are never regenerated or
refunded. Transport retry policy, RESERVATION_V2 ledger and full protected
Validation reserve remain unchanged. Historical Canary1–5 remain immutable
INVALID attempts under their original contracts. Fresh Canary requires complete
initialization, Responsibility/target and a complete production opportunity;
no invalid-rate or efficacy gate applies. Pilot and post-search paired Validation
require distinct fresh scopes and immutable SEARCH_COMPLETE_RECEIPT. Real A1
keeps Pattern/Memory off; Test and all other arms/seeds remain unauthorized.

### MATH terminal-invalid recovery (fresh binding only)

`MATH_V2_1_LOW_COST_EXECUTION_BINDING_V1` opts into
`MATH_SOLVER_INVALID_RECOVERY_V1` and prediction validity V2. The frozen
interface V5, Solver decoding V1, mathematical evaluator and reference domain
remain unchanged. Each logical Solver evaluation receives one first attempt
and at most three fresh semantic retries with identical request bytes.
Successful invalid responses consume semantic attempts; transport failures
retain their separate frozen retry budget. The first valid prediction stops
recovery, including a mathematically wrong answer. Only four actually completed
invalid responses constitute terminal-invalid: wrong, no valid vote, retain
the row and continue. Insufficient retry budget stops execution without
manufacturing a terminal prediction. Recovery does not multiply GEPA logical
metric calls or add an invalidity-specific allocation/transition guard.

Only resolved-valid or exhausted-terminal outputs enter exact caches, with
original predictions, attempt counts, finish reasons and validity metadata.
`DURABLE_EXACT_OUTPUT_CACHE_V1` seals private entries to the same attempt,
namespace, source, startup, authorization, binding, decoding and recovery.
Fresh scientific attempts, including Canary to Pilot, share no outputs.
Stage and member identity remain telemetry; identical scientific request
identities share the original realization. Cache hits generate no physical
call or token charge. Durable output caching alone does not authorize a
process resume: lifecycle, authorization and checkpoint checks still apply.

### Low-cost MATH development protocol

`MATH_V2_1_LOW_COST_DEV_PROTOCOL_V1` derives Pilot Optimize60, Shadow40
and Validation100 from the immutable Optimize150/Shadow300/Validation300
supersets. Canary Optimize12 is nested in Pilot Optimize60. Subject×level
integer largest-remainder allocation has lexical tie breaking and no minimum
quota; stable SHA256 ordering selects within strata and preserves superset
output order. Selection uses metadata alone, before provider execution.
The full source split and sealed Test remain immutable. Membership hashes
enter binding, manifest, startup, request/cache and authorization identities.

Canary has its own immutable floor on 12 rows and closes immediately after
one complete production opportunity. Fresh A1 Seed81 Pilot starts with its
own floor on all 60 Optimize rows, uses the existing adaptive gate on 40
Shadow rows, and stops through the unchanged team_epoch_no_commit_v1
semantics. Responsibility, failure discount, opportunity allocation, initial
floor, strict team gain, aggregation, team size and GEPA numerical budgets
remain unchanged. Search never reads Validation correctness. After immutable
SEARCH_COMPLETE_RECEIPT, paired Initial/Final Validation uses all 100 rows,
identical Solver policy/recovery and shared exact realization caching.
Mandatory first-attempt Validation reserve covers 1000 logical evaluations;
the expected base physical count is at most 500+100×changed members before
retry overhead. Retries reserve individually when needed. The 40M ceiling is
an upper bound, with every physical transport separately charged, and never
an expenditure target. This is development evidence only; no Test, formal
three-seed run, other arm or other seed follows any result.
