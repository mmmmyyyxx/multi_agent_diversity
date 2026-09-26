# Current Implementation Specification

This is the normative implementation specification for two explicitly distinct
runtime families: the v15 canonical replay method below and the active,
opt-in two-layer research architecture later in this file. It is not a paper
narrative or an experiment report. The active direction does not inherit the
v15 method identity. Runtime identifiers and frozen constants remain
authoritative in `multi_dataset_diverse_rl/versions.py`; experiment-specific
identity and authorization belong to the frozen manifest.

## Canonical v15 Replay Runtime

The canonical runtime is `member_aware_peer_state_v15`, checkpoint v25:
Repairability-Adjusted Dual-Target Prompt-Team Optimization. It retains the
semantic hard-veto Teacher-Critic-Student pipeline. Experimental evidence does
not modify this section without a separate method change and version update.

### Identity and team semantics

- **INV-ID-001** — Runtime protocol identifiers and frozen weights MUST be read
  from `versions.py`; no parallel constants authority is permitted.
- **INV-VOTE-001** — The team has exactly five equally weighted members.
  Plurality ties abstain and are incorrect. For each row, `G` is valid gold
  support, `H` the largest valid wrong cluster, and the vote is correct iff
  `G - H > 0`.

### Dataset lifecycle and isolation

- **INV-DATA-001** — Training diagnosis and candidate rollout use only the
  frozen optimization split. The canonical runtime performs no validation
  rollout or validation-based checkpoint selection; the final active state is
  selected automatically.
- **INV-TEST-001** — Test data MUST NOT influence search, candidate generation,
  candidate selection, arm selection, validation selection, or final-state
  selection. If enabled, test is evaluated at most once after training freeze.

### Responsibility and target scheduling

- **INV-RESP-001** — Only vote-wrong rows create residuals. Only currently
  wrong members are eligible. Eligibility is the lexicographic argmax of
  counterfactual `(DeltaV, DeltaM)` with exact ties retained; gain, wait,
  history, memory, and load cannot alter eligibility.
- **INV-ROUTE-001** — Each serviceable residual is routed to exactly one legally
  eligible member. Service portfolios are disjoint. Each member exposes one
  active lane and only its routed same-lane slice.
- **INV-TARGET-001** — S1/S2 select the two highest-ranked distinct actionable
  members under the repairability-adjusted W1 score. One actionable member
  degrades to one branch; none causes `no_actionable_responsibility`.
- **INV-REPAIR-001** — Normal branches update attempt/feasible/failure state.
  Operational failures do not. Only an accepted changed team state resets all
  state-local repairability counters.

### Branch construction and TCS

- **INV-BRANCH-001** — S1/S2 construct two independent branches from the same
  parent team, peers, responsibility snapshot, routing, lanes, and profiles.
  Each branch has two candidates; branches cannot observe each other's changes.
- **INV-TCS-001** — Program selects numerical evidence; Teacher proposes one
  bounded repair hypothesis; Critic applies canonical semantic blockers;
  Student realizes replacement prompts; rollout supplies empirical value.
  Student sees no raw peer output, identity, score, wait, load, or empirical
  rejection history. Invalid recovery is bounded.

### Candidate evaluation, Common-Safe, and ranking

- **INV-COMMONSAFE-001** — With four peers fixed, a candidate must preserve
  target correct count, preserve team vote correct count, strictly improve at
  least one of them, and not increase terminal-invalid count.
- **INV-RANK-001** — Branch candidates and branch winners use the versioned
  Common-Safe ranking. Cross-branch comparison cannot use target absolute
  accuracy, raw lane utility, or raw portfolio size.
- **INV-COMMIT-001** — At most one prompt commits per update. The global winner
  is chosen before atomic prompt/profile/anchor update, successor diagnosis,
  repairability reset, responsibility refresh, persistence, and audit. Any
  failure rolls all affected state back.

### Artifacts and access evidence

- **INV-ARTIFACT-001** — Publishable artifacts may contain hashes, identifiers,
  counters, categories, metrics, and token totals, but not prompts, questions,
  gold/model answers, raw responses, secrets, endpoints, caches, checkpoints,
  or absolute host paths.
- **INV-STRUCTURAL-EVIDENCE-001** — Every fixed-probe initialization and Full
  candidate evaluation MUST atomically persist a sanitized per-example
  categorical profile before downstream selection. Every accepted commit MUST
  atomically persist the realized state's frozen-plurality `P_0..P_4` audit.
  These artifacts contain only example IDs, categorical choices or hashes,
  correctness/invalid bits, candidate/state hashes, and structural counts; no
  prompt, question, gold/model answer, reasoning, or raw response is retained.
- **INV-AUTH-001** — API use fails closed unless the user explicitly authorizes
  it and the frozen manifest independently authorizes the phase and role with a
  matching preregistration hash and frozen budget.
- **INV-MANIFEST-001** — Experiment design, API authorization, budget,
  validation/test policy, selection rule, source identity, evidence type, and
  lifecycle MUST be explicit. Protocol changes after freeze require an
  amendment; historical artifacts are never rewritten.

## Reduced matrix

The canonical matrix is Static, S0 Generic, S1 member-aware dual-target, and S2
responsibility-conditioned dual-target. S2 is the full method. Common-Safe is a
shared write-back policy rather than a separate module. Legacy and auxiliary
settings require their explicit opt-ins.

## Next Candidate Architecture

The latest fixed-parent four-arm experiment selected this candidate pipeline:

```text
Teacher-Clean
-> deterministic hard gate
-> Student
-> empirical rollout
```

Its status is exactly `SELECTED_FOR_NEXT_ONLINE_VALIDATION`. Evidence supports
higher Student throughput, more feasible candidates, and better target transfer
within the frozen local experiment. It does **not** demonstrate validation Vote
improvement, online trajectory superiority, test improvement, or promotion to
canonical runtime. The canonical Critic path above remains unchanged.

## Active Two-Layer Research Architecture (opt-in)

The active research direction separates a replaceable Local Prompt Optimizer
from the Team-Level Responsibility/Search Controller. Official frozen GEPA is
the current grounded local backend; responsibility, target allocation,
persistent realizability, team evaluation, Common-Safe, Shadow, selection, and
atomic write-back remain Layer-2 concerns. These opt-in identities do not alter
the canonical v15 runtime above.

- **INV-LAYER-OWNERSHIP-001** — Layer 2 defines the target member, team
  responsibility, exact responsibility/latest-transition focus/latest-transition
  anchor/local-evaluation examples and ordered evidence schedule before Layer 1
  begins. The immutable packet is the
  complete optimization curriculum. Layer 1 owns prompt mutation, optimizer-
  specific reasoning, parent/candidate search, frontier and acceptance, but may
  not select or fetch examples outside the packet.
- **INV-LAYER2-RAW-RESPONSIBILITY-001** — Layer-2 responsibility scores
  (`D_i`, `N_i`, `C_i`), primary lane, target selection, and responsibility
  evidence MUST derive from the current parent state's raw legal
  member–residual assignments, before historical service routing, active-lane
  slicing, portfolio/load balancing, or freeze policy. A residual legally
  assigned to multiple members remains in each member's opportunity set.
  Historical routing state MUST NOT affect Layer-2 allocation or evidence.
- **INV-LAYER2-FINAL-SEMANTICS-001** — The production Layer-2 responsibility
  graph is the overlapping raw legal member–residual relation. Primary
  responsibility justifies choosing a target and does not grant exclusive
  ownership. The current score is `max(4D_i, 2N_i, C_i)/(1+f_i)`; equal scores
  are ordered by ascending member ID, independent of seed, update index or
  RR state. Evidence examples are ranked by the stated heuristic and then
  ascending SHA-256 of example ID (raw ID breaks a theoretical hash tie).
  Identical five-prompt initialization is retained. Historical
  service routing remains archival and cannot enter production Layer-2.
- **INV-LAYER2-FEED-CONTROL-001** — Native GEPA and Native MARS remain separate
  controls with their frozen native data flow. Treatment deliberately replaces
  native example selection while retaining each optimizer's search core.
  Treatment effects therefore include allocation, responsibility, curriculum
  construction and team admission; they are not component-level ablations.
- **INV-LAYER2-TRANSITION-EVIDENCE-001** — Layer-2 evidence is
  `E_L2 = E_team union E_transition`, and optimizer input is
  `M_t = (M_resp, M_focus, M_anchor, M_eval)`. Responsibility rows are current
  team residuals. Focus rows are exactly parent-correct to child-wrong cases,
  and anchor rows are exactly parent-wrong to child-correct cases from the
  current parent's latest accepted transition. Root focus/anchor sets are empty;
  histories do not accumulate in optimizer input. Local evaluation is frozen
  independently through explicit Layer-2 example identities and role intersections
  are persisted as sanitized counts. Missing explicit identities fail closed; a
  backend may never fetch or backfill from a global pool. Direct packet fixtures
  without explicit identities use deterministic team-hard rows selected by Layer 2.
  These semantics do not add
  SEPO search operators, architects, breadcrumb search, Lexicase selection,
  archive admission, or lineage-parent selection to Layer 2.
- **INV-UNIFIED-BACKEND-RUNTIME-001** — `optimizer_backend` (`gepa` or `mars`)
  and `optimization_mode` (`native` or `layer2`) are independent runtime
  configuration fields. GEPA_NATIVE, GEPA_LAYER2, MARS_NATIVE, and MARS_LAYER2
  execute from one commit. Native mode retains backend-owned Optimize-only data
  selection; Layer2 mode consumes the one shared immutable evidence packet.
  Backend additions extend the registry and do not fork Layer 2 or require a
  permanent Git branch.
- **INV-UNIFIED-RUN-SCHEMA-001** — All four modes emit the common top-level run
  schema in `infrastructure/unified_backend_run.schema.json`. Backend-specific
  fields appear only under `backend_details`; the manifest records backend,
  mode, fidelity, code/config identity, budget, data split and initial state.
- **INV-OPTIMIZER-FIDELITY-001** — The official GEPA backend is a Level-B
  API-compatible adaptation. It calls the frozen, source-verified official
  search engine through supported API seams and does not edit or replace its
  search core. Ours + GEPA is not described as a native GEPA reproduction.
- **INV-LOCAL-COMPONENT-001** — The sole mutable GEPA candidate component is
  `decision_procedure`. The adapter/evaluator instantiates the immutable solver
  shell and output interface outside GEPA exactly once.

### Fixed-budget and saturation execution regimes

The unified four-mode runtime supports two distinct experimental regimes. In
`fixed_budget` mode, ordinary metric-call, optimizer-round, and team-opportunity
limits remain scientific controls for matched-resource comparisons. In
`saturation` mode those ordinary limits are disabled and termination is based
on repeated complete optimization units with no accepted deployable update.

- **INV-SATURATION-001** — Native GEPA counts a complete pinned-engine training
  epoch; native MARS counts a complete Teacher/Critic/Student/Target round.
  Equal-score archive entries are not deployable updates and do not reset
  patience.
- **INV-SATURATION-002** — A Layer-2 local unit is a complete pass through the
  immutable packet schedule for GEPA or a complete packet-owned MARS round.
  `LAYER2_EVIDENCE_EPOCH_POLICY_V1` replays the same frozen packet; exhaustion
  ends an evidence epoch and never falls back to backend-owned sampling.
- **INV-SATURATION-003** — Layer-2 team saturation is backend-neutral. A team
  epoch observes existing scheduler decisions until every member eligible at
  epoch start has received an opportunity. Members may repeat; scheduler
  scores, deterministic member-ID tie ordering, persistent realizability, and focus/anchor semantics are
  unchanged. Only a successful atomic team commit resets outer patience. A
  Common-Safe commit with zero Vote delta still counts; local-only acceptance
  does not.
- **INV-SATURATION-004** — Provider-call, optimizer-step, team-epoch, and wall
  ceilings remain mandatory operational safeguards. Reaching one aborts the
  run and MUST NOT be reported as scientific convergence.

Saturation comparisons estimate each method's approximate performance ceiling
under its own search process. Primary interpretations are paired within
optimizer (`GEPA_NATIVE` vs `GEPA_LAYER2`, and `MARS_NATIVE` vs
`MARS_LAYER2`). They do not establish cross-optimizer efficiency or superiority;
those questions require separate fixed-budget experiments.

- **INV-LOCAL-RESULT-001** — The GEPA seed/root program is a baseline, not a
  proposal. The `changed_candidates_only_v1` boundary returns only frontier
  prompts whose bytes differ from the parent. No changed frontier produces an
  empty candidate tuple with `no_local_improvement`; parent-versus-parent work
  must not enter TeamMiniBatch.
- **INV-LOCAL-GEPA-CONTRACT-001** — The opt-in local GEPA backend uses the
  versioned `decision_procedure_proposer_v1` reflection template, explicitly
  freezes pinned-engine strict-improvement, perfect-score skip, epoch-shuffled
  sampling, full validation evaluation, merge-off, unit evidence weights, and
  the 3000-character mutable boundary. Parent prompts fail before GEPA starts;
  changed candidates are contract-validated and prompt-hash deduplicated before
  Top-K selection.
- **INV-LOCAL-REFLECTION-DATA-001** — The versioned
  `component_specific_reasoning_evidence_v1` reflective dataset exposes only
  problem text, reasoning-only trace, coarse correctness outcome, and an
  allowlisted evidence-group/reasoning-lane focus. It excludes gold labels,
  raw failure codes, raw Solver responses, free-form controller instructions,
  and immutable answer/interface lines. This is a component-representation
  adapter boundary; it does not change official GEPA search semantics.
- **INV-TEAM-MINIBATCH-001** — Production TeamMiniBatch12 is exactly twelve
  unique Optimize rows: four target-member primary-lane repair residuals,
  four currently team-correct preservation rows, and four global team-hard
  vote-wrong residuals. Team-hard rows may also have legal member
  responsibility; they do not require an unassigned residual. The same ID
  cannot occupy two slots. Missing quotas fail closed; no silent backfill or
  smaller minibatch is permitted. Preservation order is latest-transition
  sensitivity first, smaller positive plurality margin, greater current
  disagreement, then ascending example ID. Team-hard order is greater current
  disagreement, more currently wrong members, then ascending example ID.
  This last count is a current-state frequency proxy, not new historical
  memory. Team collateral effects are measured after generation in
  TeamMiniBatch/Full/Common-Safe/Shadow, never predicted by a coalition module.
- **INV-PAIRED-EVAL-001** — Paired final evaluations share one exact-request
  realization cache across arms. Equal prompt/question/solver-contract request
  identities reuse the same provider realization, so byte-identical final teams
  have byte-identical paired evaluation evidence.
- **INV-SOLVER-STAGE-001** — Any runtime context capable of producing a Solver
  ledger record contains an explicit non-empty `phase` before evaluation starts.
  Layer-2 evaluation producers use the canonical phase names; when the redundant
  `evaluation_stage` field is present it must equal `phase`. Consumers remain
  fail-closed and never infer or default a missing phase.

## Authority map

Machine-readable mirrors of these invariant IDs live in
`docs/design/invariants.yaml`. Experiment-specific deltas belong in manifests;
evidence and conclusions belong in reports. `method.md` is explanatory prose.
