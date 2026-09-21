# Current Implementation Specification

This is the normative implementation specification for the current algorithm.
It is not a paper narrative or an experiment report. Runtime identifiers and
frozen constants remain authoritative in
`multi_dataset_diverse_rl/versions.py`.

## Canonical Runtime

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
  responsibility, exact repair/preservation/local-evaluation examples and
  ordered evidence schedule before Layer 1 begins. The immutable packet is the
  complete optimization curriculum. Layer 1 owns prompt mutation, optimizer-
  specific reasoning, parent/candidate search, frontier and acceptance, but may
  not select or fetch examples outside the packet.
- **INV-LAYER2-FEED-CONTROL-001** — Native GEPA and Native MARS remain separate
  controls with their frozen native data flow. Treatment deliberately replaces
  native example selection while retaining each optimizer's search core.
  Treatment effects therefore include allocation, responsibility, curriculum
  construction and team admission; they are not component-level ablations.
- **INV-OPTIMIZER-FIDELITY-001** — The official GEPA backend is a Level-B
  API-compatible adaptation. It calls the frozen, source-verified official
  search engine through supported API seams and does not edit or replace its
  search core. Ours + GEPA is not described as a native GEPA reproduction.
- **INV-LOCAL-COMPONENT-001** — The sole mutable GEPA candidate component is
  `decision_procedure`. The adapter/evaluator instantiates the immutable solver
  shell and output interface outside GEPA exactly once.

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
- **INV-TEAM-MINIBATCH-001** — TeamMiniBatch12 is exactly twelve unique Optimize
  rows: four from the selected primary responsibility lane, four global
  coalition rows, and four global preservation rows. Missing quotas fail closed;
  no silent backfill or smaller minibatch is permitted.
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
