---
archive_status: HISTORICAL_REPLAY_ONLY
source_document: AGENTS.md
source_commit: ec993b13a252699f19b93348b01570ffd9bea879
source_section: Historical two-layer architecture
---

## Historical two-layer architecture (frozen replay reference only)

The remainder of this section describes the former production direction and
must not constrain the unified method. Its frozen rules still govern exact
historical reproduction.

The active research direction is a backend-neutral two-layer architecture:

```text
Layer 2 — Team-Level Responsibility/Search Controller
    responsibility and primary-residual assignment
    target scheduling and persistent realizability
    team-level candidate evaluation
    Common-Safe and Shadow checks
    competitive selection and atomic write-back
                         |
                         | LocalPromptOptimizer interface
                         v
Layer 1 — Local Prompt Optimizer
    current grounded backend: official frozen GEPA
    replaceable future backends: SEPO, ESPO, or another conforming optimizer
```

Layer 1 is an interchangeable prompt-optimization backend. It must not own
member selection, responsibility attribution, persistent realizability,
team-level acceptance, or write-back. Layer 2 is the team-level research layer
and must not depend on GEPA, Teacher, Critic, Student, or any other backend's
internal implementation. Replacing GEPA must not require redesigning Layer 2.

This architectural direction does not silently promote an experimental runtime
to canonical status. `versions.py` still identifies the canonical historical
runtime as `member_aware_peer_state_v15`, checkpoint v25. Opt-in two-layer and
Layer-2 scheduler identities must remain explicit in their manifests until a
separate, versioned promotion changes `CURRENT_SPEC.md`, `versions.py`, code,
and tests together.

### Two-layer ownership boundary

Layer 2 owns member selection, responsibility attribution, exact responsibility,
latest-transition focus/anchor, and local-evaluation evidence selection, the
ordered evidence schedule, persistent realizability, team-level empirical evaluation,
Common-Safe and Shadow checks, cross-member competition, and atomic write-back.
Layer 1 retains optimizer-specific prompt-search mechanics, but treatment
backends may consume only the immutable Layer-2 evidence packet. They may not
fetch, select, replace, expand, reorder for selection, or fall back to examples
outside that packet.

Layer 1 owns how the assigned local problem is searched: internal sampling,
proposal generation, candidate population and selection, and optimizer-specific
state. It must not choose the target member, redefine responsibility, use
persistent team realizability or plurality outcomes for target allocation, or
commit prompts to the team.

```text
Layer 2 defines WHO to optimize, WHAT responsibility to pursue, and WHICH
evidence Layer 1 must optimize from.
Layer 1 retains HOW to mutate, reason over, compare and accept prompt candidates.
```

The experimental treatment decomposes Layer-2 evidence as
`E_L2 = E_team union E_transition`: current team residuals provide
responsibility evidence, while only the latest accepted parent-to-child edit
provides focus (newly broken) and anchor (newly fixed) evidence. Root prompts
have empty focus and anchor sets. Layer 2 freezes `M_eval` from the exact
TeamMiniBatch12 example IDs before candidate generation. Identity reuse does
not merge decisions: GEPA local acceptance compares target-member performance;
TeamMiniBatch subsequently tests fixed-peer plurality/team metrics. No
TeamMiniBatch, Full, Common-Safe, or Shadow result feeds back into local
acceptance. Preservation vulnerability is latest accepted target-member
transition change OR current target pivotality to correct plurality, ordered
first, then smaller positive plurality margin, greater valid-answer
disagreement, ascending SHA256(example ID), and raw ID only for a theoretical
hash collision. This deterministic proxy is not learned or frequency memory.
The focus/anchor design borrows transition-evidence semantics associated with
SEPO; it does not import
SEPO structural edits, architects, breadcrumb search, Lexicase selection,
archive admission, or lineage-parent selection into Layer 2.

The opt-in V4 diagnostic computes responsibility `D/N/C/V`, lane and target
scores from the complete pre-routing raw-legal universe. Before selecting a
target, Layer 2 masks members unable to form the unchanged TeamMiniBatch12 or
an exact-transition packet with at least four repair slots. Among feasible
members it retains `V/(1+f)` and member-ID ties; skipped members do not update
`f`. The immutable Layer-1 search packet schedules at most 36 nominal role
items, reserving exact focus/anchor and taking deterministic primary-lane
repair rows for the remainder. Full responsibility, scheduled evidence and
actually delivered GEPA evidence must be audited separately. The packet and
assignment carry raw `V`, never the discounted target score. No feasible
member ends the opportunity scientifically with
`NO_FEASIBLE_LAYER2_OPPORTUNITY` before its provider calls.

The sole active production Layer-2 responsibility graph is the overlapping raw
legal member–residual relation. Primary responsibility selects a target; it
does not exclusively own residuals. Equal target scores break by ascending
member ID, not seed or round-robin state. TeamMiniBatch12 uses four repair,
four team-correct preservation, and four team-hard residual examples; no
coalition/unassigned-residual evidence group is part of the production
contract. Historical V1 freeze documents remain archival evidence and require
a new freeze before any experiment under these semantics.

### Unified backend runtime

Optimizer and controller choices are runtime configuration, never scientific
facts inferred from a branch name. The current executable matrix is:

```text
optimizer_backend=gepa  optimization_mode=native
optimizer_backend=gepa  optimization_mode=layer2
optimizer_backend=mars  optimization_mode=native
optimizer_backend=mars  optimization_mode=layer2
```

All four modes come from one commit and one shared Layer-2 implementation.
Native mode preserves the backend's own Optimize-only data flow. Layer2 mode
requires the backend to consume the immutable `ResponsibilityEvidencePacket`
without fetching or selecting other examples. Formal manifests record both
fields and the backend fidelity identity. Backend-specific telemetry is nested
under `backend_details` in the common run record.

The only long-lived development branch is `main`. Backend work may use a
temporary feature branch, but formal experiments are frozen by commit SHA,
configuration/data/protocol hashes, and immutable preregistration/result tags.
Do not create permanent experiment branches for an optimizer or controller.

Replacing GEPA with SEPO/ESPO must not require an algorithmic change to Layer 2.

### Production architecture rules

New experiments use `multi_dataset_diverse_rl.experiment.run_experiment` and
the single `scripts/run_experiment.py` composition entrypoint. Do not add a new
experiment-named production runner. Historical runners may remain for exact
reproduction or translate legacy arguments into the production contract, but
production modules must never import them.

Do not put scientific logic in scripts. Layer 1 owns local search only. Layer 2
owns target/responsibility/evidence, team admission and write-back, and must not
import a concrete optimizer backend. GEPA and MARS implement the same typed
`LocalOptimizerBackend` contract. Production Layer 1 receives an explicit
`LocalOptimizationRequest` and `RuntimeContext`, never the historical
monolithic `Config` attribute bag.

Backend, native/Layer2 scope and fixed-budget/saturation regime are orthogonal
`ExperimentSpec` fields. Shared stopping belongs in `saturation.py`. Provider
construction uses `ProviderClientFactory`; identity, authorization and
lifecycle use the centralized governance layer. A new experiment should
normally require only a manifest/configuration, not a runner, controller,
backend adapter or stopping implementation.

### Saturation-mode stopping

Saturation mode disables ordinary scientific hard budgets. Native backends use
backend-native complete optimization units. Layer-2 modes distinguish local
accepted updates from successful team commits, and team-level no-update
patience is reset only by a successful atomic team commit. A Vote-neutral
Common-Safe commit is still a team update. Emergency ceilings are mandatory
operational safeguards and never imply scientific convergence.

The native optimizer data flow remains a separate control. The Layer-2
treatment intentionally replaces native example selection while preserving the
optimizer's prompt-search core. GEPA and MARS native budget units need not match
and must not be used for direct cross-backend superiority claims.

### Local optimizer fidelity policy

The current official GEPA backend is **Level B: API-compatible adaptation**.
Allowed adaptations use supported optimizer seams: adapters/interfaces, system
text-component representation, local datasets/problem domain, metrics and
evaluator, reflection templates, provider/model configuration, resource budget,
callbacks, and logging. Editing or forking the optimizer search core,
reimplementing its population/Pareto/parent selection, replacing its local
acceptance, or making Layer 2 control its internal search is forbidden. A
derived optimizer doing any of those requires a separate identity and must not
claim Level-B fidelity.

Official optimizer code fidelity and optimization-problem equivalence are
different concepts. Ours + GEPA uses an API-defined local problem while keeping
the official GEPA search engine unmodified; never describe it as an exact native
GEPA reproduction. This distinction applies equally to future SEPO/ESPO
backends.

<!-- END_VERBATIM_ARCHIVE_BODY -->
