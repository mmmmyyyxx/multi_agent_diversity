# Two-layer prompt-team optimization architecture

> Historical architecture note. Some interface names and backend examples below
> predate the unified production runtime. The sole active Layer-2 semantics are
> specified in `docs/design/CURRENT_SPEC.md` and summarized in
> `docs/CURRENT_ARCHITECTURE.md`; do not use this note as an experiment freeze.

```text
┌─────────────────────────────────────────────┐
│ Layer 2: Team-Level Responsibility Search   │
│                                             │
│ responsibility → target → residuals         │
│       ↓                                     │
│ build LocalOptimizationTask                 │
│       ↓                                     │
│ [ Local Prompt Optimizer ]                  │
│       ↓                                     │
│ candidate pool                              │
│       ↓                                     │
│ team evaluation → selection → commit        │
└─────────────────────────────────────────────┘
                    │
                    │ backend interface
                    ▼
┌─────────────────────────────────────────────┐
│ Layer 1: Local Prompt Optimizer              │
│                                             │
│ GEPA | Legacy TCS | future SEPO | ESPO       │
└─────────────────────────────────────────────┘
```

Layer 1 is an infrastructure/backbone. Layer 2 is the scientific method under
study. The only downward call is
`LocalPromptOptimizer.optimize(LocalOptimizationTask)`. The only upward value is
a `LocalOptimizationResult` containing a bounded candidate set, local evidence,
opaque backend state, lineage, and local cost accounting.

Responsibility, target scheduling, plurality state, team evaluation, Common-Safe,
Shadow, and atomic write-back are exclusively Layer 2 concerns. A local backend
does not import or receive those policies. The controller does not import GEPA,
Teacher, Critic, or Student internals and contains no backend switch.

Layer 2 defines the optimization problem; Layer 1 owns the optimizer. Layer 2
may choose the target, primary responsibility domain, admissible local evidence,
and resource budget, but official GEPA alone chooses its internal parents,
reflection minibatches, mutations, population/frontier, and local survivors.
Team-level evidence is never fed back into the same GEPA search state.

The `gepa` backend is a Level-B API-compatible adaptation that resolves the same
official frozen checkout used by
`independent_gepa_repro`: GEPA v0.1.1 at commit
`b4dbb55b7601dac448cdb836d5a401ca7d9eb920`. It calls the public
`gepa.optimize()` API with instance Pareto selection, reflection minibatches,
the versioned `decision_procedure_proposer_v1` template, pinned-engine strict
subsample improvement, explicit perfect-score skipping, epoch-shuffled sampling,
full validation evaluation, merge disabled, and official callbacks. GEPA's
local frontier is a single-member per-example search construct; it is never a
team candidate frontier.

GEPA receives exactly one mutable component named `decision_procedure`. The
system adapter instantiates that procedure with the existing immutable
COMMON_SOLVER_CONTRACT_V1 shell and output interface; those immutable bytes are
not GEPA components. The versioned reflection template explains the component
through GEPA's official API, while the official reflective proposer remains in
use. This is “Ours + GEPA” or a “GEPA local optimizer backend,” not an exact
native GEPA reproduction.

Each team opportunity starts a fresh local GEPA state because a commit changes
the prompt, peers, responsibility, and residual distribution. The opaque state
slot remains available for a future, separately governed resume or memory
protocol. `NoOpContextProvider` is the only context-memory provider implemented
here.

The canonical v15 runtime remains unchanged. The active research direction uses
the opt-in two-layer identities and the implemented production binding from
runtime responsibility objects and fixed-probe evaluators to these facades.
Official GEPA is the current grounded Layer-1 backend; Layer 2 remains backend
neutral. Experiments and implementation evidence do not silently promote this
architecture to canonical status: promotion still requires a separately
versioned update to the normative specification, runtime identities, code, and
tests.

The GEPA seed/root program is a local baseline, not a Layer-1 proposal. The
`changed_candidates_only_v1` result contract returns only frontier prompts whose
bytes differ from the parent. When GEPA finds no changed frontier candidate,
Layer 1 returns an empty candidate tuple with `no_local_improvement`; it does
not send parent-versus-parent work through TeamMiniBatch.

Before GEPA starts, the parent prompt and unit-weight evidence contract fail
closed. Search/validation rows sharing an ID must be byte-for-byte equivalent.
Changed frontier prompts are hard-validated and deduplicated by prompt hash
before ranking and Top-K truncation. The 3000-character boundary and proposer
template hash are part of the Layer-1 protocol identity.

Layer 2 evaluates candidates on a strict TeamMiniBatch12: four responsibility
rows from the same primary lane used by Local GEPA, plus four global coalition
and four global preservation rows. Every quota is exact and all twelve IDs are
unique; unavailable evidence is a construction error, never a silent backfill.

With local validation size 12, reflection minibatch 3, and metric budget 36,
the frozen no-overshoot arithmetic is 12 seed calls, 6 calls per proposal pair,
and 12 calls per accepted full-local evaluation. Thus the budget supports at
most four rejected proposals or one accepted child/generation. This is an
explicitly shallow local GEPA budget, not an implicit promise of four returned
children.
