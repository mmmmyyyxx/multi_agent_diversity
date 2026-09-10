# Two-layer prompt-team optimization architecture

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

The `gepa` backend resolves the same official frozen checkout used by
`independent_gepa_repro`: GEPA v0.1.1 at commit
`b4dbb55b7601dac448cdb836d5a401ca7d9eb920`. It calls the public
`gepa.optimize()` API with instance Pareto selection, reflection minibatches,
strict subsample improvement, merge disabled, and official callbacks. GEPA's
local frontier is a single-member per-example search construct; it is never a
team candidate frontier.

Each team opportunity starts a fresh local GEPA state because a commit changes
the prompt, peers, responsibility, and residual distribution. The opaque state
slot remains available for a future, separately governed resume or memory
protocol. `NoOpContextProvider` is the only context-memory provider implemented
here.

The canonical v15 runtime remains unchanged. `two_layer_rg_gepa_v1` is an
opt-in candidate architecture pending a separately authorized API pilot. Its
remaining transitional debt is the production binding from existing runtime
responsibility objects and fixed-probe evaluators to these facades; the
backend-neutral controller and official GEPA lifecycle are already exercised
with zero-API deterministic tests.
