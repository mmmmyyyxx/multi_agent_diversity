# Current Runtime Composition

The sole active research architecture is Unified Team Prompt Search.
V2.4 is the only current executable method, composed through
`build_current_team_prompt_search` and `CURRENT_STRUCTURED_SYSTEM_COMPOSITION_V2`.

```mermaid
flowchart TD
  B[Closed V2.4 binding] --> S[Structured System / raw problem User]
  S --> A[Gold-blind extraction and bounded recovery]
  A --> R[Member responsibility and feasibility]
  R --> G[Per-example Gradient with optional written steps]
  G --> P[Actionable clustering and Pattern responsibility]
  P --> M[Measured competence and block edit Memory]
  M --> L[Single-block bounded Layer1 search]
  L --> V[Independent Optimize SearchValidation]
  V --> T[Fixed-peer TeamProbe and Full Optimize]
  T --> D[Initial floor and target or team progress]
  D --> H[Winner-only Shadow and atomic commit]
  H --> E[Memory transaction and scientific stop]
```

`scripts/run_experiment.py` resolves only `MATHStructuredBinding`. Every Solver
stage uses the same two-message wire contract. Historical parent JSON supplies
pinned dataset/settings provenance; it cannot select an executor or supply
responses, competence or authorization. Full current policies and fresh exact
single-use scopes are mandatory. Development and preparation remain zero API.

The active scientific authority is [CURRENT_SPEC.md](CURRENT_SPEC.md), with
[the structured prompt contract](STRUCTURED_SYSTEM_PROMPT_V24.md). Retired method
implementations are recoverable from the Git sources in the cleanup tombstone.
