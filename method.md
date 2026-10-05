# Unified Team Prompt Search

The sole active research architecture is Unified Team Prompt Search.
This is a conceptual overview. [CURRENT_SPEC](docs/design/CURRENT_SPEC.md)
defines the scientific contract and versions.py defines runtime identities.

A benchmark adapter defines public problems, parsing, scoring and capabilities.
The current five-member prompt team and its evaluated predictions become an
immutable team-state snapshot. Diagnosis computes responsibility and feasibility;
the opportunity builder selects a member and freezes role-bearing evidence.

The current bounded Layer1 SearchEngine explores prompts within that opportunity.
All admissible scored edits remain eligible for team evaluation, including local
rejections, under the unchanged six-generation/36-metric/four-export ceilings. Candidate
exploration and deployable team transition are separate decisions.

The candidate pipeline evaluates a proposed single-member replacement with fixed
peers using the benchmark-selected aggregation, TeamProbe and Full scopes.
Transition policy applies its frozen safety/ranking rules; only its selected
winner enters the adaptive gate. An accepted winner commits atomically.

Structured history records failures, commits, latest transitions and epochs.
The current bundle binds Gradient Pattern V4 and Rolling Risk Memory. Older
Pattern/Memory/search combinations require explicit historical replay.
Global stopping uses the frozen
team-epoch rule; emergency ceilings are operational failure conditions.
Post-freeze Validation and sealed Test do not feed back into optimization.

## Current research benchmarks

MATH, IFBench and HotpotQA form the formal research suite. BBH remains a
historical development/replay benchmark. MATH uses equivalence plurality;
HotpotQA uses normalized answer plurality with question-only retrieval;
IFBench uses LLM aggregation of equal raw responses and its responsibility
policy remains HOLD. The method cores remain benchmark-neutral.

Canonical source freeze is separate from experiment-facing Optimize150,
Shadow300, Validation300 and Test300 memberships (IFBench Test294). Shadow
is adaptive; Validation is read-only after freeze; Test stays sealed. All
members, arms, seeds and adaptive re-evaluations bind Solver qwen3-8b;
optimizer/reflection and Pattern bind qwen3.7-flash. Future initial prompts
must be task-valid and frozen independently for each benchmark.

Historical paper text is preserved in
[v15 method archive](docs/archive/methods/v15_method.md); historical contracts
do not define the active method. See [open questions](docs/research/OPEN_QUESTIONS.md)
for changes requiring separate scientific identities and experiments.

## Unified Team Prompt Search V2.1

```text
Diagnosis -> Target -> Pattern diagnostic (optional)
    -> Variable evidence composition -> GEPA search
       (local survival and team candidate exposure are separate decisions)
    -> TeamProbe -> promotion -> Full -> initial competence / strict team gain -> winner-only Shadow
    -> atomic transition -> Memory outcome update (optional)
```

V2 exposes changed, contract-valid, unique, Solver-evaluated proposals even when
GEPA rejects their local improvement or they leave its frontier. GEPA internal
search remains strict. Team-level outcomes decide deployment; local scores do
not enter promotion or transition keys. Evidence uses technical minima and
budget-derived maxima; V1's 4+4+4 view remains only for explicit replay.

The target-specific mechanism diagnostic and deterministic private-success /
shared-risk memory are implemented dormant mechanisms, default disabled. They
alter reflection context only when explicitly selected and bound. Null mechanisms
add no context or tokens. Memory learns only after complete outcomes, with
prepared updates and no provider call at apply. Benchmark blockers remain in
force. V2 mechanism correctness has zero-API proofs; efficacy is unverified.

Responsibility allocates resources without opportunity equality. Pattern selects
one repair mechanism after the target; low concentration is diagnostic, never
an intervention gate. The current explicit opt-in path uses per-example textual
gradients followed by gradients-only semantic aggregation and the same support F;
see [the V4 contract](docs/design/PATTERN_GRADIENT_DISCOVERY_V4.md). Other mechanisms do not fill repair evidence. Deployment
permits incumbent member decline only above initial competence and with strict
team gain. Memory distills actual edit strategies into private success and shared
failure experience; unclassified edits produce no fabricated lesson. All claims
remain bounded to the whole method unless a separate causal comparison is run.
