# Unified Team Prompt Search

The sole active research architecture is Unified Team Prompt Search.
The sole current method retains V2.12 count-only exhaustive responsibility,
Group diagnosis and V2.9 Parent Population, with the explicit Hendrycks scoring
replacement: `unified_team_prompt_search_count_responsibility_hendrycks_v1`.
See the [count-only responsibility amendment](docs/design/COUNT_RESPONSIBILITY_V212.md).
See the [V2.11 auxiliary diagnosis and evidence projection amendment](docs/design/GROUP_DIAGNOSED_REFLECTION_V211.md).
See the [historical weighted responsibility amendment](docs/design/WEIGHTED_ERROR_RESPONSIBILITY_V210.md).
See the [V2.9 population amendment](docs/design/VOTE_GUIDED_PARENT_POPULATION_V29.md).

[Current specification](docs/design/CURRENT_SPEC.md) and
[V2.5 scientific contract](docs/design/RESPONSIBILITY_FALLBACK_REPAIR_V25.md) define the
complete editable Role/Strategy/Answer System Prompt, raw-question User message,
natural response evidence, gold-blind extraction and bounded invalid recovery.
Every member error has positive responsibility; bounded structural recovery remains.
V2.9 replaces the local chain/Probe with independent population repairs and
complete Optimize fitness, then jointly evaluates the final per-pool argmax team.
The [output recovery amendment](docs/design/GENERATED_OUTPUT_RECOVERY_V25.md) localizes
typed generated-content failures and checks symbolic overlap with source context.
Full admission, plurality, Shadow, Memory transactions and atomic commit remain.

Current execution uses `scripts/run_experiment.py` and
`build_current_team_prompt_search`. Paid APIs require a fresh frozen exact
single-use scope. The [flexible extraction amendment](docs/design/FLEXIBLE_ANSWER_EXTRACTION_V3.md)
accepts clear equivalent answer presentations while keeping mathematical scoring
unchanged. New paid execution remains gated; historical results retain their frozen
parser. Synthetic improvements are test fixtures. Historical code is recovered from frozen Git sources;
reports and manifests retain their original scientific meaning.

The [diagnostic limit amendment](docs/design/GRADIENT_DIAGNOSTIC_LIMITS_400_V2.md) sets observed_failure and diagnosis
to 400 characters each, with a fresh policy identity. Correction, provenance,
leakage guards, Solver, aggregation, team gates and stopping remain unchanged.
No paid execution or efficacy is established by this offline adoption.

The [direct readable evidence amendment](docs/design/DIRECT_READABLE_OPTIMIZER_EVIDENCE_V2.md)
restores original complete response and preview strings for Gradient and Reflection.
Solver concurrency 16, Gradient concurrency 8 and private request timing remain.
This version has no real execution or model capability result; old scopes stay closed.


Current MATH scoring uses only [C-EVOLVE / Hendrycks V1](docs/math_scoring_cevolve_hendrycks_v1.md).
Old scoring continuations, caches and frozen experiment authorizations cannot be
reused. This migration performs no model calls and does not resume A2 Test.
