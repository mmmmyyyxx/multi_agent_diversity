# Current Architecture

The sole active research architecture is Unified Team Prompt Search.
Research interpretation follows [scientific principles](design/SCIENTIFIC_PRINCIPLES.md)
and the [MATH measurement contract](design/MATH_SCIENTIFIC_CONTRACT.md).
The sole current method is V2.10 weighted error responsibility with V2.9 Parent Population,
`unified_team_prompt_search_v2_10_weighted_responsibility_v1`.
See the [weighted responsibility amendment](design/WEIGHTED_ERROR_RESPONSIBILITY_V210.md).
See the [V2.9 population amendment](design/VOTE_GUIDED_PARENT_POPULATION_V29.md).

[Current specification](design/CURRENT_SPEC.md) and
[V2.5 scientific contract](design/RESPONSIBILITY_FALLBACK_REPAIR_V25.md) define the
complete editable Role/Strategy/Answer System Prompt, raw-question User message,
natural response evidence, gold-blind extraction and bounded invalid recovery.
Seeded zero-responsibility fallback and bounded structural recovery remain.
V2.9 replaces the local chain/Probe with independent population repairs and
complete Optimize fitness, then jointly evaluates the final per-pool argmax team.
The [output recovery amendment](design/GENERATED_OUTPUT_RECOVERY_V25.md) localizes
typed generated-content failures and checks symbolic overlap with source context.
Full admission, plurality, Shadow, Memory transactions and atomic commit remain.

Current execution uses `scripts/run_experiment.py` and
`build_current_team_prompt_search`. Paid APIs require a fresh frozen exact
single-use scope. The [flexible extraction amendment](design/FLEXIBLE_ANSWER_EXTRACTION_V3.md)
accepts clear equivalent answer presentations while keeping mathematical scoring
unchanged. New paid execution remains gated; historical results retain their frozen
parser. Synthetic improvements are test fixtures. Historical code is recovered from frozen Git sources;
reports and manifests retain their original scientific meaning.

The [diagnostic limit amendment](design/GRADIENT_DIAGNOSTIC_LIMITS_400_V2.md) sets observed_failure and diagnosis
to 400 characters each, with a fresh policy identity. Correction, provenance,
leakage guards, Solver, aggregation, team gates and stopping remain unchanged.
No paid execution or efficacy is established by this offline adoption.

The [direct readable evidence amendment](design/DIRECT_READABLE_OPTIMIZER_EVIDENCE_V2.md)
restores original complete response and preview strings for Gradient and Reflection.
Solver concurrency 16, Gradient concurrency 8 and private request timing remain.
This version has no real execution or model capability result; old scopes stay closed.
