# Unified Team Prompt Search

The sole active research architecture is Unified Team Prompt Search.
The sole current method is V2.5 with flexible mathematical answer extraction,
`unified_team_prompt_search_v2_5_flexible_answer_v1`.

[Current specification](docs/design/CURRENT_SPEC.md) and
[V2.5 scientific contract](docs/design/RESPONSIBILITY_FALLBACK_REPAIR_V25.md) define the
complete editable Role/Strategy/Answer System Prompt, raw-question User message,
natural response evidence, gold-blind extraction and bounded invalid recovery.
V2.5 adds seeded zero-responsibility fallback, quota-free evidence, structural
recovery, evaluated local parents and assigned repair plus independent Probe.
Full admission, plurality, Shadow, Memory transactions and atomic commit remain.

Current execution uses `scripts/run_experiment.py` and
`build_current_team_prompt_search`. Paid APIs require a fresh frozen exact
single-use scope. The [flexible extraction amendment](docs/design/FLEXIBLE_ANSWER_EXTRACTION_V3.md)
accepts clear equivalent answer presentations while keeping mathematical scoring
unchanged. New paid execution remains gated; historical results retain their frozen
parser. Synthetic improvements are test fixtures. Historical code is recovered from frozen Git sources;
reports and manifests retain their original scientific meaning.
