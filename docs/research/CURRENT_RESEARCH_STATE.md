# Current Research State

The sole active research architecture is Unified Team Prompt Search.
The sole current method is V2.4 Structured System Prompt Optimization,
`unified_team_prompt_search_v2_4_structured_system_prompt`.

[Current specification](../design/CURRENT_SPEC.md) and
[V2.4 scientific contract](../design/STRUCTURED_SYSTEM_PROMPT_V24.md) define the
complete editable Role/Strategy/Answer System Prompt, raw-question User message,
natural response evidence, gold-blind extraction and bounded invalid recovery.
The existing team-level responsibility, search, plurality, admission, Shadow,
Memory transaction and atomic commit rules remain unchanged.

Current execution uses `scripts/run_experiment.py` and
`build_current_team_prompt_search`. Paid APIs require a fresh frozen exact
single-use scope. This refactor is zero API; Validation is unauthorized and Test
sealed. Initial real V2.4 performance is unmeasured. Synthetic improvements
are test fixtures. Historical code is recovered from frozen Git sources;
reports and manifests retain their original scientific meaning.
