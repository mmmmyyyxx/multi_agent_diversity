# Current Research State

The sole active research architecture is Unified Team Prompt Search.
The sole current method is V2.5 Responsibility Fallback and Repair Probe,
`unified_team_prompt_search_v2_5_responsibility_fallback_repair_probe`.

[Current specification](../design/CURRENT_SPEC.md) and
[V2.5 scientific contract](../design/RESPONSIBILITY_FALLBACK_REPAIR_V25.md) define the
complete editable Role/Strategy/Answer System Prompt, raw-question User message,
natural response evidence, gold-blind extraction and bounded invalid recovery.
V2.5 adds seeded zero-responsibility fallback, quota-free evidence, structural
recovery, evaluated local parents and assigned repair plus independent Probe.
Full admission, plurality, Shadow, Memory transactions and atomic commit remain.

Current execution uses `scripts/run_experiment.py` and
`build_current_team_prompt_search`. Paid APIs require a fresh frozen exact
single-use scope. The current user task authorizes one newly frozen V2.5 Canary;
Validation, Test and Pilot are unauthorized. Initial V2.5 performance is unmeasured
until that attempt completes. Synthetic improvements
are test fixtures. Historical code is recovered from frozen Git sources;
reports and manifests retain their original scientific meaning.
