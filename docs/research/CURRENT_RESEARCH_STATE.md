# Current Research State

The sole active research architecture is Unified Team Prompt Search.
Research claims follow [scientific principles](../design/SCIENTIFIC_PRINCIPLES.md)
and the [MATH measurement contract](../design/MATH_SCIENTIFIC_CONTRACT.md).
The sole current method is V2.5 with flexible answer extraction and bounded output recovery,
`unified_team_prompt_search_v2_6_parallel_compact_v1`.

[Current specification](../design/CURRENT_SPEC.md) and
[V2.5 scientific contract](../design/RESPONSIBILITY_FALLBACK_REPAIR_V25.md) define the
complete editable Role/Strategy/Answer System Prompt, raw-question User message,
natural response evidence, gold-blind extraction and bounded invalid recovery.
V2.5 adds seeded zero-responsibility fallback, quota-free evidence, structural
recovery, evaluated local parents and assigned repair plus independent Probe.
The [output recovery amendment](../design/GENERATED_OUTPUT_RECOVERY_V25.md) localizes
typed generated-content failures and checks symbolic overlap with source context.
Full admission, plurality, Shadow, Memory transactions and atomic commit remain.

Current execution uses `scripts/run_experiment.py` and
`build_current_team_prompt_search`. Paid APIs require a fresh frozen exact
single-use scope. The [flexible extraction amendment](../design/FLEXIBLE_ANSWER_EXTRACTION_V3.md)
accepts clear equivalent answer presentations while keeping mathematical scoring
unchanged. New paid execution remains gated; historical results retain their frozen
parser. Synthetic improvements are test fixtures. Historical code is recovered from frozen Git sources;
reports and manifests retain their original scientific meaning.

## Latest frozen development evidence and open questions

The [OpenLux Arm B Canary/Pilot report](../../reports/a4_v25_arm_b_openlux_seed84_dual_scope_execution_20261010/README.md)
was published at `30a57d82`, using frozen scientific source `01306fbf`.
Pilot deployed two Strategy edits: Optimize Vote 47/60 to 49/60 and Oracle 50/60
to 53/60. Eight member-level valid-wrong profiles became correct, while two old
correct profiles were lost. All three new Oracle-covered examples still failed
Vote; its two gains occurred inside already covered examples. This is descriptive
development evidence, not independent generalization or component causality.

The five-opportunity operational cap ended the Pilot before scientific stopping.
Whether new coverage can become Vote gain during longer optimization remains
unresolved. It does not currently justify new aggregation, mechanisms or gates.
Prioritize offline diagnosis of Gradient field-length/JSON failures and repeated
input cost. Any future generation change needs a fresh version freeze and exact
authorization; this diagnostic work grants neither a new experiment nor API access.

The [diagnostic field-length feasibility study](GRADIENT_DIAGNOSTIC_LENGTH_FEASIBILITY_V1.md)
replays original responses with all remaining validators intact. It recommends
400 characters for both diagnostic fields as a future version candidate; current
production remains at 240. Recorded local retry savings do not establish future
cost, safety, mathematical or team gains. Correction remains limited to 400.

The [diagnostic limit amendment](../design/GRADIENT_DIAGNOSTIC_LIMITS_400_V2.md) sets observed_failure and diagnosis
to 400 characters each, with a fresh policy identity. Correction, provenance,
leakage guards, Solver, aggregation, team gates and stopping remain unchanged.
No paid execution or efficacy is established by this offline adoption.

The [parallel compact amendment](../design/PARALLEL_COMPACT_OPTIMIZER_V1.md)
sets default Solver concurrency to 16 and independent Gradient concurrency to 8.
Optimizer evidence uses lossless view references; request timing is private and
nonadaptive. This new version is offline-verified preparation, with no paid efficacy
result or reusable execution authorization.
