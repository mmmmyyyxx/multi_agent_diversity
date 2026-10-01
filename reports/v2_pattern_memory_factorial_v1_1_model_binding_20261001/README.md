# V1.1 explicit model binding amendment — zero API

The user resolves Solver selection as `qwen3-8b` for all arms and selects
`qwen3.7-flash` for optimizer/reflection and Pattern, with provider `lwj`.
The current Solver default is corrected and the V1.1 draft binds those explicit
values. All A1–A4 × seeds 81/82/83 model rows match. The existing model/provider
check passes; this does not grant complete Unified execution admission.

The previous STOP reports remain unchanged snapshots. Their model conflict is
resolved by the new user directives, rather than relabeled in old evidence.
The current configuration's five identical initial prompts are reconstructed
without constructing providers; only their hashes are recorded here.

The protocol remains DRAFT. PREEXECUTION_SHA and EXECUTION_SHA are not created,
no authorization is consumed, and no real API, Validation, Test or download
occurs. A2C total-count/input matching clarification is pending before freeze.
Current Unified execution governance also remains to be completed.

- [Model authority amendment](model_authority_audit.json)
- [Initial team identity](initial_team_identity.json)
- [Prior report preservation](prior_evidence_preservation.json)

No push is authorized. No efficacy result has been observed.
V2_CORE_EFFECT = NOT_ESTIMATED_IN_V1_1.
