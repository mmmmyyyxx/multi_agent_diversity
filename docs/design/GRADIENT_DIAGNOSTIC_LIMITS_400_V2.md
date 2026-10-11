# Gradient diagnostic limits 400 V2

This amendment is normative for
`unified_team_prompt_search_v2_5_diagnostic_limits_400_v2`. The user selected
400 after the [zero-API threshold replay](../../reports/gradient_diagnostic_threshold_replay_20261011/README.md).
It changes only the upper bounds for `observed_failure` and `diagnosis`, from
240 to 400 characters each. Bounds apply inclusively to the original decoded
string using Python `len`, including whitespace and Unicode characters.
Neither truncation nor deletion, JSON repair or skipped checks is permitted.

These fields describe an observation and a repair hypothesis. They are private
diagnostic evidence, not proven causal attribution or Pattern repair text.
`reusable_correction` retains its 400-character limit and every abstraction,
source, specific-answer and leakage guard. `expected_effect` retains 240.
The exact object schema, ACTIONABLE/UNCERTAIN semantics, trajectory and reference
provenance, three-draw recovery bound and first-valid-response selection remain.
Clustering receives validated corrections only; diagnostics do not enter Solver.

Prompt and Validator read the same versioned `GRADIENT_OUTPUT_LIMITS` policy.
The complete limits enter Pattern policy and method/binding identity. Changes
require a new frozen identity; arbitrary limits cannot execute as this version.

| Scope | Fresh identity |
|---|---|
| Method | unified_team_prompt_search_v2_5_diagnostic_limits_400_v2 |
| Runtime composition | CURRENT_DIAGNOSTIC_400_COMPOSITION_V6 |
| Binding | MATH_DIAGNOSTIC_400_ROLE_ROUTED_BINDING_V6 |
| Cache | ROLE_ROUTED_DIAGNOSTIC_400_MEMBER_LANE_CACHE_V7 |
| Per-example Gradient | PER_EXAMPLE_DIAGNOSTIC_400_GRADIENT_V4 |
| Pattern discovery | gradient_diagnostic_400_partial_partition_v7 |
| Pattern schema | GRADIENT_DIAGNOSTIC_400_PARTITION_V5 |
| Gradient prompt | STRUCTURED_SYSTEM_GRADIENT_PROMPT_V8 |
| Field limits | GRADIENT_DIAGNOSTIC_FIELD_LIMITS_V2 |
| Output schema | OBSERVATION_HYPOTHESIS_BLOCK_CORRECTION_EFFECT_V3 |
| Evidence binding | DIAGNOSTIC_400_EVIDENCE_POLICY_V6 |
| Trajectory policy binding | MATH_DIAGNOSTIC_400_OBSERVED_RESPONSE_POLICY_V5 |

The trajectory payload schema and capacity are unchanged; its policy identity
changes because it binds the new Gradient prompt. Provider routing, models,
Solver, aggregation, Layer1, Full/Shadow gates, competence floor and
`team_epoch_no_commit_v1` retain their existing contracts. Unified Team Prompt
Search remains the architecture; response eligibility is a scientific policy
change and prospective results require this new version.

The retrospective Pilot replay at 400 yielded 94 fully valid physical responses,
60 logical diagnoses with a valid response and 39 first-draw passes. Its local
retry suffix estimate was 55 draws / 224175 usage tokens. These are frozen
counterfactual observations, not prospective savings, billing or efficacy.
Longer diagnostics may change corrections and downstream search; token usage,
leakage prevalence and team performance require new evidence.

All historical identities, manifests, responses, scores and classifications
remain frozen. Replay uses their exact source, including the original 240 bounds.
Old scopes, caches and checkpoints cannot be promoted to current execution.
Preparation is DRAFT with no paid attempt, bound execution artifact or API grant.
A new source/hash freeze, membership and budget review, concrete handoff and
exact single-use authorization are required before any real run. This task
performs only guarded offline checks and accesses no Validation or Test data.
