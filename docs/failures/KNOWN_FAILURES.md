# Known Failures

Generated from `docs/failures/registry.yaml`; edit the YAML authority, not this file.

## FAIL-TARGET-CONCENTRATION: Target allocation concentration

- Status: `DIAGNOSED`
- Evidence level: `observed`
- First observed: `v17_formal_5arm_3seed`
- Root-cause status: Allocation behavior was isolated from residual-context behavior.
- Symptom: Member-aware allocation can repeatedly concentrate proposal service.
- Forbidden inference: Do not infer that every alternative allocation improves Vote.
- Mitigation: Preserve unique routing, wait accounting, and explicit target telemetry.

## FAIL-LOW-UPDATE-THROUGHPUT: Low feasible-update throughput

- Status: `DIAGNOSED`
- Evidence level: `observed`
- First observed: `v17_formal_5arm_3seed`
- Root-cause status: Multiple upstream bottlenecks exist; no single universal cause is established.
- Symptom: Actionable branches frequently fail before a feasible update is available.
- Forbidden inference: Throughput improvement alone is not quality or final Vote improvement.
- Mitigation: Record each funnel layer separately and preserve compute parity.

## FAIL-PRE-STUDENT-CRITIC-GATE: Pre-Student semantic Critic gate bottleneck

- Status: `DIAGNOSED`
- Evidence level: `causally_supported`
- First observed: `gepa_critic_gate_failure_audit`
- Root-cause status: Shadow continuation established feasible candidate-supply loss on fixed parents.
- Symptom: Many branches stop at semantic Critic rejection before Student generation.
- Forbidden inference: Do not infer that removing the Critic improves online or validation Vote.
- Mitigation: Candidate pipeline alternatives require prospective online validation before promotion.

## FAIL-CANDIDATE-SELECTION-NOT-PRIMARY: Historical winner selection is not the primary harmful-pool bottleneck

- Status: `NOT_PRIMARY`
- Evidence level: `observed`
- First observed: `gepa_candidate_selection_audit`
- Root-cause status: Retrospective evidence excludes only the audited pools and rule.
- Symptom: GEPA-style frontier replay retained the historical winner in key harmful pools.
- Forbidden inference: Do not claim ranking is universally optimal or causally irrelevant.
- Mitigation: Keep ranking fixed while testing candidate-supply interventions.

## FAIL-FEASIBLE-SET-QUALITY: Feasible-set candidate quality gap

- Status: `DIAGNOSED`
- Evidence level: `observed`
- First observed: `v18_writeback_quality_diagnostic`
- Root-cause status: Candidate breadth remained untested where the pre-Student gate blocked generation.
- Symptom: Harmful accepted pools contained no zero-loss feasible alternative.
- Forbidden inference: Do not label proposal breadth ineffective when Student was never reached.
- Mitigation: Separate reach, validity, feasibility, and selection in the funnel.

## FAIL-WRITEBACK-TRANSFER: Train-safe write-back can transfer poorly to validation

- Status: `DIAGNOSED`
- Evidence level: `observed`
- First observed: `v18_online_accumulation`
- Root-cause status: Transfer and trajectory overwrite are measured; a train-only discriminator is not established.
- Symptom: Some train-beneficial commits cause validation collateral or later overwrite.
- Forbidden inference: Do not use validation outcomes inside candidate acceptance.
- Mitigation: Report train-to-validation transfer separately from write-back legality.

## FAIL-TEACHER-PRESERVATION-PATTERN: Stable Teacher preservation-rule safety marker pattern

- Status: `DIAGNOSED`
- Evidence level: `observed`
- First observed: `historical_teacher_safety_audit`
- Root-cause status: Pattern is stable but does not discriminate canonical Critic pass from block.
- Symptom: Historical deterministic markers concentrate in preservation_rule and repeat after retry.
- Forbidden inference: Do not infer that preservation_rule alone causes semantic rejection.
- Mitigation: Record field-local safety categories prospectively before changing the schema.

## FAIL-DETERMINISTIC-SAFETY-OVERBROAD-RISK: Deterministic safety classifier may over-block benign wording

- Status: `OPEN`
- Evidence level: `hypothesized`
- First observed: `safety_only_prospective_pilot`
- Root-cause status: Historical regression is low for the selected candidate gate, but broader sufficiency is unproven.
- Symptom: A strict rule can classify preservation wording as output-contract or copying risk.
- Forbidden inference: Do not call the deterministic gate generally safe from six passing prospective plans.
- Mitigation: Keep conservative lexical rules and historical false-positive regression bounds.

## FAIL-SEMANTIC-CRITIC-OVERFILTERING: Semantic Critic removes some useful fixed-parent opportunities

- Status: `DIAGNOSED`
- Evidence level: `causally_supported`
- First observed: `shadow_raw_critic_pilot`
- Root-cause status: Fixed-parent candidate-supply effect is supported; trajectory efficacy is untested.
- Symptom: Canonically rejected plans produced feasible candidates under no-commit shadow continuation.
- Forbidden inference: Do not claim Critic removal improves Vote, test, or online trajectories.
- Mitigation: Candidate C remains selected only for future online validation.
