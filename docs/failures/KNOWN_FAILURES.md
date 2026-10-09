# Known Failures

Generated from `docs/failures/registry.yaml`; edit the YAML authority, not this file.

## FAIL-V23-SOLVER-SERIAL-LOCK-AND-LENGTH-OUTPUT: Solver network calls were serialized and length failures mixed long derivation with repeated reasoning

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `a4_v23_only_pilot_v1`
- Root-cause status: CONFIRMED_SERIAL_LOCK_AND_OBSERVED_MIXED_OUTPUT_FAILURES
- Symptom: A global lock covered the entire network call. Historical full and partial attempts include 26 and 32 length responses, all without a final marker.
- Forbidden inference: Synthetic throughput is not provider throughput. More capacity is not proven to repair repeated reasoning, improve accuracy or generate commits. Old API scopes remain closed.
- Mitigation: Explicit fresh eight-worker execution, immutable compatible initial prefixes and bounded truncation-only 6144 recovery. Keep scoring, instructions and four total draws unchanged.

## FAIL-V23-CANARY-OWNER-REVIEW-TIMEOUT: Completed initial Solver profiles waited without an owner review decision

- Status: `DIAGNOSED`
- Lifecycle: `DIAGNOSED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `a4_v23_only_pilot_v1`
- Root-cause status: CONFIRMED_MISSING_OWNER_DECISION_BEFORE_FROZEN_DEADLINE
- Symptom: All 300 initial profiles completed; no owner decision arrived within the frozen 3600-second deadline, so execution aborted before optimization.
- Forbidden inference: No Gradient, mutation, Full, Shadow or commit was observed. Do not infer method failure, approve the terminated attempt retrospectively, or reuse the consumed scope/cache/remaining allowance. The monitor account-error timing is not established.
- Mitigation: Direct owner supervision in a fresh explicitly approved handoff; bounded monitoring and immediate independent review. Keep timeout, receipt binding and consumed-scope rejection unchanged.

## FAIL-A4-OPTIMIZE-ROLE-OVERLAP-EDIT-EVIDENCE-GAP: Local validation reused mutation examples and Full rejection lacked complete edit evidence

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_2_gradient_pattern_seed81_pilot_v2`
- Root-cause status: CONFIRMED_ROLE_OVERLAP_AND_INFORMATION_OMISSION_NOT_OPTIMIZER_CAUSALITY
- Symptom: Seven completed opportunities reused panel membership for Probe and the same correct anchor; measured Full losses were outside that panel.
- Forbidden inference: Synthetic conformance does not show a real optimizer improvement, establish hidden Solver causes, or authorize any real API or protected split access.
- Mitigation: Explicit V2.3 disjoint Optimize roles, rotating current-correct pool and scope-bound actual-diff Memory; historical execution unchanged.

## FAIL-GRADIENT-DIAGNOSTIC-ARCHIVE-IDENTITY-FIELD: Matched raw diagnostic confused archive schema identity with discovery policy identity

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_1_gradient_pattern_seed81_canary_v2`
- Root-cause status: CONFIRMED_IMPLEMENTATION_MISMATCH
- Symptom: Pre-provider identity failure despite exact archived prompt hash; zero diagnostic calls.
- Forbidden inference: Do not alter scientific prompts, guard thresholds, partition semantics, sample selection, budgets or regeneration policy to repair this metadata error.
- Mitigation: Check archive schema and discovery policy fields separately; preserve frozen prompts and evidence.

## FAIL-GRADIENT-COORDINATED-ACTOR-ABSTRACTION-COVERAGE: Coordinated participant constraint bypassed the specific-content abstraction guard

- Status: `MITIGATED`
- Lifecycle: `OPEN`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_1_gradient_pattern_seed81_canary_v1`
- Root-cause status: CONFIRMED_IMPLEMENTATION_COVERAGE_GAP
- Symptom: A copied participant literal in a coordinated refusal-to-participate context passed V3.
- Forbidden inference: Do not accept example entities, widen the hard 400-character gradient limit, add regeneration, or infer a hidden reasoning trace to resolve contract noncompliance.
- Mitigation: V4 current guard adds coordinated participation constraints; V3 replay helper remains unchanged.

## FAIL-UNIFIED-METHOD-SEMANTIC-DRIFT: Historical V2 implementation differs from the newly supplied method semantic contract

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: ZERO_API_CONFORMANCE_ONLY; REAL_EXECUTION_AND_EFFICACY_UNVERIFIED
- Evidence level: `observed`
- First observed: `unified_semantic_contract_v2_1`
- Root-cause status: Historical policy definitions were narrower or different than the user-authored current research contract.
- Symptom: Assigned-only DPR, other-repair technical backfill, incumbent monotonicity and fixed-template memory contradict the current semantic contract.
- Forbidden inference: Fake conformance does not prove real efficacy, memory transfer benefit, counterfactual allocation optimality or independent component causality.
- Mitigation: Fresh V2.1 policies and runtime identity; old frozen V2 artifacts preserved; new real execution requires fresh binding and authorization.

## FAIL-TARGET-CONCENTRATION: Target allocation concentration

- Status: `DIAGNOSED`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `observed`
- First observed: `v17_formal_5arm_3seed`
- Root-cause status: Allocation behavior was isolated from residual-context behavior.
- Symptom: Member-aware allocation can repeatedly concentrate proposal service.
- Forbidden inference: Do not infer that every alternative allocation improves Vote.
- Mitigation: Preserve unique routing, wait accounting, and explicit target telemetry.

## FAIL-LOW-UPDATE-THROUGHPUT: Low feasible-update throughput

- Status: `DIAGNOSED`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `observed`
- First observed: `v17_formal_5arm_3seed`
- Root-cause status: Multiple upstream bottlenecks exist; no single universal cause is established.
- Symptom: Actionable branches frequently fail before a feasible update is available.
- Forbidden inference: Throughput improvement alone is not quality or final Vote improvement.
- Mitigation: Record each funnel layer separately and preserve compute parity.

## FAIL-PRE-STUDENT-CRITIC-GATE: Pre-Student semantic Critic gate bottleneck

- Status: `DIAGNOSED`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `causally_supported`
- First observed: `gepa_critic_gate_failure_audit`
- Root-cause status: Shadow continuation established feasible candidate-supply loss on fixed parents.
- Symptom: Many branches stop at semantic Critic rejection before Student generation.
- Forbidden inference: Do not infer that removing the Critic improves online or validation Vote.
- Mitigation: Candidate pipeline alternatives require prospective online validation before promotion.

## FAIL-CANDIDATE-SELECTION-NOT-PRIMARY: Historical winner selection is not the primary harmful-pool bottleneck

- Status: `NOT_PRIMARY`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `observed`
- First observed: `gepa_candidate_selection_audit`
- Root-cause status: Retrospective evidence excludes only the audited pools and rule.
- Symptom: GEPA-style frontier replay retained the historical winner in key harmful pools.
- Forbidden inference: Do not claim ranking is universally optimal or causally irrelevant.
- Mitigation: Keep ranking fixed while testing candidate-supply interventions.

## FAIL-FEASIBLE-SET-QUALITY: Feasible-set candidate quality gap

- Status: `DIAGNOSED`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `observed`
- First observed: `v18_writeback_quality_diagnostic`
- Root-cause status: Candidate breadth remained untested where the pre-Student gate blocked generation.
- Symptom: Harmful accepted pools contained no zero-loss feasible alternative.
- Forbidden inference: Do not label proposal breadth ineffective when Student was never reached.
- Mitigation: Separate reach, validity, feasibility, and selection in the funnel.

## FAIL-WRITEBACK-TRANSFER: Train-safe write-back can transfer poorly to validation

- Status: `DIAGNOSED`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `observed`
- First observed: `v18_online_accumulation`
- Root-cause status: Transfer and trajectory overwrite are measured; a train-only discriminator is not established.
- Symptom: Some train-beneficial commits cause validation collateral or later overwrite.
- Forbidden inference: Do not use validation outcomes inside candidate acceptance.
- Mitigation: Report train-to-validation transfer separately from write-back legality.

## FAIL-TEACHER-PRESERVATION-PATTERN: Stable Teacher preservation-rule safety marker pattern

- Status: `DIAGNOSED`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `observed`
- First observed: `historical_teacher_safety_audit`
- Root-cause status: Pattern is stable but does not discriminate canonical Critic pass from block.
- Symptom: Historical deterministic markers concentrate in preservation_rule and repeat after retry.
- Forbidden inference: Do not infer that preservation_rule alone causes semantic rejection.
- Mitigation: Record field-local safety categories prospectively before changing the schema.

## FAIL-DETERMINISTIC-SAFETY-OVERBROAD-RISK: Deterministic safety classifier may over-block benign wording

- Status: `OPEN`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `hypothesized`
- First observed: `safety_only_prospective_pilot`
- Root-cause status: Historical regression is low for the selected candidate gate, but broader sufficiency is unproven.
- Symptom: A strict rule can classify preservation wording as output-contract or copying risk.
- Forbidden inference: Do not call the deterministic gate generally safe from six passing prospective plans.
- Mitigation: Keep conservative lexical rules and historical false-positive regression bounds.

## FAIL-SEMANTIC-CRITIC-OVERFILTERING: Semantic Critic removes some useful fixed-parent opportunities

- Status: `DIAGNOSED`
- Lifecycle: `HISTORICAL`
- Lifecycle context: Historical observed facts retained; not an active Unified scientific decision.
- Evidence level: `causally_supported`
- First observed: `shadow_raw_critic_pilot`
- Root-cause status: Fixed-parent candidate-supply effect is supported; trajectory efficacy is untested.
- Symptom: Canonically rejected plans produced feasible candidates under no-commit shadow continuation.
- Forbidden inference: Do not claim Critic removal improves Vote, test, or online trajectories.
- Mitigation: Candidate C remains selected only for future online validation.

## FAIL-GEPA-SERIAL-EVALUATION: GEPA synchronous local evaluation serializes Solver batches

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: MITIGATED_IMPLEMENTATION; REAL_PERFORMANCE_UNVERIFIED. Fake equivalence and concurrency only.
- Evidence level: `observed`
- First observed: `formal_v3_attempt3_seed80_preliminary`
- Root-cause status: Serial batch submission replaced by ordered existing gather path; deterministic fake equivalence passed.
- Symptom: Old GEPA local adapter had effective concurrency 1 despite a higher configured semaphore.
- Forbidden inference: Do not infer real throughput, efficacy or API authorization from fake concurrency calibration.
- Mitigation: Use the versioned current batch path only within separately frozen and authorized execution.

## FAIL-MATH-CANARY-FINAL-MARKER: MATH canary initial Solver response omitted final marker

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: ZERO_API_INTERFACE_CONFORMANCE_ONLY; REAL_OUTPUT_ADHERENCE_UNVERIFIED
- Evidence level: `observed`
- First observed: `math_v2_a1_seed81_real_canary_v1`
- Root-cause status: R2 proven by exact request reconstruction: system layer contained one marker contract; response contained zero markers. Underlying model noncompliance cause remains unproven.
- Symptom: First qwen3-8b response contained zero FINAL_ANSWER marker lines; frozen parser rejected without regeneration.
- Forbidden inference: Do not infer mathematical correctness, search effectiveness or generalization from this aborted canary.
- Mitigation: Versioned immutable formatting interface and request/cache identity under zero-API tests. Real compliance unverified; parser unchanged. No old-attempt continuation.

## FAIL-MATH-AUTONOMOUS-TOKEN-ACCOUNTING-UNBOUND: Autonomous real-token ceiling lacks a frozen provider accounting upper bound

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: RESOLVED_BY_USER_ACCOUNTING_POLICY_V2; NOT_PROVIDER_BILLING_PROOF
- Evidence level: `observed`
- First observed: `math_v2_autonomous_accounting_preflight_v1`
- Root-cause status: Frozen lwj binding has no tokenizer identity, framing bound, alias accounting binding or failed-request accounting contract. Validation required reserve is also unresolved from metadata alone.
- Symptom: Existing call-count ceilings and output request limit do not establish a reliable total billable-token bound before each transport.
- Forbidden inference: Do not infer provider billing bounds from upstream model names, successful historical usage or an arbitrary characters-per-token multiplier. No efficacy was observed.
- Mitigation: User explicitly authorizes RESERVATION_V2 operational byte-plus-margin accounting, full fallback charge and isolated Validation length preparation. Historical STOP evidence remains immutable. New real calls require fresh source and exact attempt admission.

## FAIL-MATH-SDK-HTTP-ERROR-MAPPING: Installed SDK error mapping rejected an unsupported keyword

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: ZERO_API_ERROR_MAPPING_CONFORMANCE_ONLY; REAL_PROVIDER_ACCESS_UNVERIFIED
- Evidence level: `observed`
- First observed: `math_v2_a1_seed81_real_canary_v2`
- Root-cause status: Installed SDK accepts response only; body keyword unsupported. Original HTTP status is unknown.
- Symptom: First HTTP error response raised TypeError before SDK status classification and response persistence.
- Forbidden inference: No Solver output, opportunity, scientific efficacy or provider denial cause is established.
- Mitigation: Use the installed SDK response-only signature and persist private HTTP error evidence before terminal abort or frozen retry.

## FAIL-MATH-SDK-AUTH-HEADER-OMITTED: Manual exact-body transport omitted SDK bearer authentication

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: ZERO_API_TRANSPORT_CONFORMANCE_ONLY; REAL_PROVIDER_ACCESS_UNVERIFIED
- Evidence level: `observed`
- First observed: `math_v2_provider_error_mapping_repair_v1`
- Root-cause status: Installed SDK inspection and a synthetic offline negative control prove the missing header. The original failed request HTTP status remains unknown.
- Symptom: Installed SDK default headers exclude authentication; the old manual request copied only those headers.
- Forbidden inference: Do not infer a historical HTTP status, credential validity, provider availability or scientific efficacy from this offline reconstruction.
- Mitigation: Build headers and URL through ordinary SDK security handling, preserve exact reserved body bytes, pin installed SDK version, and deny uncounted redirects.

## FAIL-MATH-REFERENCE-PARSER-DOMAIN-MISMATCH: Frozen nonempty MATH references exceed the scoring parser domain

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: SOURCE_GROUNDED_SCIENTIFIC_AMENDMENT_ZERO_API_PASS; REAL_EXECUTION_AND_EFFICACY_UNVERIFIED
- Evidence level: `observed`
- First observed: `math_v2_a1_seed81_real_canary_v3`
- Root-cause status: Reference admission checks nonempty extraction; scoring requires payload_supported for both operands. Nine Optimize references fail the lexical guard, six the tuple guard.
- Symptom: Final-marker compliant Solver response and its admitted Optimize reference both trigger the ordered-tuple payload guard.
- Forbidden inference: No mathematical correctness, answer equivalence, efficacy or generalization is inferred from representation support.
- Mitigation: Explicit user-authorized V2 native mathematical parsing, compatible object families and full-source scorable eligibility. Fresh versioned split and exact new phase freeze required; historical Canary V3 remains INVALID.

## FAIL-MATH-V21-FINAL-MARKER-OMITTED: Solver normally completed without the frozen final marker

- Status: `OPEN`
- Lifecycle: `OPEN`
- Lifecycle context: FAILED_ATTEMPT_PRESERVED; REPAIR_PENDING
- Evidence level: `observed`
- First observed: `math_v2_1_a1_seed81_real_canary_v1`
- Root-cause status: OUTPUT_FORMAT_NONCOMPLIANCE; parser correctly fails closed; provider cause unknown.
- Symptom: 27th Solver response has zero FINAL_ANSWER markers and finish_reason stop below the output cap.
- Forbidden inference: No mathematical correctness, method efficacy or causal claim is established.
- Mitigation: Separate immutable interface version; strict parser unchanged; fresh source, authorization and process.

## FAIL-MATH-V21-SOLVER-OUTPUT-TRUNCATION: Solver exhausted its frozen output cap during initialization

- Status: `OPEN`
- Lifecycle: `OPEN`
- Lifecycle context: STOP_NEW_METHOD_SCIENTIFIC_POLICY_REQUIRED; FAILED_ATTEMPTS_PRESERVED; NO_PILOT_OR_VALIDATION_EXECUTION
- Evidence level: `observed`
- First observed: `math_v2_1_a1_seed81_real_canary_v3`
- Root-cause status: OUTPUT_TRUNCATION_AND_REPETITIVE_STRUCTURE_OBSERVED; provider/backend cause unknown; no serialization mismatch found.
- Symptom: Length termination at cap 1800 in Canary v3 and cap 3600 in Canary v4; both before completed initialization.
- Forbidden inference: No method efficacy or performance-driven budget selection is established.
- Mitigation: Preserve strict parser, failed responses, frozen initial prompts and durable ledger. Cap enlargement failed. Task clause 147 stops before unapproved generation policy changes.

## FAIL-MATH-V21-FROZEN-DECODING-V1: Fresh frozen Solver decoding attempt failed closed

- Status: `OPEN`
- Lifecycle: `OPEN`
- Lifecycle context: STOP_SOLVER_DECODING_POLICY_INSUFFICIENT
- Evidence level: `observed`
- First observed: `math_v2_1_a1_seed81_real_canary_v5`
- Root-cause status: Exact client request policy verified; provider internal cause unverified.
- Symptom: STOP_SOLVER_DECODING_POLICY_INSUFFICIENT
- Forbidden inference: No efficacy conclusion or parameter adaptation.
- Mitigation: Preserve evidence and ledger; user scientific decision required.

## FAIL-MATH-V21-OPTIMIZER-TOTAL-OUTPUT-CAP: Optimizer reported total output exceeds the frozen answer cap assumption

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: APPROVED_WIRE_CORRECTION_IMPLEMENTED_AND_DISPATCH_VERIFIED; HISTORICAL_FAILURE_AND_FALLBACK_CHARGE_IMMUTABLE
- Evidence level: `observed`
- First observed: `math_v2_1_a1_seed81_low_cost_canary_v1`
- Root-cause status: Observed cap/usage dimension mismatch. Qwen documentation distinguishes answer-only max_tokens from combined max_completion_tokens. The actual provider internal cause is unverified; detailed usage was not retained.
- Symptom: Frozen max_tokens=1800 request returned finish_reason stop with 1814 reported output tokens; first reflection aborted before proposal admission.
- Forbidden inference: No backend thinking-token count, method efficacy, generalization or performance-based tuning is inferred.
- Mitigation: Preserve failed attempt and full fallback charge, close scopes, and require explicit optimizer generation/accounting policy decision before fresh freeze and execution.

## FAIL-MATH-V21-OPTIMIZER-TOTAL-COMPLETION-TRUNCATION: Correctly dispatched optimizer completion cap exhausted before answer

- Status: `OPEN`
- Lifecycle: `OPEN`
- Lifecycle context: STOP_NEW_SCIENTIFIC_POLICY_REQUIRED; FAILED_CANARY_IMMUTABLE; PILOT_AND_VALIDATION_NOT_EXECUTED
- Evidence level: `observed`
- First observed: `math_v2_1_a1_seed81_low_cost_canary_v2`
- Root-cause status: Frozen wire policy dispatched exactly; 1810 accounting/verification ceiling respected. Existing truncation guard aborted. No client implementation or serialization defect found.
- Symptom: Explicit thinking=true/max_completion_tokens1800 request returned completion_tokens1800/reasoning_tokens1800, empty answer and finish_reason length.
- Forbidden inference: No method efficacy, generalization or provider over-cap violation is established. The10-token measurement tolerance does not accept a truncated generation.
- Mitigation: Close failed scope and undispatched Pilot, preserve all evidence/charges, stop for a new user optimizer generation-policy decision. Do not tune, rerun, enlarge cap or disable thinking autonomously.

## FAIL-MATH-V21-OPTIMIZER-NONTHINKING-TELEMETRY-UNCONFIRMED: Frozen nonthinking confirmation requires unavailable provider telemetry

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: STOP_NEW_SCIENTIFIC_POLICY_REQUIRED; NO_CANARY_PILOT_VALIDATION_OR_TEST_DISPATCH
- Evidence level: `observed`
- First observed: `math_v2_1_optimizer_truncation_root_cause_v1`
- Root-cause status: No metadata parser loss found. Missing reasoning_tokens cannot satisfy task clause 16 actual-zero evidence requirement. Truncation and repetition were not reproduced.
- Symptom: Explicit false witness returned stop with 384 completion tokens and absent reasoning_content, but raw usage omitted reasoning_tokens.
- Forbidden inference: Do not infer missing counts as zero, hidden thinking as present, provider cap violation, model efficacy or an admissible candidate.
- Mitigation: Close single-use witness authorization, preserve diagnostics and ledger, keep both generation policies and validators unchanged; require a human evidence-policy decision before fresh Canary.

## FAIL-MATH-V21-PROPOSER-MUTABLE-GUARD-MISMATCH: Complete optimizer proposals do not reach admissible mutable candidates

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: Versioned semantic boundary and Layer1 repairs passed offline regression and a fresh real Canary; six admissible proposals reached local Solver scoring and four reached team evaluation. One complete production opportunity ended without commit because its selected candidate failed the frozen Shadow gate. Historical Pilot invalid-proposal evidence remains immutable; scientific efficacy is not estimated.
- Evidence level: `observed`
- First observed: `math_v2_1_a1_seed81_low_cost_pilot_v4`
- Root-cause status: Six output-control, three answer-representation and one benign proposal under owner semantic rubric; lexical overreach coexists with real proposer violations and five append-only failures.
- Symptom: 10 physical proposals become 28 contract-invalid logical proposals; zero export/team evaluation/commit.
- Forbidden inference: Neutral identity comparison does not identify efficacy; one finite owner annotation is not a population false-positive rate; phrase masking is not a safe admission repair.
- Mitigation: Versioned proposer/validator semantic alignment, including mathematical representation boundary; preserve output-interface isolation, replacement and example-copying checks.

## FAIL-MATH-PATTERN-LONG-SUPPORT-ID-COPY: Pattern support identifier loses characters on the provider wire

- Status: `OPEN`
- Lifecycle: `OPEN`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_1_a4_seed81_pattern_canary_v1`
- Root-cause status: Provider identifier copy loss; original universe and client parser were correct.
- Symptom: One of seven supplied eighty-character IDs was returned with two characters omitted; strict membership validation rejected the unknown ID and missing original.
- Forbidden inference: No fuzzy ID repair, semantic reassignment, parser loosening or efficacy inference.
- Mitigation: Explicit opt-in lossless support-ID alias transport, strict exact decoding, fresh identity/source/preexecution/authorization and fresh Canary under user attachment section 102.

## FAIL-MATH-PATTERN-GENERIC-COMMAND-NAME-GUARD: Generic mathematical imperative misclassified as example-specific proper name

- Status: `MITIGATED`
- Lifecycle: `OPEN`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_1_a4_seed81_pattern_canary_v2`
- Root-cause status: CONFIRMED_IMPLEMENTATION_FALSE_POSITIVE
- Symptom: Generic mathematical command tokens were treated as example-specific proper names and rejected by the abstraction guard.
- Forbidden inference: Do not loosen example copying, answer, numeric, interface or proper-name guards; do not tune Pattern prompt or sampling.
- Mitigation: Opt-in versioned generic mathematical-command classification; new freeze, authorization and fresh Canary.

## FAIL-MATH-GRADIENT-GENERIC-DIGIT-CONTRACT: A generic mathematical numeric literal violates the frozen absolute digit prohibition

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_1_gradient_pattern_seed81_pilot_v1`
- Root-cause status: GENERATED_CONTRACT_NONCOMPLIANCE_FROZEN_GUARD_ENFORCED
- Symptom: Fourth gradient is267 characters and valid JSON, but contains a numeric literal in a general mathematical rule; search aborts before clustering.
- Forbidden inference: No confirmed implementation bug, copied problem-specific value, efficacy or population compliance rate; no automatic retry or parser loosening.
- Mitigation: Preserve the aborted prefix, close consumed and unused conditional retry scopes, stop before new API calls.

## FAIL-MATH-WINDOWS-TOKEN-SNAPSHOT-REPLACE: Windows atomic token snapshot replacement aborts fresh Pilot initialization

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: Append-only monitor fix verified offline; incident holder and fresh runtime persistence closure remain unestablished.
- Evidence level: `observed`
- First observed: `math_v2_1_gradient_pattern_seed81_pilot_v2`
- Root-cause status: CONFIRMED_ATOMIC_REPLACEMENT_FAILURE_SITE_ACTUAL_LOCK_HOLDER_NOT_ESTABLISHED
- Symptom: WinError5 in atomic derived snapshot replacement after a durable RESPONSE charge;263/300 initial evaluations, no Gradient observations.
- Forbidden inference: Do not claim proven monitor causality, complete persistence closure, new guard noncompliance, convergence, efficacy or completed Pilot; do not reuse consumed authorization.
- Mitigation: Preserve raw evidence and the one-response evidence gap; replay unchanged authoritative journal into derived snapshot; close one-shot scope. Monitor reads append-only journal instead of live atomic files;100 isolated replacements pass.

## FAIL-MATH-GRADIENT-PARTITION-MISSING-ALIASES: Generated gradient partition omits supplied aliases without unassigned membership

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_1_gradient_pattern_seed81_pilot_v3`
- Root-cause status: GENERATED_PARTITION_CONTRACT_NONCOMPLIANCE_NO_IMPLEMENTATION_BUG_ESTABLISHED
- Symptom: Second cluster returned12groups covering36of38aliases, omitted e2/e24 and left unassigned_ids empty.
- Forbidden inference: Do not classify this as operational invalidity or completed Pilot; do not silently complete membership, regenerate, loosen parsing, or reuse conditional operational retry scope.
- Mitigation: Preserve raw response, charge and prefix; close exact single-use scope; request separate user scientific contract decision.

## FAIL-MATH-GRADIENT-V5-NUMERIC-PROVENANCE-PILOT4: Fresh Pilot stops on a short Gradient rejected by frozen numeric provenance guard

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_1_gradient_pattern_seed81_pilot_v4`
- Root-cause status: CONFIRMED_COPIED_NUMERIC_EXPRESSION_LITERAL_IN_ATTEMPT4
- Symptom: Second opportunity sixth Gradient (44th total),139characters, valid JSON and within hard400, rejected by unchanged numeric provenance guard; second cluster not reached.
- Forbidden inference: Do not label operational invalidity, a partition-completion bug, a complete valid Pilot, comprehensive semantic provenance, population compliance or causal efficacy; no fresh retry under engineering authority.
- Mitigation: Preserve raw response and charged prefix, close consumed scope, retain frozen Gradient V1 Prompt V3 Guard V5; any continuation requiring admissibility changes needs a separate scientific decision.

## FAIL-MATH-PILOT-POST-STOP-TERMINAL-SERIALIZATION: Terminal metadata keyword collision aborts runner after complete scientific search

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_1_gradient_pattern_seed81_pilot_v6`
- Root-cause status: CONFIRMED_POST_SCIENTIFIC_STOP_IMPLEMENTATION_DEFECT
- Symptom: Complete ten-opportunity trace and final state persist, then duplicate validation_status keyword raises TypeError.
- Forbidden inference: Do not rewrite raw lifecycle, invent original completion receipts, claim causal efficacy, or rerun the complete zero-commit scientific result.
- Mitigation: Use explicit metadata merge and canonical JSON-compatible summary; verify both terminal branches with fake providers.

## FAIL-IDENTICAL-INITIAL-STRICT-VOTE-DEADLOCK: Single-member progress cannot deploy against four identical wrong peer votes

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: IMPLEMENTED_ZERO_API_ONLY; REAL_EXECUTION_HOLD
- Evidence level: `observed`
- First observed: `transition_reachability_audit_v1`
- Root-cause status: Strict-Vote deployment semantics; coupling in winner key and N-commit bound.
- Symptom: V2.1 requires strict team Vote gain, preventing first commit when peer outcomes are identical; a target-only gain cannot outvote four peers.
- Forbidden inference: Byte-identical prompts do not guarantee identical stochastic member outputs; zero-API conformance proves neither real efficacy nor generalization.
- Mitigation: Fresh V2.2 OR-progress identity and target-second ranking; execution and Pilot bounds fail closed until separately frozen. Historical V2.1 is preserved.

## FAIL-V22-PILOT-RESOURCE-BOUND-NOT-FROZEN: Commit finiteness does not establish a resource-realistic opportunity bound

- Status: `MITIGATED`
- Lifecycle: `MITIGATED`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_2_a4_pilot_execution_audit_v1`
- Root-cause status: Conservative carried-failure recurrence grows beyond realistic execution resources; tighter bound unresolved.
- Symptom: A conservative finite commit bound did not establish a resource-realistic provider opportunity ceiling.
- Forbidden inference: Not an empirical A4 failure or proof that no tighter bound exists.
- Mitigation: Retain HOLD; do not guess a ceiling or change scientific allocation/stopping.

## FAIL-V22-PATTERN-CLUSTER-1800-OUTPUT-TRUNCATION: First real fixed-horizon cluster response reached its frozen output cap

- Status: `OPEN`
- Lifecycle: `OPEN`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `math_v2_2_gradient_pattern_seed81_pilot_v1`
- Root-cause status: Observed response reached the frozen 1800-token request cap; no claim that every cluster set needs a larger cap.
- Symptom: 38 accepted Gradient records entered clustering; provider returned finish_reason=length at 1800 output tokens before opportunity creation.
- Forbidden inference: Not a scientific zero-commit result, convergence, V2.2 transition failure, guard failure, or authorization to retry or increase generation limits.
- Mitigation: Fail closed; preserve paid response and accounting; consumed one-attempt scope remains closed.

## FAIL-V24-CANARY-FORMAT-COVERAGE-FEASIBILITY: Canary format rejection leaves insufficient correct evidence to enter structured prompt search

- Status: `DIAGNOSED`
- Lifecycle: `OPEN`
- Lifecycle context: See evidence below.
- Evidence level: `observed`
- First observed: `a4_v24_structured_canary_v1`
- Root-cause status: Frozen boundary semantics explain observed categorical rejection; the mechanical feasibility stop is confirmed. Latent mathematical accuracy and remedy effectiveness remain unestablished.
- Symptom: 45/60 terminal initial profiles invalid after four draws; five members each score 3/12, below the frozen minimum six correct examples. No optimization opportunity occurs.
- Forbidden inference: Not optimizer failure, convergence, missing visible reasoning, a gate implementation defect, or evidence that only three mathematical answers were intrinsically correct. No full Pilot result or permission to rerun follows.
- Mitigation: Preserve the valid Canary evidence and stop Pilot as requested. Review initialization/boundary compatibility and panel feasibility offline; any behavioral amendment requires a new version, freeze and authorization.
