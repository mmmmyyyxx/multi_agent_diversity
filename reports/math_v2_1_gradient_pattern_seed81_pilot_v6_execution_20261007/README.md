# A4 Seed81 Gradient Pattern Memory Pilot search audit

Status: **VALID_A4_SEED81_PILOT_SEARCH**. Scientific stop: `SATURATION_REACHED`. This is a descriptive training-process run; component efficacy and generalization are **NOT_ESTIMATED**.

Optimize60 and private Shadow40 use existing frozen memberships. Completed opportunities: 10; built: 10; target sequence: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]; commits: 0. Incomplete target: None.

Initial Optimize metrics: `{"MeanMemberAcc": 0.36666666666666664, "MinMemberAcc": 0.36666666666666664, "OracleAcc": 0.36666666666666664, "VoteAcc": 0.36666666666666664, "count": 60, "invalid_count_by_member": [2, 2, 2, 2, 2], "mean_disagreement": 0.9666666666666667, "member_correct_counts": [22, 22, 22, 22, 22], "oracle_correct_count": 22, "vote_correct_count": 22}`.

Final deployed-prefix Optimize metrics: `{"MeanMemberAcc": 0.36666666666666664, "MinMemberAcc": 0.36666666666666664, "OracleAcc": 0.36666666666666664, "VoteAcc": 0.36666666666666664, "count": 60, "invalid_count_by_member": [2, 2, 2, 2, 2], "mean_disagreement": 0.9666666666666667, "member_correct_counts": [22, 22, 22, 22, 22], "oracle_correct_count": 22, "vote_correct_count": 22}`.

Provider calls: 1613; charged tokens: 832198; cumulative: 3072683; remaining: 36927317 of the durable 40M authorization. Roles and detailed stages are in [provider usage](provider_usage.json) and [cost trajectory](cost_trajectory.json).

Validation: `SKIPPED_NO_TEAM_CHANGE` (0 calls). Test: sealed (0 calls). Other arms/seeds, raw diagnostic and LLM judge: 0. No tuning or scientific rerun. Sanitized publication to origin/main is authorized.

Executable source: `d294025acc9151e146efae258ac3b9c90d05baa9`. Preexecution commit: `30b38600e25c0d63cb51f280ca4001bf5c16c88d`. Base: `05e60bdbb73f2a89ed05ec88db9f4e1fe5a7e579`. Full current suite: 1495 passed, 2 skipped. Historical private tests: NOT_RUN; full historical replay PASS is not claimed.

The [analysis index](analysis_index.md) links the full module traces. Exact gradients, procedures/diffs and Memory read sets/states are in the ignored local bundle `runs/gradient_contract_recovery_v1/analysis_bundle`; its SHA manifest hash is `5ee007f401f9c2104bf7477f6f7f4ca33b2559af1a49abb155461b905a60d732`. Public evidence contains hashes, counters, categories and aggregate metrics.

Recovery: {"accepted_gradient_count": 380, "gradient_contract_recovery_rate": 0.007894736842105263, "gradient_first_pass_valid": 377, "gradient_retry_once": 3, "gradient_retry_twice": 0, "gradient_three_fail": 0, "length_rejections": 0, "logical_gradient_count": 380, "numeric_warnings": 19, "physical_gradient_calls": 383, "schema_rejections": 0, "strong_leakage_rejections": 2, "successful_recovery_rate": 1.0}.

Mutation behavior: {"Solver_evaluated_candidates": 57, "behavior_baseline": "actual local parent; exact returned text comparison", "behavior_change_rate": 0.5964912280701754, "behavior_changed_candidates": 34, "correctness_and_class_baseline": "actual local parent; frozen binary evaluator", "correctness_change_rate": 0.2807017543859649, "correctness_changed_candidates": 16, "local_classes": {"NEUTRAL": 41, "PURE_REGRESSION": 12, "PURE_REPAIR": 4}, "parsed_answer_change_rate": 0.5789473684210527, "prompt_changed_behavior_unchanged_candidates": 23, "prompt_changed_but_behavior_unchanged_rate": 0.40350877192982454, "repair_candidates": 15, "repair_target": "selected Pattern support intersect local panel, root wrong", "repair_yield": 0.2631578947368421, "root_relative_fixed_broken_fields_retained": true, "scientific_efficacy": "NOT_CAUSALLY_ESTABLISHED"}.

Descriptive bottleneck: `TEAMPROBE_TO_FULL`. No method adaptation or causal efficacy claim. See [mutation trace](mutation_behavior_trace.jsonl), [repair summary](repair_yield_summary.json), [TeamProbe breakdown](teamprobe_failure_breakdown.json) and [analysis index](analysis_bundle_index.json).

The original runner ended with `EXECUTION_ABORTED` after the scientific stop because terminal Validation metadata repeated a keyword. Its raw lifecycle, missing runtime completion receipt and missing runtime summary are preserved. The [owner completion audit](owner_scientific_completion_receipt.json) independently certifies the full scientific trace, final state, stop and all charges; it adds no provider calls or scientific continuation. A complete zero-commit result is retained.

Post-stop engineering verification: **1497 current tests passed, 2 skipped**; both changed-team and unchanged-team terminal regression cases pass. The duplicate metadata keyword and JSON tuple/list equality defects are repaired in current source; the frozen executable and original run artifacts remain unchanged. See [post-stop repair verification](post_stop_engineering_verification.json). No provider call or scientific rerun was used for this repair.
