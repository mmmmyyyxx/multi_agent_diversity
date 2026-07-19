# Refactor Function Matrix

Generated from `68` Python files.

## TraceBeamSearchSystem methods

| Function | Lines | Current location | Target responsibility | Duplicate-name count |
| --- | ---: | --- | --- | ---: |
| `update_prompt_with_beam` | 1330 | `system.py:7904` | `optimization` | 1 |
| `evaluate_candidate_prompt` | 414 | `system.py:7489` | `optimization` | 1 |
| `propose_candidates_teacher_critic_student` | 348 | `system.py:6264` | `optimization` | 1 |
| `_summarize_rollout_rows` | 336 | `system.py:10243` | `evaluation` | 1 |
| `select_joint_active_team` | 293 | `system.py:9834` | `qd` | 1 |
| `propose_candidates_one_shot` | 258 | `system.py:6643` | `optimization` | 1 |
| `generate_student_candidates` | 229 | `system.py:6034` | `optimization` | 1 |
| `_window_update_diagnosis` | 228 | `system.py:4708` | `orchestration` | 1 |
| `_build_case_generation_batches` | 227 | `system.py:5006` | `orchestration` | 1 |
| `_window_accuracy_diagnosis` | 211 | `system.py:4496` | `metrics` | 1 |
| `_propose_accuracy_candidates` | 201 | `system.py:5319` | `optimization` | 1 |
| `_evaluate_candidate_prompt_accuracy_only` | 187 | `system.py:7301` | `optimization` | 1 |
| `run_one` | 183 | `system.py:7509` | `orchestration` | 4 |
| `_build_teacher_context` | 167 | `system.py:5521` | `optimization` | 1 |
| `_structured_fallback_role` | 161 | `system.py:2971` | `orchestration` | 1 |
| `compute_rollout_metrics` | 155 | `system.py:3798` | `evaluation` | 1 |
| `__init__` | 148 | `system.py:483` | `orchestration` | 3 |
| `maybe_update_prompts` | 148 | `system.py:9380` | `optimization` | 1 |
| `_candidate_trajectory_feasibility` | 146 | `system.py:1199` | `optimization` | 1 |
| `refresh_all_prompt_beams` | 144 | `system.py:9235` | `optimization` | 1 |
| `_infer_target_error_pattern` | 125 | `system.py:4007` | `orchestration` | 1 |
| `_chat` | 104 | `system.py:3133` | `orchestration` | 1 |
| `_candidate_reward_vote_useful_diversity` | 99 | `system.py:7149` | `optimization` | 1 |
| `record_train_rollout` | 97 | `system.py:9656` | `evaluation` | 1 |
| `generate_approved_teacher_question` | 96 | `system.py:5830` | `optimization` | 1 |
| `_select_hybrid_beam` | 90 | `system.py:1636` | `orchestration` | 1 |
| `_select_vote_pareto_beam` | 90 | `system.py:1473` | `metrics` | 1 |
| `evaluate_competence_probe` | 88 | `system.py:10128` | `evaluation` | 1 |
| `_candidate_boundary_error_metrics` | 83 | `system.py:898` | `optimization` | 1 |
| `write_run_meta` | 77 | `system.py:2421` | `persistence` | 1 |
| `_select_boundary_reward_agents` | 76 | `system.py:4419` | `metrics` | 1 |
| `competence_relative_specialization_strength` | 75 | `system.py:272` | `orchestration` | 1 |
| `evaluate_dataset` | 75 | `system.py:10580` | `orchestration` | 1 |
| `_prewarm_factorized_candidate_rollouts` | 74 | `system.py:3426` | `optimization` | 1 |
| `_apply_hybrid_soft_guards` | 71 | `system.py:1564` | `orchestration` | 1 |
| `_weighted_vote_with_diagnostics` | 69 | `system.py:3728` | `metrics` | 1 |
| `_candidate_reward_guarded` | 67 | `system.py:7081` | `optimization` | 1 |
| `select_reward_agents_for_update` | 67 | `system.py:4287` | `metrics` | 1 |
| `run_one` | 66 | `system.py:7310` | `orchestration` | 4 |
| `_record_llm_call` | 64 | `system.py:1793` | `orchestration` | 1 |
| `_apply_no_effective_evolution_tracking` | 63 | `system.py:9529` | `orchestration` | 1 |
| `_select_hybrid_reward_agents` | 63 | `system.py:4355` | `metrics` | 1 |
| `repair_student_json_response` | 61 | `system.py:5972` | `optimization` | 1 |
| `_empty_optimizer_generation_diagnostics` | 60 | `system.py:2615` | `orchestration` | 1 |
| `embedding_overlap_diagnostics` | 60 | `system.py:3607` | `orchestration` | 1 |
| `evaluate_one` | 59 | `system.py:10583` | `orchestration` | 2 |
| `_effective_reward_weights` | 57 | `system.py:6994` | `metrics` | 1 |
| `_load_recorded_solver_rollouts` | 56 | `system.py:2067` | `evaluation` | 1 |
| `propose_teacher_question` | 55 | `system.py:5689` | `optimization` | 1 |
| `_behavior_context_for_baseline` | 54 | `system.py:807` | `orchestration` | 1 |
| `_candidate_residual_metrics` | 54 | `system.py:982` | `optimization` | 1 |
| `_base_log_fields` | 51 | `system.py:2314` | `orchestration` | 1 |
| `_evaluate_prompt_on_stable_probe` | 51 | `system.py:9782` | `optimization` | 1 |
| `complete_competence_epoch` | 51 | `system.py:680` | `orchestration` | 1 |
| `critique_teacher_question` | 51 | `system.py:5745` | `optimization` | 1 |
| `evaluate_joint_trace_diversity` | 49 | `system.py:6902` | `qd` | 1 |
| `_build_homogeneous_cases` | 44 | `system.py:4137` | `orchestration` | 1 |
| `_optimizer_case_payload` | 44 | `system.py:5234` | `orchestration` | 1 |
| `retry_student_candidates_json_only` | 44 | `system.py:5927` | `optimization` | 1 |
| `_teacher_metadata_from_diagnostics` | 43 | `system.py:2790` | `optimization` | 1 |
| `_candidate_reward_competence_depth` | 41 | `system.py:7249` | `optimization` | 1 |
| `_trajectory_event` | 40 | `system.py:1346` | `orchestration` | 1 |
| `get_or_create_solver_rollout` | 40 | `system.py:3294` | `evaluation` | 1 |
| `_flush_capability_profile` | 39 | `system.py:1096` | `persistence` | 1 |
| `solve_once` | 39 | `system.py:3238` | `evaluation` | 2 |
| `_student_candidate_schema_json` | 38 | `system.py:2862` | `optimization` | 1 |
| `_candidate_v7_log_fields` | 37 | `system.py:1037` | `optimization` | 1 |
| `validate_tcs_candidate_metadata` | 37 | `system.py:114` | `optimization` | 1 |
| `_homogeneity_impact_metrics` | 36 | `system.py:5282` | `metrics` | 1 |
| `_candidate_behavior_metrics` | 35 | `system.py:862` | `optimization` | 1 |
| `_record_stable_qd_archive_snapshot` | 35 | `system.py:2167` | `qd` | 1 |
| `solve_with_prompts_reusing_records` | 35 | `system.py:3369` | `optimization` | 1 |
| `compute_crowding_distances` | 34 | `system.py:440` | `orchestration` | 1 |
| `select_agents_for_update` | 34 | `system.py:4221` | `orchestration` | 1 |
| `_is_redundant_candidate_prompt` | 33 | `system.py:2581` | `optimization` | 1 |
| `_record_solver_rollout` | 33 | `system.py:1937` | `evaluation` | 1 |
| `_solve_agent_rollout` | 33 | `system.py:3335` | `evaluation` | 1 |
| `compute_coverage_depth_transitions` | 33 | `system.py:203` | `metrics` | 1 |
| `propose_for_parent` | 32 | `system.py:7965` | `orchestration` | 1 |
| `rewrite_teacher_question` | 32 | `system.py:5797` | `persistence` | 1 |
| `solve_train_example_without_update` | 32 | `system.py:9623` | `evaluation` | 1 |
| `select_error_agents_for_update` | 30 | `system.py:4256` | `orchestration` | 1 |
| `sync_prompt_history_current_state` | 30 | `system.py:2529` | `persistence` | 1 |
| `_mean_metric_dict` | 29 | `system.py:9593` | `metrics` | 1 |
| `_required_optimizer_fields` | 29 | `system.py:2698` | `orchestration` | 1 |
| `evaluate_example` | 29 | `system.py:9785` | `orchestration` | 1 |
| `propose_candidates` | 29 | `system.py:6613` | `optimization` | 1 |
| `_vote_pareto_feasibility` | 28 | `system.py:1387` | `metrics` | 1 |
| `_make_refill_candidate` | 27 | `system.py:2203` | `optimization` | 1 |
| `_select_stable_qd_parents` | 27 | `system.py:2260` | `orchestration` | 1 |
| `_build_validity_cases` | 26 | `system.py:4182` | `orchestration` | 1 |
| `_capability_specialization_diagnostics` | 25 | `system.py:10217` | `orchestration` | 1 |
| `_target_error_cases_for_agent` | 25 | `system.py:4955` | `orchestration` | 1 |
| `compute_oracle_coverage_transitions` | 25 | `system.py:153` | `metrics` | 1 |
| `_existing_run_meta_matches_solver_cache` | 24 | `system.py:2026` | `evaluation` | 1 |
| `_window_random_case_summaries` | 24 | `system.py:4981` | `orchestration` | 1 |
| `_git_provenance` | 23 | `system.py:2387` | `orchestration` | 1 |
| `_prepare_v8_candidate_text_fields` | 23 | `system.py:2947` | `optimization` | 1 |
| `_reward_phase_state` | 23 | `system.py:6970` | `metrics` | 1 |
| `non_dominated_sort` | 23 | `system.py:415` | `orchestration` | 1 |
| `_empty_cost_summary` | 21 | `system.py:1746` | `orchestration` | 1 |
| `_encode_trace_document` | 21 | `system.py:3574` | `orchestration` | 1 |
| `_safe_float` | 21 | `system.py:2754` | `orchestration` | 1 |
| `behavior_fingerprint_similarity` | 21 | `system.py:1150` | `orchestration` | 1 |
| `compute_candidate_metric_deltas` | 21 | `system.py:84` | `optimization` | 1 |
| `compute_vote_transitions` | 21 | `system.py:180` | `metrics` | 1 |
| `_make_beam_item` | 20 | `system.py:2131` | `orchestration` | 1 |
| `_student_failure_log_fields` | 20 | `system.py:2841` | `optimization` | 1 |
| `_vote_pareto_active_sort_key` | 20 | `system.py:1416` | `metrics` | 1 |
| `ensure_recorded_rollouts_for_prompts` | 20 | `system.py:3405` | `optimization` | 1 |
| `save_state` | 20 | `system.py:10656` | `orchestration` | 1 |
| `_answer_behavior_preview` | 19 | `system.py:3969` | `orchestration` | 1 |
| `_record_solver_rollouts` | 19 | `system.py:1985` | `evaluation` | 1 |
| `_vote_pareto_crowding_sort_key` | 19 | `system.py:1437` | `metrics` | 1 |
| `_write_json_snapshot` | 19 | `system.py:10703` | `persistence` | 1 |
| `rule_invalid_check` | 19 | `system.py:3517` | `orchestration` | 1 |
| `_attach_stable_mechanism_representation` | 18 | `system.py:1727` | `qd` | 1 |
| `_default_prompt_bank` | 18 | `system.py:1881` | `optimization` | 1 |
| `_missing_optimizer_fields` | 18 | `system.py:2728` | `orchestration` | 1 |
| `_normalize_llm_call_stage` | 18 | `system.py:1862` | `orchestration` | 1 |
| `_recompute_effective_residual_strength` | 18 | `system.py:732` | `orchestration` | 1 |
| `normalize_mechanism_signature` | 18 | `system.py:244` | `qd` | 1 |
| `_peer_behavior_summary` | 17 | `system.py:3989` | `orchestration` | 1 |
| `_split_trace_for_embedding` | 17 | `system.py:3549` | `orchestration` | 1 |
| `_target_trace_novelty` | 17 | `system.py:3683` | `orchestration` | 1 |
| `_truncate_candidate_text_fields` | 17 | `system.py:2925` | `optimization` | 1 |
| `error_pareto_dominates` | 17 | `system.py:364` | `orchestration` | 1 |
| `rollout_train_example` | 17 | `system.py:9754` | `evaluation` | 1 |
| `_append_prompt_history_event` | 16 | `system.py:2512` | `persistence` | 1 |
| `competence_depth_dominates` | 16 | `system.py:383` | `orchestration` | 1 |
| `_competence_depth_sort_key` | 15 | `system.py:1457` | `orchestration` | 1 |
| `_effective_reward_log_fields` | 15 | `system.py:7052` | `metrics` | 1 |
| `_iter_recorded_rollout_files` | 15 | `system.py:2051` | `evaluation` | 1 |
| `_mark_mechanism_novelty` | 15 | `system.py:2288` | `qd` | 1 |
| `_record_optimizer_generation_diagnostics` | 15 | `system.py:2676` | `orchestration` | 1 |
| `_solver_rollout_cache_key_from_hashes` | 15 | `system.py:1918` | `evaluation` | 1 |
| `_refresh_joint_representatives` | 14 | `system.py:2152` | `qd` | 1 |
| `_student_refusal_or_explanation` | 14 | `system.py:2910` | `optimization` | 1 |
| `_add_solver_rollout_cache_row` | 13 | `system.py:1971` | `evaluation` | 1 |
| `_trace_diversity_for_indices` | 13 | `system.py:3701` | `orchestration` | 1 |
| `_update_vote_context_profile` | 13 | `system.py:1136` | `metrics` | 1 |
| `_v7_residual_protocol_enabled` | 13 | `system.py:778` | `orchestration` | 1 |
| `evaluate_one` | 13 | `system.py:10179` | `orchestration` | 2 |
| `in_dominant_wrong_cluster` | 13 | `system.py:7593` | `orchestration` | 1 |
| `take` | 13 | `system.py:5037` | `orchestration` | 1 |
| `_apply_competence_depth1_candidate_guard` | 12 | `system.py:667` | `optimization` | 1 |
| `_candidate_eval_audit_fields` | 12 | `system.py:7068` | `optimization` | 1 |
| `_contains_task_specific_content` | 12 | `system.py:2560` | `orchestration` | 1 |
| `_expire_agent_probation_branches` | 12 | `system.py:2238` | `orchestration` | 1 |
| `_init_prompt_history` | 12 | `system.py:2499` | `persistence` | 1 |
| `_useful_trace_diversity` | 12 | `system.py:3715` | `orchestration` | 1 |
| `call_factory` | 12 | `system.py:3346` | `orchestration` | 1 |
| `competence_non_dominated_sort` | 12 | `system.py:401` | `orchestration` | 1 |
| `solve_with_agent_prompt_override` | 12 | `system.py:3504` | `optimization` | 1 |
| `solve_with_prompts_limited` | 12 | `system.py:3281` | `optimization` | 1 |
| `_load_embedding_model` | 11 | `system.py:3537` | `orchestration` | 1 |
| `capability_alignment` | 11 | `system.py:1075` | `orchestration` | 1 |
| `evaluate_one_candidate` | 11 | `system.py:8315` | `optimization` | 1 |
| `_build_task_spec` | 10 | `system.py:632` | `orchestration` | 1 |
| `_lookup_solver_rollout` | 10 | `system.py:2005` | `evaluation` | 1 |
| `_vector_cosine_similarity` | 10 | `system.py:3596` | `orchestration` | 1 |
| `prewarm` | 10 | `system.py:3458` | `orchestration` | 1 |
| `_active_prompt_list` | 9 | `system.py:2304` | `optimization` | 1 |
| `_append_solver_rollout_record` | 9 | `system.py:2016` | `evaluation` | 1 |
| `_candidate_reward_coverage_useful_diversity` | 9 | `system.py:7291` | `optimization` | 1 |
| `_redact_optimizer_text` | 9 | `system.py:3959` | `orchestration` | 1 |
| `_split_integrity_metadata` | 9 | `system.py:2411` | `orchestration` | 1 |
| `_stable_probe_cache_key` | 9 | `system.py:9772` | `evaluation` | 1 |
| `behavior_fingerprint_utility` | 9 | `system.py:1172` | `orchestration` | 1 |
| `finalize` | 9 | `system.py:2972` | `orchestration` | 1 |
| `_accumulate_capability_evidence` | 8 | `system.py:1087` | `orchestration` | 1 |
| `_clip01` | 8 | `system.py:6952` | `orchestration` | 1 |
| `_hybrid_candidate_type_rejection_reason` | 8 | `system.py:2901` | `optimization` | 1 |
| `_nonnegative` | 8 | `system.py:6961` | `orchestration` | 1 |
| `_read_previous_execution_session_id` | 8 | `system.py:2366` | `orchestration` | 1 |
| `_usage_value` | 8 | `system.py:1784` | `orchestration` | 1 |
| `paired_utility_improvement` | 8 | `system.py:1182` | `orchestration` | 1 |
| `pareto_dominates` | 8 | `system.py:354` | `orchestration` | 1 |
| `write_cost_summary` | 8 | `system.py:10730` | `persistence` | 1 |
| `_append_bounded_archive` | 7 | `system.py:1191` | `qd` | 1 |
| `_client_role_from_stage` | 7 | `system.py:1768` | `orchestration` | 1 |
| `_estimate_tokens` | 7 | `system.py:1776` | `orchestration` | 1 |
| `_flush_jsonl` | 7 | `system.py:10677` | `persistence` | 1 |
| `_solver_cache_settings` | 7 | `system.py:1910` | `evaluation` | 1 |
| `_vote_with_diagnostics` | 7 | `system.py:3668` | `metrics` | 1 |
| `result` | 7 | `system.py:4024` | `orchestration` | 1 |
| `value` | 7 | `system.py:4332` | `orchestration` | 2 |
| `_build_initial_prompts` | 6 | `system.py:1900` | `optimization` | 1 |
| `_candidate_has_required_optimizer_fields` | 6 | `system.py:2747` | `optimization` | 1 |
| `_effective_support_shrinkage` | 6 | `system.py:756` | `orchestration` | 1 |
| `_experiment_protocol_version` | 6 | `system.py:792` | `orchestration` | 1 |
| `_expire_probation_branches` | 6 | `system.py:2231` | `orchestration` | 1 |
| `_initialize_prompt_beams` | 6 | `system.py:2124` | `optimization` | 1 |
| `_merge_student_diagnostics` | 6 | `system.py:2834` | `optimization` | 1 |
| `_normalize_vector` | 6 | `system.py:3567` | `orchestration` | 1 |
| `mechanism_signature_distance` | 6 | `system.py:264` | `qd` | 1 |
| `_accuracy_cases_for_agent` | 5 | `system.py:4949` | `metrics` | 1 |
| `_cases_for_agent` | 5 | `system.py:4937` | `orchestration` | 1 |
| `_optimizer_generation_diagnostics_for_parent` | 5 | `system.py:2692` | `orchestration` | 1 |
| `_tcs_call_group_id` | 5 | `system.py:2381` | `orchestration` | 1 |
| `_validity_cases_for_agent` | 5 | `system.py:4943` | `orchestration` | 1 |
| `clear_homogeneity_windows` | 5 | `system.py:4215` | `orchestration` | 1 |
| `flush_update_logs` | 5 | `system.py:10685` | `persistence` | 1 |
| `has_guiding_question` | 5 | `system.py:5844` | `orchestration` | 1 |
| `tcs_metadata_applicable` | 5 | `system.py:107` | `orchestration` | 1 |
| `_candidate_generation_source` | 4 | `system.py:2785` | `optimization` | 1 |
| `_candidate_pool_source` | 4 | `system.py:2780` | `optimization` | 1 |
| `_current_joint_change_limit` | 4 | `system.py:2255` | `qd` | 1 |
| `_effective_progressive_weight` | 4 | `system.py:751` | `orchestration` | 1 |
| `_sanitize_prompt` | 4 | `system.py:2573` | `optimization` | 1 |
| `_trace_method_preview` | 4 | `system.py:3954` | `orchestration` | 1 |
| `_uses_vote_pareto_selection` | 4 | `system.py:766` | `metrics` | 1 |
| `competence_specialization_strength` | 4 | `system.py:238` | `orchestration` | 1 |
| `prompt_change_ratio` | 4 | `system.py:802` | `optimization` | 2 |
| `_case_key` | 3 | `system.py:4133` | `orchestration` | 1 |
| `_is_optimizer_generated_candidate_source` | 3 | `system.py:2776` | `optimization` | 1 |
| `_model_role_for_client_role` | 3 | `system.py:1858` | `orchestration` | 1 |
| `_pareto_value` | 3 | `system.py:349` | `orchestration` | 1 |
| `_prompt_ends_with_sentence_boundary` | 3 | `system.py:2943` | `optimization` | 1 |
| `_safe_agent_correct` | 3 | `system.py:3679` | `orchestration` | 1 |
| `_uses_vote_error_pareto_selection` | 3 | `system.py:771` | `metrics` | 1 |
| `expire_probation_branches` | 3 | `system.py:2251` | `orchestration` | 1 |
| `flush_llm_call_logs` | 3 | `system.py:10726` | `persistence` | 1 |
| `flush_test_trace_history_logs` | 3 | `system.py:10699` | `persistence` | 1 |
| `flush_train_step_logs` | 3 | `system.py:10691` | `persistence` | 1 |
| `flush_train_trace_history_logs` | 3 | `system.py:10695` | `persistence` | 1 |
| `rate` | 3 | `system.py:4423` | `orchestration` | 1 |
| `solve_agent` | 3 | `system.py:3287` | `evaluation` | 1 |
| `value` | 3 | `system.py:4368` | `orchestration` | 2 |
| `_current_execution_session_id` | 2 | `system.py:2375` | `orchestration` | 1 |
| `_hash` | 2 | `system.py:1907` | `orchestration` | 2 |
| `_is_accuracy_only_mode` | 2 | `system.py:643` | `metrics` | 1 |
| `_is_competence_depth_reward_mode` | 2 | `system.py:655` | `metrics` | 1 |
| `_is_coverage_useful_diversity_mode` | 2 | `system.py:652` | `metrics` | 1 |
| `_is_guarded_reward_mode` | 2 | `system.py:646` | `metrics` | 1 |
| `_is_stable_qd_lineage` | 2 | `system.py:664` | `qd` | 1 |
| `_is_v82_hybrid` | 2 | `system.py:661` | `orchestration` | 1 |
| `_is_vote_useful_diversity_mode` | 2 | `system.py:649` | `metrics` | 1 |
| `_normalized_prompt_hash` | 2 | `system.py:799` | `optimization` | 1 |
| `_prompt_signature` | 2 | `system.py:2578` | `optimization` | 1 |
| `_residual_specialization_enabled` | 2 | `system.py:775` | `orchestration` | 1 |
| `_rollout_any_correct` | 2 | `system.py:3676` | `evaluation` | 1 |
| `_solver_rollout_cache_key` | 2 | `system.py:1934` | `evaluation` | 1 |
| `_target_case_keys` | 2 | `system.py:5279` | `orchestration` | 1 |
| `_update_attempt_id` | 2 | `system.py:2378` | `orchestration` | 1 |
| `_uses_baseline_candidate_metrics` | 2 | `system.py:763` | `optimization` | 1 |
| `_uses_competence_depth_pareto_selection` | 2 | `system.py:658` | `orchestration` | 1 |
| `flush_prompt_history` | 2 | `system.py:10723` | `persistence` | 1 |
| `is_homogeneity_window_warmup_done` | 2 | `system.py:4209` | `orchestration` | 1 |
| `is_update_window_ready` | 2 | `system.py:4212` | `orchestration` | 1 |
| `solve_with_current_prompts` | 2 | `system.py:3501` | `optimization` | 1 |
| `solve_with_prompts` | 2 | `system.py:3278` | `optimization` | 1 |

## Duplicate function names

| Name | Count |
| --- | ---: |
| `fake_chat` | 16 |
| `main` | 14 |
| `_system` | 11 |
| `fake_approved` | 7 |
| `fake_student` | 7 |
| `fake_critic` | 5 |
| `fake_rewrite` | 5 |
| `fake_solve` | 5 |
| `fake_teacher` | 5 |
| `_float` | 4 |
| `fake_prewarm` | 4 |
| `fake_propose_candidates` | 4 |
| `from_dict` | 4 |
| `read_jsonl` | 4 |
| `run_one` | 4 |
| `write_csv` | 4 |
| `__init__` | 3 |
| `_system_without_init` | 3 |
| `candidate` | 3 |
| `fake_eval` | 3 |
| `read_json` | 3 |
| `write_jsonl` | 3 |
| `_append_common_cli_args` | 2 |
| `_candidate` | 2 |
| `_collect_run_dirs` | 2 |
| `_fingerprint` | 2 |
| `_format_mmlu_question` | 2 |
| `_hash` | 2 |
| `_latest_test_metrics` | 2 |
| `_safe_mean` | 2 |
| `_selected_settings` | 2 |
| `_to_choice_letter` | 2 |
| `analyze` | 2 |
| `analyze_run` | 2 |
| `balanced_sample` | 2 |
| `build_parser` | 2 |
| `canonical_number_str` | 2 |
| `dedupe_rows` | 2 |
| `encode` | 2 |
| `evaluate_one` | 2 |
| `extract_all_numbers` | 2 |
| `extract_pred_answer_mmlu` | 2 |
| `flaky_replace` | 2 |
| `infer_task_type` | 2 |
| `load_rows` | 2 |
| `make_system` | 2 |
| `normalize_spaces` | 2 |
| `parse_gsm8k_gold` | 2 |
| `parse_mmlu_gold` | 2 |
| `prompt_change_ratio` | 2 |
| `run_all` | 2 |
| `run_and_record` | 2 |
| `run_one_staggered` | 2 |
| `solve_once` | 2 |
| `to_dict` | 2 |
| `vals` | 2 |
| `value` | 2 |
| `write_markdown` | 2 |
