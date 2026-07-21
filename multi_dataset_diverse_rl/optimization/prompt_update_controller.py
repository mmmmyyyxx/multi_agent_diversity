"""Typed prompt-update pipeline stages."""
from dataclasses import dataclass
from ..system_shared import *

@dataclass
class PromptUpdateContext:
    agent_id: int
    overlap_diagnosis: Dict[str, Any]
    eval_batch: List[Dict[str, str]]
    step_id: int
    epoch_id: int


def _select_rollout_archive_for_update(system, context):
    incumbent_hash = system._normalized_prompt_hash(context.agent.current_prompt)
    for item in context.evaluated:
        item['is_incumbent'] = str(item.get('prompt_hash', '')) == incumbent_hash
        item['archive_bucket'] = (
            'safe' if item['is_incumbent'] or bool(item.get('metrics', {}).get('rollout_quality_guard_passed', False))
            else 'catastrophic'
        )
    archive_items = [*getattr(context.agent, 'safe_qd_archive', []), *context.evaluated]
    if system._is_state_conditioned_method():
        context.agent.safe_qd_archive = select_state_conditioned_archive(
            archive_items,
            incumbent_hash,
            int(system.cfg.qd_archive_size_per_agent),
            system.cfg,
        )
    else:
        context.agent.safe_qd_archive = select_rollout_archive(
            archive_items,
            incumbent_hash, int(system.cfg.qd_archive_size_per_agent), system.cfg,
            vote_ready=system._is_vote_ready_rollout_method(),
        )
    context.agent.probation_archive = []
    if system._is_state_conditioned_method():
        context.agent.prompt_beam = select_state_conditioned_representatives(
            context.agent.safe_qd_archive,
            incumbent_hash,
            int(system.cfg.state_representative_capacity),
            system.cfg,
        )
    else:
        context.agent.prompt_beam = select_rollout_representatives(
            context.agent.safe_qd_archive, incumbent_hash,
            int(system.cfg.joint_representative_beam_size), system.cfg,
            vote_ready=system._is_vote_ready_rollout_method(),
        )
    context.agent.prompt_beam = context.agent.prompt_beam or [
        system._make_beam_item(context.agent.current_prompt, None, {}, None, 0)
    ]
    context.selected = list(context.agent.prompt_beam)
    context.requirements = {
        'met': len(context.agent.safe_qd_archive) > 1,
        'safe_non_incumbent_count': sum(
            str(item.get('prompt_hash', '')) != incumbent_hash for item in context.agent.safe_qd_archive
        ),
        'missing': [] if len(context.agent.safe_qd_archive) > 1 else ['missing_state_conditioned_candidate' if system._is_state_conditioned_method() else 'missing_rollout_distinct_candidate'],
    }
    context.pareto_summary.update({
        'safe_archive_size': len(context.agent.safe_qd_archive),
        'probation_archive_count': 0,
        'rollout_signature_count': len({
            str(item.get('metrics', {}).get('rollout_profile', {}).get('rollout_signature_hash', ''))
            for item in context.agent.safe_qd_archive
        }),
        **context.requirements,
    })
    system._record_candidate_funnel_outcomes(
        agent_id=context.agent_id, evaluated=context.evaluated,
        safe_archive=context.agent.safe_qd_archive, epoch=context.epoch_id,
    )
    context.agent.optimizer_update_count_by_epoch[str(context.epoch_id)] = int(
        context.agent.optimizer_update_count_by_epoch.get(str(context.epoch_id), 0) or 0
    ) + 1

class CandidateGenerationStage:

    @staticmethod
    async def run(system, context):
        context.agent = system.agents[context.agent_id]
        context.reference_values = list(getattr(system, 'previous_epoch_per_agent_acc', []) or []) or list(context.overlap_diagnosis.get('per_agent_accuracy', []))
        context.target_reference = float(context.reference_values[context.agent_id]) if context.agent_id < len(context.reference_values) else 0.0
        context.ordered_reference = sorted((float(context.value) for context.value in context.reference_values))
        context.team_bottom2_reference = float(np.mean(context.ordered_reference[:min(2, len(context.ordered_reference))])) if context.ordered_reference else 0.0
        context.team_best = max(context.ordered_reference, default=0.0)
        context.competence_log_fields = {'specialization_strength': float(getattr(system, 'specialization_strength', 0.0)), 'competence_floor_low': float(getattr(system.cfg, 'competence_floor_low', 0.55)), 'competence_floor_high': float(getattr(system.cfg, 'competence_floor_high', 0.65)), 'target_agent_reference_accuracy': context.target_reference, 'target_agent_competence_deficit': max(0.0, float(getattr(system.cfg, 'competence_floor_high', 0.65)) - context.target_reference), 'team_bottom2_reference_accuracy': context.team_bottom2_reference, 'team_best_minus_bottom2_gap': context.team_best - context.team_bottom2_reference}
        context.update_attempt_id = system._update_attempt_id(context.epoch_id, context.step_id, context.agent_id)
        context.agent_update_turn = sum((int(context.value or 0) for context.value in context.agent.optimizer_update_count_by_epoch.values())) + 1
        context.beam = getattr(context.agent, 'prompt_beam', []) or [system._make_beam_item(context.agent.current_prompt, None, {}, None, 0)]
        context.parent_sources = ['active'] * len(context.beam)
        if system._is_stable_qd_lineage():
            context.beam, context.parent_sources = system._select_stable_qd_parents(context.agent, context.epoch_id)
        elif system._is_state_conditioned_method():
            if not context.agent.prompt_memory:
                context.incumbent = system._make_beam_item(
                    context.agent.current_prompt, None, {}, None, 0
                )
                context.incumbent['prompt_hash'] = system._normalized_prompt_hash(
                    context.agent.current_prompt
                )
                context.incumbent['prompt_memory_slot'] = 'active'
                context.agent.prompt_memory = [context.incumbent]
            context.beam, context.parent_sources = select_memory_parents(
                context.agent.prompt_memory,
                int(system.cfg.state_memory_parent_count_per_update),
                rotation_offset=max(0, int(context.agent_update_turn) - 1),
            )
            context.parent_selection_diagnostics = {
                'mode': 'deterministic_prompt_memory',
                'selected_slots': list(context.parent_sources),
            }
            for source in context.parent_sources:
                system.state_parent_selection_source_counts[source] = int(
                    system.state_parent_selection_source_counts.get(source, 0) or 0
                ) + 1
                if source == 'safe_diversity_parent':
                    system.state_search_diagnostics['safe_diversity_parent_use_count'] = int(
                        system.state_search_diagnostics.get('safe_diversity_parent_use_count', 0) or 0
                    ) + 1
            selected_parent_hashes = {
                str(item.get('prompt_hash', '')) for item in context.beam
            }
            for memory_item in context.agent.prompt_memory:
                if str(memory_item.get('prompt_hash', '')) in selected_parent_hashes:
                    memory_item['parent_use_count'] = int(memory_item.get('parent_use_count', 0) or 0) + 1
        elif system._is_rollout_qd_method():
            context.parent_sources = [
                "active" if system._normalized_prompt_hash(str(item.get('prompt', ''))) == system._normalized_prompt_hash(context.agent.current_prompt)
                else "rollout_representative"
                for item in context.beam
            ]
        context.generation = max([int(context.x.get('generation', 0) or 0) for context.x in context.beam] + [0]) + 1
        context.candidate_pool: List[Dict[str, Any]] = []
        context.seen = set()
        context.generation_batches = system._build_case_generation_batches(context.agent_id, context.overlap_diagnosis)
        if not context.generation_batches:
            context.generation_batches = [{'batch_type': 'window_update_diagnosis', 'cases': [], 'purpose': 'general reward-relevant window repair'}]
        context.requested = max(1, int(system.cfg.num_candidates_per_parent))
        context.optimizer_generation_records: List[Dict[str, Any]] = []
        context.parent_jobs = []
        for context.parent_idx, context.parent in enumerate(context.beam):
            context.parent_prompt = str(context.parent.get('prompt', context.agent.current_prompt))
            context.parent_id = str(context.parent.get('id', system._hash(context.parent_prompt)))
            if system._is_v82_hybrid():
                context.parent_batches = list(context.generation_batches)
            elif system._is_state_conditioned_method():
                route_offset = (context.parent_idx + context.agent_update_turn - 1) % len(context.generation_batches)
                context.parent_batches = [
                    context.generation_batches[(route_offset + context.i) % len(context.generation_batches)]
                    for context.i in range(context.requested)
                ]
            else:
                context.parent_batches = [
                    context.generation_batches[context.i % len(context.generation_batches)]
                    for context.i in range(context.requested)
                ]
            context.parent_jobs.append({
                'parent_idx': context.parent_idx,
                'parent': context.parent,
                'parent_prompt': context.parent_prompt,
                'parent_id': context.parent_id,
                'parent_batches': context.parent_batches,
                'parent_source': context.parent_sources[context.parent_idx]
                if context.parent_idx < len(context.parent_sources) else 'active',
                'parent_archive_slot': str(context.parent.get('state_archive_slot', 'active')),
                'parent_was_exploration': str(
                    context.parent.get('state_archive_slot', '')
                ) == 'rollout_exploration',
            })
        context.configured_parent_concurrency = int(getattr(system.cfg, 'optimizer_parent_concurrency', 1) or 1)
        context.parent_concurrency = max(1, min(context.configured_parent_concurrency, len(context.parent_jobs) or 1))
        context.parent_sem = asyncio.Semaphore(context.parent_concurrency)

        async def propose_for_parent(job: Dict[str, Any]) -> Dict[str, Any]:
            async with context.parent_sem:
                feedback = job.get('refill_feedback')
                generation_round = int(feedback.get('refill_round', 0) or 0) if isinstance(feedback, dict) else 0
                context_token = TCS_AUDIT_CONTEXT.set({'optimizer_architecture': str(getattr(system.cfg, 'optimizer_architecture', '') or ''), 'epoch': int(context.epoch_id), 'step': int(context.step_id), 'agent_id': int(context.agent_id), 'parent_id': str(job['parent_id']), 'execution_session_id': system._current_execution_session_id(), 'update_attempt_id': context.update_attempt_id, 'tcs_call_group_id': system._tcs_call_group_id(context.update_attempt_id, str(job['parent_id']), str(job['parent_prompt']), generation_round), 'teacher_critic_round': 0})
                try:
                    proposals = await system.propose_candidates(agent_id=context.agent_id, parent_prompt=str(job['parent_prompt']), overlap_diagnosis=context.overlap_diagnosis, num_candidates=context.requested, generation_batches=job['parent_batches'], refill_feedback=feedback if isinstance(feedback, dict) else None)
                finally:
                    TCS_AUDIT_CONTEXT.reset(context_token)
                return {**job, 'proposals': proposals}
        context.propose_for_parent = propose_for_parent
        context.parent_results = await asyncio.gather(*[context.propose_for_parent(context.job) for context.job in context.parent_jobs])
        context.parent_results.sort(key=lambda x: int(context.x.get('parent_idx', 0)))
        for context.result in context.parent_results:
            context.parent_prompt = str(context.result.get('parent_prompt', context.agent.current_prompt))
            context.parent_id = str(context.result.get('parent_id', system._hash(context.parent_prompt)))
            context.parent_batches = context.result.get('parent_batches', [])
            if not isinstance(context.parent_batches, list) or not context.parent_batches:
                context.parent_batches = [context.generation_batches[0]]
            context.proposals = context.result.get('proposals', [])
            if not isinstance(context.proposals, list):
                context.proposals = []
            context.parent_diagnostics = system._empty_optimizer_generation_diagnostics()
            if context.proposals:
                context.proposal_diag = context.proposals[0].get('optimizer_generation_diagnostics', {}) if isinstance(context.proposals[0], dict) else {}
                if isinstance(context.proposal_diag, dict):
                    context.parent_diagnostics.update(context.proposal_diag)
            else:
                context.parent_diagnostics.update(system._optimizer_generation_diagnostics_for_parent(context.agent_id, context.parent_prompt))
            context.optimizer_generation_records.append(context.parent_diagnostics)
            for context.idx, context.proposal in enumerate(context.proposals):
                context.prompt = str(context.proposal.get('candidate_prompt', '')).strip()
                context.prompt, context._ = system._sanitize_prompt(context.prompt, context.agent_id)
                context.key = normalize_spaces(context.prompt).lower()
                context.preserve_duplicate_objects = str(getattr(system.cfg, 'candidate_eval_execution_mode', 'legacy')) == 'factorized_cached'
                if not context.prompt or (context.key in context.seen and (not context.preserve_duplicate_objects)):
                    continue
                context.seen.add(context.key)
                context.batch = context.parent_batches[context.idx % len(context.parent_batches)]
                context.candidate_pool.append({'candidate_id': f'g{context.generation}_a{context.agent_id}_p{system._hash(context.parent_id)}_{context.idx}_{system._hash(context.prompt)}', 'prompt': context.prompt, 'parent_id': context.parent_id, 'parent_source': str(context.result.get('parent_source', 'active')), 'parent_prompt': context.parent_prompt, 'parent_prompt_hash': system._normalized_prompt_hash(context.parent_prompt), 'parent_archive_slot': str(context.result.get('parent_archive_slot', 'active')), 'parent_was_exploration': bool(context.result.get('parent_was_exploration', False)), 'generation': context.generation, 'source': 'optimizer', 'candidate_pool_source': 'optimizer', 'candidate_source': str(context.proposal.get('candidate_source', 'optimizer') or 'optimizer'), 'generation_batch_type': str(context.proposal.get('generation_batch_type', '')) or str(context.batch.get('batch_type', '')), 'optimization_route': str(context.proposal.get('optimization_route', context.batch.get('optimization_route', 'general_accuracy')) or 'general_accuracy'), 'generation_case_ids': context.proposal.get('generation_case_ids', []), 'target_error_pattern': str(context.proposal.get('target_error_pattern', '')), 'accuracy_repair_rule': str(context.proposal.get('accuracy_repair_rule', '')), 'expected_accuracy_effect': str(context.proposal.get('expected_accuracy_effect', '')), 'diversity_contribution': str(context.proposal.get('diversity_contribution', '')), 'error_correlation_reduction': str(context.proposal.get('error_correlation_reduction', '')), 'task_alignment_rule': str(context.proposal.get('task_alignment_rule', '')), 'peer_redundancy_avoidance': str(context.proposal.get('peer_redundancy_avoidance', '')), 'candidate_prompt_char_count': int(context.proposal.get('candidate_prompt_char_count', len(context.prompt)) or len(context.prompt)), 'candidate_prompt_over_soft_limit': bool(context.proposal.get('candidate_prompt_over_soft_limit', False)), 'candidate_prompt_over_hard_limit': bool(context.proposal.get('candidate_prompt_over_hard_limit', False)), 'candidate_prompt_overlength_rejected': bool(context.proposal.get('candidate_prompt_overlength_rejected', False)), 'candidate_prompt_ends_with_sentence_boundary': bool(context.proposal.get('candidate_prompt_ends_with_sentence_boundary', system._prompt_ends_with_sentence_boundary(context.prompt))), 'optimizer_generation_diagnostics': context.proposal.get('optimizer_generation_diagnostics', {}), 'tcs_call_group_id': str(context.proposal.get('tcs_call_group_id', '') or ''), 'execution_session_id': str(context.proposal.get('execution_session_id', system._current_execution_session_id()) or system._current_execution_session_id()), 'update_attempt_id': str(context.proposal.get('update_attempt_id', context.update_attempt_id) or context.update_attempt_id), 'proposal': context.proposal, 'prompt_hash': system._normalized_prompt_hash(context.prompt)})
                context.candidate_pool[-1]['generation_question_hashes'] = sorted({
                    str(case.get('question_hash', case.get('sample_hash', case.get('case_id', ''))))
                    for case in context.batch.get('cases', [])
                    if isinstance(case, dict)
                    and str(case.get('question_hash', case.get('sample_hash', case.get('case_id', ''))))
                })
                context.candidate_metadata = {'optimizer_architecture': str(context.proposal.get('optimizer_architecture', getattr(system.cfg, 'optimizer_architecture', ''))), 'candidate_source': str(context.proposal.get('candidate_source', '')), 'candidate_pool_source': 'optimizer', 'tcs_call_group_id': str(context.proposal.get('tcs_call_group_id', '') or ''), 'execution_session_id': str(context.proposal.get('execution_session_id', system._current_execution_session_id()) or system._current_execution_session_id()), 'update_attempt_id': str(context.proposal.get('update_attempt_id', context.update_attempt_id) or context.update_attempt_id), **dict(context.proposal.get('optimizer_generation_diagnostics', {}) or {})}
                context.metadata_errors = validate_tcs_candidate_metadata(context.candidate_metadata)
                if context.metadata_errors:
                    context.candidate_id = str(context.candidate_pool[-1].get('candidate_id', ''))
                    raise RuntimeError(f"Invalid Teacher-Critic-Student candidate metadata: agent_id={context.agent_id} epoch={context.epoch_id} step={context.step_id} parent_id={context.parent_id} candidate_id={context.candidate_id} tcs_call_group_id={context.candidate_metadata.get('tcs_call_group_id', '')} metadata_errors={','.join(context.metadata_errors)}")
                system._record_candidate_funnel_item(
                    context.candidate_pool[-1], context.agent_id, "schema_valid_candidate_count"
                )
        for context.parent in context.beam:
            context.prompt = str(context.parent.get('prompt', context.agent.current_prompt))
            context.key = normalize_spaces(context.prompt).lower()
            if context.key in context.seen:
                continue
            context.seen.add(context.key)
            context.candidate_pool.append({'candidate_id': str(context.parent.get('id', '')) or f'beam_{system._hash(context.prompt)}', 'prompt': context.prompt, 'parent_id': context.parent.get('parent_id'), 'generation': int(context.parent.get('generation', 0) or 0), 'source': 'existing_beam', 'candidate_pool_source': 'existing_beam', 'candidate_source': 'existing_beam', 'execution_session_id': system._current_execution_session_id(), 'update_attempt_id': context.update_attempt_id, 'generation_batch_type': '', 'generation_case_ids': [], 'target_error_pattern': '', 'accuracy_repair_rule': '', 'expected_accuracy_effect': '', 'diversity_contribution': '', 'error_correlation_reduction': '', 'task_alignment_rule': '', 'peer_redundancy_avoidance': '', 'optimizer_generation_diagnostics': system._empty_optimizer_generation_diagnostics(), 'proposal': {}, 'prompt_hash': system._normalized_prompt_hash(context.prompt)})

class CheapPrescreenStage:

    @staticmethod
    async def run(system, context):
        context.current_key = normalize_spaces(str(context.agent.current_prompt)).lower()
        if context.current_key not in context.seen:
            context.current_prompt = str(context.agent.current_prompt)
            context.candidate_pool.append({'candidate_id': f'active_{system._hash(context.current_prompt)}', 'prompt': context.current_prompt, 'parent_id': None, 'generation': context.generation, 'source': 'current_active_fallback', 'candidate_pool_source': 'current_active_fallback', 'candidate_source': 'current_active_fallback', 'execution_session_id': system._current_execution_session_id(), 'update_attempt_id': context.update_attempt_id, 'generation_batch_type': '', 'generation_case_ids': [], 'target_error_pattern': '', 'accuracy_repair_rule': '', 'expected_accuracy_effect': '', 'diversity_contribution': '', 'error_correlation_reduction': '', 'task_alignment_rule': '', 'peer_redundancy_avoidance': '', 'optimizer_generation_diagnostics': system._empty_optimizer_generation_diagnostics(), 'proposal': {}, 'prompt_hash': system._normalized_prompt_hash(context.current_prompt)})
        context.initial_prescreen_failures = []
        if system._is_stable_qd_lineage():
            context.accepted_pool, context.prescreen_seen = ([], set())
            for context.candidate in context.candidate_pool:
                if system._candidate_pool_source(context.candidate) != 'optimizer':
                    context.accepted_pool.append(context.candidate)
                    continue
                context.reasons = cheap_prescreen(context.candidate, system._normalized_prompt_hash(str(context.candidate.get('parent_prompt', context.agent.current_prompt))), context.prescreen_seen, parent=next((context.parent for context.parent in context.beam if str(context.parent.get('id', '')) == str(context.candidate.get('parent_id', ''))), None))
                if context.reasons:
                    context.candidate['cheap_prescreen_reasons'] = context.reasons
                    context.initial_prescreen_failures.append({'candidate_type': str(context.candidate.get('proposal', {}).get('candidate_type', '')), 'failure_stage': 'cheap_prescreen', 'reasons': context.reasons})
                    continue
                context.prescreen_seen.update({str(context.candidate.get('prompt_hash', '')), normalize_spaces(str(context.candidate.get('prompt', ''))).lower()})
                context.accepted_pool.append(context.candidate)
            context.candidate_pool = context.accepted_pool
        for context.candidate in context.candidate_pool:
            system._record_candidate_funnel_item(
                context.candidate, context.agent_id, "prescreen_pass_count"
            )
        context.target_case_ids = {str(context.c.get('case_id', '')) for context.b in context.generation_batches if str(context.b.get('batch_type', '')) == 'target_error_repair' for context.c in context.b.get('cases', []) if isinstance(context.c, dict) and str(context.c.get('case_id', ''))}
        context.num_target_error_cases = len(context.target_case_ids)
        context.num_accuracy_repair_candidates = sum((1 for context.c in context.candidate_pool if str(context.c.get('generation_batch_type', '')) == 'target_error_repair' or bool(str(context.c.get('target_error_pattern', '')).strip()) or 'accuracy_repair' in str(context.c.get('candidate_source', ''))))
        context.num_diversity_candidates = sum((1 for context.c in context.candidate_pool if str(context.c.get('generation_batch_type', '')) in {'useful_diversity_repair', 'random_window', 'window_update_diagnosis'} and (not bool(str(context.c.get('target_error_pattern', '')).strip()))))
        context.requested_optimizer_candidates = len(context.beam) * context.requested
        context.num_optimizer_candidates = sum((1 for context.c in context.candidate_pool if system._is_optimizer_generated_candidate_source(system._candidate_generation_source(context.c))))
        context.num_fallback_candidates = sum((1 for context.c in context.candidate_pool if 'fallback' in system._candidate_generation_source(context.c)))
        context.num_existing_beam_candidates = sum((1 for context.c in context.candidate_pool if system._candidate_pool_source(context.c) == 'existing_beam'))
        context.num_tcs_optimizer_candidates = sum((1 for context.c in context.candidate_pool if system._candidate_generation_source(context.c) == 'teacher_critic_student' and system._candidate_pool_source(context.c) == 'optimizer'))
        context.num_tcs_metadata_invalid_candidates = 0
        context.num_tcs_metadata_valid_candidates = context.num_tcs_optimizer_candidates
        context.tcs_execution_complete = bool(context.num_tcs_optimizer_candidates) and context.num_tcs_optimizer_candidates == context.num_tcs_metadata_valid_candidates and (context.num_tcs_metadata_invalid_candidates == 0)
        context.fallback_enabled = str(getattr(system.cfg, 'optimizer_fallback_mode', 'none') or 'none').lower() == 'template'
        context.optimizer_underfilled = context.num_optimizer_candidates < context.requested_optimizer_candidates
        context.optimizer_generation_summary = system._empty_optimizer_generation_diagnostics()
        for context.record in context.optimizer_generation_records:
            if not isinstance(context.record, dict):
                continue
            for context.key in ['optimizer_raw_response_empty', 'optimizer_json_parse_failed', 'optimizer_raw_candidate_count', 'optimizer_empty_prompt_count', 'optimizer_sanitized_count', 'optimizer_redundant_filtered_count', 'optimizer_schema_filtered_count', 'optimizer_final_candidate_count', 'teacher_critic_rounds', 'teacher_rewrite_count', 'student_candidate_count_raw', 'student_candidate_count_final', 'student_candidate_filtered_count', 'student_missing_required_field_count', 'num_teacher_calls', 'num_critic_calls', 'num_teacher_rewrite_calls', 'num_student_calls', 'num_student_retry_calls', 'num_student_repair_calls']:
                context.optimizer_generation_summary[context.key] += int(context.record.get(context.key, 0) or 0)
            for context.key in ['student_raw_response_empty', 'student_json_parse_failed', 'student_json_retry_attempted', 'student_json_retry_succeeded', 'student_json_repair_attempted', 'student_json_repair_succeeded', 'student_json_has_candidates_key', 'student_candidates_is_list', 'student_candidates_empty_list', 'student_refusal_or_explanation']:
                context.optimizer_generation_summary[context.key] = bool(context.optimizer_generation_summary.get(context.key, False) or context.record.get(context.key, False))
        context.optimizer_generation_summary['optimizer_underfilled'] = bool(context.optimizer_underfilled)
        for context.key in ['optimizer_architecture', 'teacher_question', 'teacher_question_approved', 'teacher_question_rejected', 'teacher_question_rejection_reason', 'teacher_question_forced_best_score', 'teacher_question_forced_best_round', 'teacher_question_forced_below_threshold', 'teacher_question_score', 'teacher_quality_critique', 'teacher_specificity_critique', 'teacher_task_alignment_critique', 'teacher_error_alignment_critique', 'teacher_diversity_critique', 'student_candidate_filter_reasons', 'student_all_candidates_filtered', 'student_missing_required_fields', 'student_raw_response_preview', 'student_json_parse_error', 'student_json_retry_raw_response_preview', 'student_json_repair_raw_response_preview', 'student_json_repair_failure_reason', 'student_failure_stage']:
            context.values = [context.record.get(context.key) for context.record in context.optimizer_generation_records if isinstance(context.record, dict) and context.record.get(context.key) not in (None, '', [])]
            if context.values:
                context.optimizer_generation_summary[context.key] = context.values[-1]

class CandidateEvaluationStage:

    @staticmethod
    async def run(system, context):
        context.evaluated = []
        context.peer_prompts = system._active_prompt_list()
        if str(getattr(system.cfg, 'candidate_eval_execution_mode', 'legacy')) == 'factorized_cached':
            context.candidate_eval_cache_stats = await system._prewarm_factorized_candidate_rollouts(agent_id=context.agent_id, eval_batch=context.eval_batch, peer_prompts=context.peer_prompts, candidate_pool=context.candidate_pool)
        else:
            context.prewarm = await system.ensure_recorded_rollouts_for_prompts(eval_batch=context.eval_batch, prompts=context.peer_prompts, source=f'candidate_peer_prewarm_agent_{context.agent_id}')
            context.candidate_eval_cache_stats = {'candidate_eval_execution_mode': 'legacy', 'candidate_eval_candidate_object_count': len(context.candidate_pool), 'candidate_eval_unique_target_prompt_count': len({system._hash(normalize_spaces(str(context.c.get('prompt', '')))) for context.c in context.candidate_pool}), 'candidate_eval_duplicate_target_prompt_count': 0, 'candidate_eval_example_count': len(context.eval_batch), 'candidate_eval_repeat_count': 1, 'candidate_eval_naive_rollout_request_count': len(context.candidate_pool) * len(system.agents) * len(context.eval_batch), 'candidate_eval_factorized_rollout_request_count': 0, 'candidate_eval_unique_rollout_key_count': 0, 'candidate_eval_memory_cache_hit_count': int(context.prewarm.get('solver_reuse_hits', 0) or 0), 'candidate_eval_persisted_cache_hit_count': 0, 'candidate_eval_inflight_reuse_count': 0, 'candidate_eval_solver_api_call_count': int(context.prewarm.get('solver_calls', 0) or 0), 'candidate_eval_rollout_failure_count': 0, 'candidate_eval_calls_saved_vs_naive': 0, 'candidate_eval_cache_hit_rate': float(context.prewarm.get('solver_reuse_hit_rate', 0.0) or 0.0), 'candidate_eval_peer_rollout_key_count': len(system.agents) * len(context.eval_batch), 'candidate_eval_target_rollout_key_count': 0, 'candidate_eval_prompt_dedup_savings': 0}
        context.baseline_cases = system._cases_for_agent(context.overlap_diagnosis, context.agent_id)
        context.configured_concurrency = int(getattr(system.cfg, 'candidate_eval_concurrency', 0) or 0)
        context.eval_concurrency = len(context.candidate_pool) if context.configured_concurrency <= 0 else min(context.configured_concurrency, len(context.candidate_pool))
        context.sem = asyncio.Semaphore(max(1, context.eval_concurrency))

        async def evaluate_one_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
            async with context.sem:
                metrics = await system.evaluate_candidate_prompt(agent_id=context.agent_id, candidate_prompt=str(candidate['prompt']), peer_prompts=context.peer_prompts, eval_batch=context.eval_batch, role_spec=candidate.get('proposal', {}), baseline_homogeneous_cases=context.baseline_cases)
                return {**candidate, 'metrics': metrics, 'reward': float(metrics.get('reward', 0.0))}
        context.evaluate_one_candidate = evaluate_one_candidate
        context.raw_evaluated = await asyncio.gather(*[context.evaluate_one_candidate(context.c) for context.c in context.candidate_pool], return_exceptions=True)
        for context.idx, context.item in enumerate(context.raw_evaluated):
            if isinstance(context.item, dict):
                context.evaluated.append(context.item)
                continue
            context.candidate = context.candidate_pool[context.idx]
            context.metrics = await system.evaluate_candidate_prompt(agent_id=context.agent_id, candidate_prompt=str(context.candidate['prompt']), peer_prompts=context.peer_prompts, eval_batch=context.eval_batch, role_spec=context.candidate.get('proposal', {}), baseline_homogeneous_cases=context.baseline_cases)
            context.evaluated.append({**context.candidate, 'metrics': context.metrics, 'reward': float(context.metrics.get('reward', 0.0))})
        if system._is_state_conditioned_method():
            audit = system._candidate_eval_audit_fields(context.eval_batch)
            for field in (
                'representative_pool_count', 'representative_fallback_count',
                'coverage_pool_requested', 'coverage_pool_actual',
                'conversion_pool_requested', 'conversion_pool_actual',
                'option_count_known_count', 'option_count_unknown_count',
            ):
                system.state_search_diagnostics[field] = int(
                    system.state_search_diagnostics.get(field, 0) or 0
                ) + int(audit.get(field, 0) or 0)
            histograms = system.state_search_diagnostics.setdefault('pool_state_histograms', {})
            for state, count in dict(audit.get('candidate_batch_state_counts', {})).items():
                histograms[str(state)] = int(histograms.get(str(state), 0) or 0) + int(count or 0)
            system.state_search_diagnostics['candidate_pool_update_count'] = int(
                system.state_search_diagnostics.get('candidate_pool_update_count', 0) or 0
            ) + 1
            count_fields = (
                'c0_to_c1_count', 'c1_to_c2_count', 'c2_to_c3_count',
                'c3plus_additional_correct_count', 'c1_to_c0_count',
                'c2_to_c1_count', 'c3_to_c2_count', 'target_wrong_to_correct_count',
                'target_correct_to_wrong_count', 'vote_gain_count', 'vote_loss_count',
                'c2_wrong_split_vote_gain_count', 'c2_wrong_split_vote_loss_count',
                'c2_wrong_split_tie_gain_count', 'c2_wrong_split_strict_gain_count',
                'c2_dominant_wrong_break_count', 'c2_dominant_wrong_create_count',
                'c2_wrong_cluster_reduction', 'c2_wrong_cluster_creation',
            )
            system.state_search_diagnostics['evaluated_candidate_count'] = int(
                system.state_search_diagnostics.get('evaluated_candidate_count', 0) or 0
            ) + len(context.evaluated)
            for item in context.evaluated:
                metrics = item.get('metrics', {})
                for field in count_fields:
                    system.state_search_diagnostics[field] = int(
                        system.state_search_diagnostics.get(field, 0) or 0
                    ) + int(metrics.get(field, 0) or 0)

class CandidateClassificationAndRefillStage:
    @staticmethod
    async def run(system, context):
        context.old_hash = system._hash(context.agent.current_prompt)
        context.trajectory_guard_enabled = system._v7_residual_protocol_enabled(); context.candidate_guard_enabled = bool(getattr(system.cfg, 'competence_depth1_candidate_guard_enabled', False))
        context.pareto_summary = {'num_pareto_feasible': None, 'num_pareto_infeasible': None, 'num_pareto_fronts': None, 'pareto_front0_size': None, 'pareto_forced_current_fallback': None}
        for context.item in context.evaluated:
            context.metrics = context.item.get('metrics', {}) if isinstance(context.item.get('metrics', {}), dict) else {}
            if system._is_v82_hybrid():
                context.proposal = context.item.get('proposal', {}) if isinstance(context.item.get('proposal', {}), dict) else {}
                context.signature = list(context.proposal.get('mechanism_signature', [])) or normalize_mechanism_signature(context.proposal.get('mechanism_steps', []))
                context.parent_item = next((context.row for context.row in context.beam if str(context.row.get('id', '')) == str(context.item.get('parent_id', ''))), None)
                context.parent_metrics = context.parent_item.get('metrics', {}) if isinstance(context.parent_item, dict) else {}
                context.parent_signature = list(context.parent_metrics.get('mechanism_signature', []))
                context.distance = mechanism_signature_distance(context.signature, context.parent_signature)
                context.metrics.update({'candidate_type': str(context.proposal.get('candidate_type', '')), 'mechanism_signature': context.signature, 'parent_mechanism_signature': context.parent_signature, 'peer_dominant_mechanism_signature': [], 'mechanism_signature_distance': context.distance, 'mechanism_novelty_bonus': 0.0 if system._is_stable_qd_lineage() else float(getattr(system.cfg, 'mechanism_novelty_bonus_weight', 0.2)) * context.distance})
                if context.signature:
                    system.mechanism_signature_by_prompt_hash[system._normalized_prompt_hash(str(context.item.get('prompt', '')))] = list(context.signature)
                if system._is_stable_qd_lineage():
                    system._attach_stable_mechanism_representation(context.item)
            if context.trajectory_guard_enabled:
                context.metrics.update(system._candidate_trajectory_feasibility(context.agent, context.item))
            if system._is_v82_hybrid():
                context.metrics = system._apply_hybrid_soft_guards(context.metrics)
            context.depth1_guard_passed = True if system._is_stable_qd_lineage() else system._apply_competence_depth1_candidate_guard(context.metrics)
            if system._is_v82_hybrid():
                context._, context._, context.hard_feasible = system._vote_pareto_feasibility(context.metrics)
                context.metrics['hard_guard_passed'] = bool(context.depth1_guard_passed and context.hard_feasible and (not context.metrics.get('rejection_reason')))
                context.item['reward'] = float(context.metrics.get('penalized_reward', context.item.get('reward', 0.0)) or 0.0)
            context.item['metrics'] = context.metrics
            context.item['trajectory_feasible'] = bool(context.depth1_guard_passed) and (not bool(context.metrics.get('rejection_reason', '')))
            if not context.item['trajectory_feasible']:
                context.item['pareto_feasible'] = False
                context.item['pareto_rank'] = None
                context.item['pareto_crowding_distance'] = None
                context.item['pareto_selected'] = False
                context.item['pareto_forced_fallback'] = False
        if system._is_stable_qd_lineage():
            context.existing_niches = list(getattr(context.agent, 'safe_qd_archive', [])) + list(getattr(context.agent, 'probation_archive', []))
            for context.item in context.evaluated:
                context.item['is_incumbent'] = str(context.item.get('prompt_hash', '')) == system._normalized_prompt_hash(context.agent.current_prompt)
                context.parent = next((context.row for context.row in context.beam if str(context.row.get('id', '')) == str(context.item.get('parent_id', ''))), None)
                system._mark_mechanism_novelty(context.item, parent=context.parent, existing=context.existing_niches)
                context.item['archive_bucket'] = 'safe' if context.item['is_incumbent'] else candidate_quality_bucket(context.item, system.cfg)
                context.existing_niches.append(context.item)
            context.safe_archive = select_safe_archive([*getattr(context.agent, 'safe_qd_archive', []), *context.evaluated], system._normalized_prompt_hash(context.agent.current_prompt), int(system.cfg.qd_archive_size_per_agent))
            for context.item in context.safe_archive:
                context.item['archive_bucket'] = 'safe'
            context.probation = [context.item for context.item in context.evaluated if context.item.get('archive_bucket') == 'probation']
            for context.item in context.probation:
                context.item['probation_created_update'] = int(context.agent_update_turn)
            context.prior_probation = list(getattr(context.agent, 'probation_archive', []))
            context.agent.probation_archive = (context.probation + context.prior_probation)[:int(system.cfg.probation_archive_size_per_agent)]
            context.agent.safe_qd_archive = context.safe_archive
            system._refresh_joint_representatives(context.agent)
            context.requirements = search_space_requirements(context.evaluated, context.agent.safe_qd_archive, context.agent.prompt_beam, system._normalized_prompt_hash(context.agent.current_prompt), system.cfg)
            context.selected = list(context.agent.prompt_beam)
            context.pareto_summary.update({'safe_archive_size': len(context.agent.safe_qd_archive), 'probation_archive_count': len(context.agent.probation_archive), **context.requirements})
            system.per_agent_optimizer_update_count[f'{context.epoch_id}:{context.agent_id}'] = int(system.per_agent_optimizer_update_count.get(f'{context.epoch_id}:{context.agent_id}', 0)) + 1
            context.refill_round_count = 0
            context.refill_requested_candidate_count = 0
            context.refill_actual_candidate_count = 0
            context.refill_trigger_reasons = list(context.requirements.get('missing', []))
            context.refill_stop_reason = 'requirements_met' if context.requirements.get('met') else 'max_rounds_reached'
            context.refill_solver_calls = 0
            context.refill_solver_call_limit_reached = False
            context.prior_probation_ids = {str(context.item.get('id', system._hash(str(context.item.get('prompt', ''))))) for context.item in getattr(context.agent, 'probation_archive', [])}
            context.prior_failures = context.initial_prescreen_failures + [{'candidate_type': str(context.item.get('metrics', {}).get('candidate_type', '')), 'failure_stage': 'candidate_evaluation', 'reasons': [str(context.item.get('metrics', {}).get('rejection_reason', context.item.get('archive_bucket', '')))], 'accuracy_delta': float(context.item.get('metrics', {}).get('accuracy_delta', 0.0) or 0.0), 'depth1_gain_count': int(context.item.get('metrics', {}).get('depth1_gain_count', 0) or 0), 'depth1_loss_count': int(context.item.get('metrics', {}).get('depth1_loss_count', 0) or 0), 'depth2_gain_count': int(context.item.get('metrics', {}).get('depth2_gain_count', 0) or 0), 'depth2_loss_count': int(context.item.get('metrics', {}).get('depth2_loss_count', 0) or 0)} for context.item in context.evaluated if context.item.get('archive_bucket') != 'safe']
            context.prior_failures.extend(({'candidate_type': str(context.item.get('metrics', {}).get('candidate_type', '')), 'failure_stage': 'archive_assignment', 'reasons': ['near_duplicate_existing_niche'], 'nearest_niche': repr(mechanism_niche_key(context.item.get('metrics', {}).get('mechanism_representation', {})))} for context.item in context.evaluated if str(context.item.get('metrics', {}).get('candidate_type', '')) == 'mechanism_alternative' and (not bool(context.item.get('metrics', {}).get('mechanism_novel', False)))))
            while bool(system.cfg.candidate_refill_enabled) and (not context.requirements.get('met')) and (context.refill_round_count < int(system.cfg.candidate_refill_max_rounds)):
                context.active_parent = context.beam[0]
                context.active_parent_id = str(context.active_parent.get('id', system._hash(context.agent.current_prompt)))
                context.parent_unique_count = sum((1 for context.item in context.evaluated if context.item.get('candidate_pool_source') == 'optimizer' and str(context.item.get('parent_id', '')) == context.active_parent_id))
                context.remaining_unique_slots = int(system.cfg.candidate_refill_max_unique_candidates_per_parent) - context.parent_unique_count
                if context.remaining_unique_slots <= 0:
                    context.refill_stop_reason = 'max_unique_candidates_reached'
                    break
                context.refill_round_count += 1
                context.round_candidate_limit = min(int(system.cfg.candidate_refill_candidates_per_round), context.remaining_unique_slots)
                context.refill_requested_candidate_count += context.round_candidate_limit
                context.missing_requirements = list(context.requirements.get('missing', []))
                context.all_schema_invalid = bool(context.prior_failures) and all(
                    'schema' in ' '.join(str(reason) for reason in failure.get('reasons', []))
                    for failure in context.prior_failures
                )
                if context.all_schema_invalid or any('repair' in str(value) or 'schema' in str(value) for value in context.missing_requirements):
                    context.refill_generator_type = 'tcs_repair'
                else:
                    context.refill_generator_type = 'open_mechanism_exploration'
                context.refill_feedback = {'refill_round': context.refill_round_count, 'required_candidate_types_missing': list(context.requirements.get('missing', [])), 'previous_candidate_failures': context.prior_failures[-6:] if bool(system.cfg.candidate_refill_feed_rejection_reasons) else [], 'preserve_successes': ['Preserve competence and any valid mechanism steps from safe candidates.']}
                context.refill_feedback.update({
                    'refill_generator_type': context.refill_generator_type,
                    'refill_missing_requirement': ','.join(str(value) for value in context.missing_requirements),
                })
                context.refill_job = {'parent_idx': 0, 'parent': context.active_parent, 'parent_prompt': str(context.active_parent.get('prompt', context.agent.current_prompt)), 'parent_id': str(context.active_parent.get('id', system._hash(context.agent.current_prompt))), 'parent_batches': list(context.generation_batches), 'refill_feedback': context.refill_feedback}
                context.refill_result = await context.propose_for_parent(context.refill_job)
                context.proposals = context.refill_result.get('proposals', []) if isinstance(context.refill_result.get('proposals', []), list) else []
                if not context.proposals:
                    context.refill_stop_reason = 'optimizer_failure'
                    break
                context.new_candidates = []
                for context.index, context.proposal in enumerate(context.proposals[:context.round_candidate_limit]):
                    context.prompt = str(context.proposal.get('candidate_prompt', '')).strip()
                    context.prompt, context._ = system._sanitize_prompt(context.prompt, context.agent_id)
                    context.candidate = system._make_refill_candidate(proposal=context.proposal, prompt=context.prompt, parent_id=context.refill_job['parent_id'], parent_prompt=context.refill_job['parent_prompt'], agent_id=context.agent_id, candidate_index=context.index, refill_round=context.refill_round_count, generation=context.generation)
                    context.candidate.update({
                        'refill_generator_type': context.refill_generator_type,
                        'refill_missing_requirement': context.refill_feedback['refill_missing_requirement'],
                        'refill_round': context.refill_round_count,
                        'refill_candidate_source': str(context.proposal.get('candidate_source', '')),
                    })
                    context.refill_metadata = {
                        'optimizer_architecture': str(context.candidate.get('optimizer_architecture', '')),
                        'candidate_source': str(context.candidate.get('candidate_source', '')),
                        'candidate_pool_source': 'optimizer',
                        'tcs_call_group_id': str(context.candidate.get('tcs_call_group_id', '') or ''),
                        'execution_session_id': str(context.candidate.get('execution_session_id', '') or ''),
                        'update_attempt_id': str(context.candidate.get('update_attempt_id', '') or ''),
                        **dict(context.candidate.get('optimizer_generation_diagnostics', {}) or {}),
                    }
                    context.refill_metadata_errors = validate_tcs_candidate_metadata(context.refill_metadata)
                    if context.refill_metadata_errors:
                        raise RuntimeError(
                            f"Invalid refill TCS candidate metadata: agent_id={context.agent_id} "
                            f"epoch={context.epoch_id} step={context.step_id} "
                            f"parent_id={context.refill_job['parent_id']} "
                            f"metadata_errors={','.join(context.refill_metadata_errors)}"
                        )
                    system._record_candidate_funnel_item(context.candidate, context.agent_id, "schema_valid_candidate_count")
                    context.prescreen = cheap_prescreen(context.candidate, system._normalized_prompt_hash(context.refill_job['parent_prompt']), context.seen, parent=context.active_parent)
                    if context.prescreen:
                        context.candidate['cheap_prescreen_reasons'] = context.prescreen
                        context.prior_failures.append({'candidate_type': str(context.proposal.get('candidate_type', '')), 'failure_stage': 'cheap_prescreen', 'reasons': context.prescreen})
                        continue
                    context.seen.add(normalize_spaces(context.prompt).lower())
                    context.new_candidates.append(context.candidate)
                    system._record_candidate_funnel_item(context.candidate, context.agent_id, "prescreen_pass_count")
                if not context.new_candidates:
                    context.refill_stop_reason = 'no_new_unique_candidate'
                    break
                context.refill_actual_candidate_count += len(context.new_candidates)
                context.solver_cap = int(system.cfg.candidate_refill_max_solver_calls_per_agent_update)
                if str(system.cfg.candidate_eval_execution_mode) == 'factorized_cached' and context.solver_cap <= 0:
                    await system._prewarm_factorized_candidate_rollouts(agent_id=context.agent_id, eval_batch=context.eval_batch, peer_prompts=context.peer_prompts, candidate_pool=context.new_candidates)
                for context.candidate in context.new_candidates:
                    context.candidate_call_upper_bound = len(context.eval_batch)
                    if context.solver_cap > 0 and context.refill_solver_calls + context.candidate_call_upper_bound > context.solver_cap:
                        context.refill_stop_reason = 'max_unique_candidates_reached'
                        context.refill_solver_call_limit_reached = True
                        break
                    context.metrics = await system.evaluate_candidate_prompt(context.agent_id, context.candidate['prompt'], context.peer_prompts, context.eval_batch, role_spec=context.candidate['proposal'], baseline_homogeneous_cases=context.baseline_cases)
                    context.candidate['metrics'] = context.metrics
                    context.candidate['reward'] = float(context.metrics.get('reward', 0.0) or 0.0)
                    context.proposal = context.candidate['proposal']
                    context.candidate['metrics'].update({'candidate_type': str(context.proposal.get('candidate_type', '')), 'mechanism_steps': list(context.proposal.get('mechanism_steps', []))})
                    system._attach_stable_mechanism_representation(context.candidate)
                    system._mark_mechanism_novelty(context.candidate, parent=context.active_parent, existing=[*getattr(context.agent, 'safe_qd_archive', []), *getattr(context.agent, 'probation_archive', []), *context.evaluated])
                    context.candidate['archive_bucket'] = candidate_quality_bucket(context.candidate, system.cfg)
                    context.evaluated.append(context.candidate)
                    context.refill_solver_calls += int(context.metrics.get('solver_calls', 0) or 0)
                if context.solver_cap > 0 and context.refill_solver_calls >= context.solver_cap:
                    context.refill_solver_call_limit_reached = True
                    break
                context.provisional_archive = select_safe_archive([*getattr(context.agent, 'safe_qd_archive', []), *context.evaluated], system._normalized_prompt_hash(context.agent.current_prompt), int(system.cfg.qd_archive_size_per_agent))
                context.provisional_representatives = select_joint_representatives(context.provisional_archive, system._normalized_prompt_hash(context.agent.current_prompt), int(system.cfg.joint_representative_beam_size), system.cfg)
                context.requirements = search_space_requirements(context.evaluated, context.provisional_archive, context.provisional_representatives, system._normalized_prompt_hash(context.agent.current_prompt), system.cfg)
                if context.requirements.get('met') and bool(system.cfg.candidate_refill_stop_when_requirements_met):
                    context.refill_stop_reason = 'requirements_met'
                    break
                context.parent_unique_count = sum((1 for context.item in context.evaluated if context.item.get('candidate_pool_source') == 'optimizer' and str(context.item.get('parent_id', '')) == str(context.refill_job['parent_id'])))
                if context.parent_unique_count >= int(system.cfg.candidate_refill_max_unique_candidates_per_parent):
                    context.refill_stop_reason = 'max_unique_candidates_reached'
                    break
            context.pareto_summary.update({'initial_candidate_count': context.num_optimizer_candidates, 'cheap_prescreen_rejection_count': sum((1 for context.failure in context.prior_failures if context.failure.get('failure_stage') == 'cheap_prescreen')), 'evaluated_candidate_count': len(context.evaluated), 'refill_round_count': context.refill_round_count, 'refill_requested_candidate_count': context.refill_requested_candidate_count, 'refill_actual_candidate_count': context.refill_actual_candidate_count, 'refill_trigger_reasons': context.refill_trigger_reasons, 'refill_stop_reason': context.refill_stop_reason, 'refill_solver_call_budget_used': context.refill_solver_calls, 'refill_solver_call_limit_reached': bool(context.refill_solver_call_limit_reached), **context.requirements})
            for context.item in context.evaluated:
                context.item['is_incumbent'] = str(context.item.get('prompt_hash', '')) == system._normalized_prompt_hash(context.agent.current_prompt)
                context.item['archive_bucket'] = 'safe' if context.item['is_incumbent'] else candidate_quality_bucket(context.item, system.cfg)
                if context.item.get('archive_bucket') == 'safe' and str(context.item.get('parent_id', '')) in context.prior_probation_ids:
                    system.probation_to_safe_conversion_count += 1
            context.converted_parent_ids = {str(context.item.get('parent_id', '')) for context.item in context.evaluated if context.item.get('archive_bucket') == 'safe' and str(context.item.get('parent_id', '')) in context.prior_probation_ids}
            context.agent.safe_qd_archive = select_safe_archive([*getattr(context.agent, 'safe_qd_archive', []), *context.evaluated], system._normalized_prompt_hash(context.agent.current_prompt), int(system.cfg.qd_archive_size_per_agent))
            context.new_probation = [context.item for context.item in context.evaluated if context.item.get('archive_bucket') == 'probation']
            [context.item.setdefault('probation_created_update', int(context.agent_update_turn)) for context.item in context.new_probation]
            context.retained_probation = [context.item for context.item in getattr(context.agent, 'probation_archive', []) if str(context.item.get('id', system._hash(str(context.item.get('prompt', ''))))) not in context.converted_parent_ids]
            context.agent.probation_archive = (context.new_probation + context.retained_probation)[:int(system.cfg.probation_archive_size_per_agent)]
            system._refresh_joint_representatives(context.agent); system._record_candidate_funnel_outcomes(agent_id=context.agent_id, evaluated=context.evaluated, safe_archive=context.agent.safe_qd_archive, epoch=context.epoch_id)
            context.selected = list(context.agent.prompt_beam)
            system._record_stable_qd_archive_snapshot(agent_id=context.agent_id, epoch=context.epoch_id, step=context.step_id, evaluated=context.evaluated, parent_sources=context.parent_sources)
            context.starvation = context.requirements['safe_non_incumbent_count'] == 0
            context.mechanism_starvation = context.requirements['safe_distinct_mechanism_count'] == 0
            system.candidate_starvation_count += int(context.starvation)
            system.mechanism_starvation_count += int(context.mechanism_starvation)
            system.search_branch_starvation_count += int(context.starvation and (not context.agent.probation_archive))
            system.refill_requirements_unmet_count += int(not context.requirements['met'])
            context.agent.optimizer_update_count_by_epoch[str(context.epoch_id)] = int(context.agent.optimizer_update_count_by_epoch.get(str(context.epoch_id), 0) or 0) + 1
        elif system._is_rollout_qd_method():
            _select_rollout_archive_for_update(system, context)
        else:
            context.requirements = {}

class ArchiveSelectionStage:

    @staticmethod
    async def run(system, context):
        context.selectable = [context.item for context.item in context.evaluated if bool(context.item.get('trajectory_feasible', True))]
        if not context.selectable:
            raise RuntimeError('Candidate guards removed the current active prompt fallback')
        context.beam_size = max(1, int(system.cfg.beam_size))
        if system._is_stable_qd_lineage() or system._is_rollout_qd_method():
            context.selected = list(context.agent.prompt_beam)
        elif system._is_v82_hybrid():
            context.selected, context.pareto_summary = system._select_hybrid_beam(context.selectable, context.beam_size, context.agent.current_prompt, agent_id=context.agent_id, epoch_id=context.epoch_id, step_id=context.step_id)
        elif system._uses_vote_pareto_selection():
            context.selected, context.pareto_summary = system._select_vote_pareto_beam(context.selectable, context.beam_size, context.agent.current_prompt)
        else:
            context.selectable.sort(key=lambda x: float(context.x.get('reward', 0.0)), reverse=True)
            context.selected = context.selectable[:context.beam_size]
            for context.item in context.evaluated:
                context.item['pareto_feasible'] = None
                context.item['pareto_rank'] = None
                context.item['pareto_crowding_distance'] = None
                context.item['pareto_selected'] = None
                context.item['pareto_forced_fallback'] = None
        context.top1_candidate_source = system._candidate_generation_source(context.selected[0]) if context.selected else ''
        context.top1_candidate_pool_source = system._candidate_pool_source(context.selected[0]) if context.selected else ''
        context.selected_by_id = {str(context.item.get('candidate_id', '')): context.rank for context.rank, context.item in enumerate(context.selected, start=1)}
        context.active_candidate_id = str(context.selected[0].get('candidate_id', '')) if context.selected else ''
        for context.item in context.selected:
            context.item.setdefault('metrics', {})['beam_slot'] = str(context.item.get('beam_slot', ''))
        if not (system._is_stable_qd_lineage() or system._is_rollout_qd_method()):
            context.agent.prompt_beam = [system._make_beam_item(prompt=str(context.x['prompt']), score=float(context.x.get('reward', 0.0)), metrics=context.x.get('metrics', {}), parent_id=context.x.get('parent_id'), generation=int(context.x.get('generation', context.generation) or context.generation), candidate_id=str(context.x.get('candidate_id', '')) or None) for context.x in context.selected] or [system._make_beam_item(context.agent.current_prompt, None, {}, None, 0)]
            context.agent.current_prompt = str(context.agent.prompt_beam[0]['prompt'])
        context.changed = context.old_hash != system._hash(context.agent.current_prompt)
        context.profile_before = dict(context.agent.capability_profile)
        if context.changed:
            context.agent.history.append(context.agent.current_prompt)
            context.agent.accept_count += 1
            if system._v7_residual_protocol_enabled():
                context.active_metrics = context.selected[0].get('metrics', {}) if context.selected else {}
                if system._residual_specialization_enabled():
                    system._update_vote_context_profile(context.agent, context.active_metrics)
                    system._accumulate_capability_evidence(context.agent, context.active_metrics, context.epoch_id)
                    system._flush_capability_profile(context.agent, context.epoch_id, force=False)
                context.agent.last_accepted_prompt_hash = system._normalized_prompt_hash(context.agent.current_prompt)
                context.fingerprint = {str(context.key): BehaviorFingerprintEntry.from_dict(context.value) for context.key, context.value in dict(context.active_metrics.get('behavior_fingerprint', {})).items() if isinstance(context.value, dict)}
                context.state = BehaviorStateSummary(state_id=f"e{int(context.epoch_id)}_s{int(context.step_id)}_a{int(context.agent_id)}_{str(context.selected[0].get('candidate_id', ''))}", epoch=int(context.epoch_id), prompt_hash=context.agent.last_accepted_prompt_hash, behavior_fingerprint=context.fingerprint, transition_vector={str(context.key): float(context.value) for context.key, context.value in dict(context.active_metrics.get('candidate_transition_vector', {})).items()}, target_accuracy=float(context.active_metrics.get('candidate_target_accuracy', 0.0) or 0.0), team_vote_accuracy=float(context.active_metrics.get('candidate_team_accuracy', 0.0) or 0.0), mean_vote_margin=float(context.active_metrics.get('candidate_mean_vote_margin', 0.0) or 0.0), preserved_mechanisms=[str(context.value) for context.value in context.selected[0].get('proposal', {}).get('preserved_mechanisms', [])] if isinstance(context.selected[0].get('proposal', {}).get('preserved_mechanisms', []), list) else [], capability_profile=dict(context.agent.capability_profile), paired_behavior_utility=system.behavior_fingerprint_utility(context.fingerprint))
                system._append_bounded_archive(context.agent.accepted_behavior_archive, context.state)
        else:
            context.agent.reject_count += 1

class CandidateEventStage:

    @staticmethod
    async def run(system, context):
        for context.item in context.evaluated:
            context.metrics = context.item.get('metrics', {})
            context.candidate_id = str(context.item.get('candidate_id', ''))
            context.rank = context.selected_by_id.get(context.candidate_id)
            context.accepted = context.rank is not None
            context.in_top_beam = bool(context.accepted)
            context.is_top1 = bool(context.candidate_id == context.active_candidate_id)
            context.active_evolution = bool(context.is_top1 and context.changed)
            if system._v7_residual_protocol_enabled() and system._candidate_pool_source(context.item) == 'optimizer':
                context.rejection_reason = str(context.metrics.get('rejection_reason', ''))
                context.retained_inactive = bool(context.in_top_beam and (not context.active_evolution))
                if not context.active_evolution and (not context.retained_inactive):
                    if not context.rejection_reason:
                        context.rejection_reason = 'not_selected'
                        context.metrics['rejection_reason'] = context.rejection_reason
                    context.rejected_state = RejectedBehaviorSummary(state_id=f'e{int(context.epoch_id)}_s{int(context.step_id)}_a{int(context.agent_id)}_{context.candidate_id}', epoch=int(context.epoch_id), prompt_hash=str(context.metrics.get('prompt_hash', system._normalized_prompt_hash(str(context.item.get('prompt', ''))))), parent_prompt_hash=str(context.metrics.get('parent_prompt_hash', '')), rejection_reason=context.rejection_reason, prompt_change_ratio=float(context.metrics.get('prompt_change_ratio', 0.0) or 0.0), max_behavior_cycle_similarity=float(context.metrics.get('max_behavior_cycle_similarity', 0.0) or 0.0), behavior_cycle_overlap=int(context.metrics.get('behavior_cycle_overlap', 0) or 0), transition_vector={str(context.key): float(context.value) for context.key, context.value in dict(context.metrics.get('candidate_transition_vector', {})).items()}, behavior_fingerprint={str(context.key): BehaviorFingerprintEntry.from_dict(context.value) for context.key, context.value in dict(context.metrics.get('behavior_fingerprint', {})).items() if isinstance(context.value, dict)}, paired_behavior_utility=system.behavior_fingerprint_utility(context.metrics.get('behavior_fingerprint', {})), failure_signature=f"{context.rejection_reason}|pivotal_loss={float(context.metrics.get('pivotal_loss_rate', 0.0) or 0.0):.4f}|shared_creation={float(context.metrics.get('shared_error_creation_score', 0.0) or 0.0):.4f}")
                    system._append_bounded_archive(context.agent.rejected_behavior_archive, context.rejected_state)
                    if context.rejection_reason == 'exact_prompt_cycle':
                        context.agent.duplicate_prompt_reject_count += 1
                    elif context.rejection_reason in {'behavior_cycle', 'accepted_state_cycle', 'rejected_failure_cycle'}:
                        context.agent.cycle_reject_count += 1
                    elif context.rejection_reason == 'unsupported_large_prompt_shift':
                        context.agent.large_shift_reject_count += 1
                system.trajectory_events.append(system._trajectory_event(agent_id=context.agent_id, epoch_id=context.epoch_id, step_id=context.step_id, item=context.item, accepted=context.active_evolution, profile_before=context.profile_before, profile_after=dict(context.agent.capability_profile)))
                if context.retained_inactive:
                    system.trajectory_events[-1]['decision'] = 'retained_beam_inactive'
            context.active_selection_key = list(system._competence_depth_sort_key(context.item) if system._uses_competence_depth_pareto_selection() else system._vote_pareto_active_sort_key(context.item)) if system._uses_vote_pareto_selection() and context.accepted else None
            context.item_diagnostics = system._empty_optimizer_generation_diagnostics()
            if isinstance(context.item.get('optimizer_generation_diagnostics', {}), dict):
                context.item_diagnostics.update(context.item.get('optimizer_generation_diagnostics', {}))
            context.item_diagnostics['optimizer_underfilled'] = bool(context.optimizer_underfilled)
            context.tcs_candidate_metadata = {'optimizer_architecture': context.item_diagnostics.get('optimizer_architecture', ''), 'candidate_source': system._candidate_generation_source(context.item), 'candidate_pool_source': system._candidate_pool_source(context.item), 'tcs_call_group_id': context.item.get('tcs_call_group_id', context.item_diagnostics.get('tcs_call_group_id', '')), 'execution_session_id': context.item.get('execution_session_id', context.item_diagnostics.get('execution_session_id', system._current_execution_session_id())), 'update_attempt_id': context.item.get('update_attempt_id', context.item_diagnostics.get('update_attempt_id', context.update_attempt_id)), **context.item_diagnostics}
            context.is_tcs_metadata_applicable = tcs_metadata_applicable(context.tcs_candidate_metadata)
            context.tcs_metadata_errors = validate_tcs_candidate_metadata(context.tcs_candidate_metadata)
            system.update_logs.append({**system._base_log_fields(), 'event': 'candidate_evaluated', 'epoch': context.epoch_id, 'step': context.step_id, 'agent_id': context.agent_id, 'search_mode': 'evolutionary_beam', 'beam_size': context.beam_size, 'candidate_id': context.item.get('candidate_id', ''), 'candidate_selection_mode': str(getattr(system.cfg, 'candidate_selection_mode', 'scalar_reward')), 'parent_id': context.item.get('parent_id'), 'tcs_call_group_id': str(context.item.get('tcs_call_group_id', context.item_diagnostics.get('tcs_call_group_id', '')) or ''), 'execution_session_id': str(context.item.get('execution_session_id', context.item_diagnostics.get('execution_session_id', system._current_execution_session_id())) or system._current_execution_session_id()), 'update_attempt_id': str(context.item.get('update_attempt_id', context.item_diagnostics.get('update_attempt_id', context.update_attempt_id)) or context.update_attempt_id), 'reward': float(context.metrics.get('reward', 0.0)), 'reward_total': float(context.metrics.get('reward_total', context.metrics.get('reward', 0.0))), 'embedding_diversity': float(context.metrics.get('embedding_diversity', 0.0)), 'mean_embedding_overlap': float(context.metrics.get('mean_embedding_overlap', 0.0)), 'target_overlap_pressure': float(context.metrics.get('target_overlap_pressure', 0.0)), 'homogeneous_case_count': float(context.metrics.get('homogeneous_case_count', 0.0)), 'resolved_case_count': float(context.metrics.get('resolved_case_count', 0.0)), 'new_homogeneous_case_count': float(context.metrics.get('new_homogeneous_case_count', 0.0)), 'team_accuracy': float(context.metrics.get('team_accuracy', 0.0)), 'target_agent_accuracy': float(context.metrics.get('target_agent_accuracy', 0.0)), 'invalid_rate': float(context.metrics.get('invalid_rate', 0.0)), 'invalid_score': float(context.metrics.get('invalid_score', 0.0)), 'baseline_team_accuracy': float(context.metrics.get('baseline_team_accuracy', 0.0)), 'candidate_team_accuracy': float(context.metrics.get('candidate_team_accuracy', context.metrics.get('team_accuracy', 0.0))), 'accuracy_delta': float(context.metrics.get('accuracy_delta', 0.0)), 'vote_delta': float(context.metrics.get('vote_delta', context.metrics.get('accuracy_delta', 0.0))), 'vote_gain_count': int(context.metrics.get('vote_gain_count', 0)), 'vote_gain_rate': float(context.metrics.get('vote_gain_rate', 0.0)), 'vote_loss_count': int(context.metrics.get('vote_loss_count', 0)), 'vote_loss_rate': float(context.metrics.get('vote_loss_rate', 0.0)), 'net_vote_count': int(context.metrics.get('net_vote_count', 0)), 'net_vote_delta': float(context.metrics.get('net_vote_delta', context.metrics.get('vote_delta', 0.0))), 'plurality_vote_gain_count': int(context.metrics.get('plurality_vote_gain_count', context.metrics.get('vote_gain_count', 0))), 'plurality_vote_gain_rate': float(context.metrics.get('plurality_vote_gain_rate', context.metrics.get('vote_gain_rate', 0.0))), 'plurality_vote_loss_count': int(context.metrics.get('plurality_vote_loss_count', context.metrics.get('vote_loss_count', 0))), 'plurality_vote_loss_rate': float(context.metrics.get('plurality_vote_loss_rate', context.metrics.get('vote_loss_rate', 0.0))), 'plurality_vote_net_count': int(context.metrics.get('plurality_vote_net_count', context.metrics.get('net_vote_count', 0))), 'plurality_vote_net_delta': float(context.metrics.get('plurality_vote_net_delta', context.metrics.get('net_vote_delta', 0.0))), 'plurality_pivotal_fix_opportunity_count': int(context.metrics.get('plurality_pivotal_fix_opportunity_count', 0)), 'plurality_pivotal_fix_opportunity_rate': float(context.metrics.get('plurality_pivotal_fix_opportunity_rate', 0.0)), 'plurality_pivotal_fix_count': int(context.metrics.get('plurality_pivotal_fix_count', 0)), 'plurality_pivotal_fix_rate': float(context.metrics.get('plurality_pivotal_fix_rate', 0.0)), 'plurality_pivotal_loss_count': int(context.metrics.get('plurality_pivotal_loss_count', 0)), 'plurality_pivotal_loss_rate': float(context.metrics.get('plurality_pivotal_loss_rate', 0.0)), 'plurality_boundary_shared_error_net_gain': float(context.metrics.get('plurality_boundary_shared_error_net_gain', 0.0)), 'pivotal_definition': str(context.metrics.get('pivotal_definition', '')), 'baseline_gold_vote_count': float(context.metrics.get('baseline_gold_vote_count', 0.0)), 'candidate_gold_vote_count': float(context.metrics.get('candidate_gold_vote_count', 0.0)), 'baseline_largest_wrong_vote_count': float(context.metrics.get('baseline_largest_wrong_vote_count', 0.0)), 'candidate_largest_wrong_vote_count': float(context.metrics.get('candidate_largest_wrong_vote_count', 0.0)), 'baseline_plurality_margin_votes': float(context.metrics.get('baseline_plurality_margin_votes', 0.0)), 'candidate_plurality_margin_votes': float(context.metrics.get('candidate_plurality_margin_votes', 0.0)), 'plurality_margin_vote_delta': float(context.metrics.get('plurality_margin_vote_delta', 0.0)), 'baseline_normalized_plurality_margin': float(context.metrics.get('baseline_normalized_plurality_margin', -1.0)), 'candidate_normalized_plurality_margin': float(context.metrics.get('candidate_normalized_plurality_margin', -1.0)), 'normalized_plurality_margin_delta': float(context.metrics.get('normalized_plurality_margin_delta', 0.0)), 'baseline_plurality_vote_tie': float(context.metrics.get('baseline_plurality_vote_tie', 0.0)), 'candidate_plurality_vote_tie': float(context.metrics.get('candidate_plurality_vote_tie', 0.0)), 'baseline_mean_vote_margin': float(context.metrics.get('baseline_mean_vote_margin', -1.0)), 'candidate_mean_vote_margin': float(context.metrics.get('candidate_mean_vote_margin', -1.0)), 'vote_margin_delta': float(context.metrics.get('vote_margin_delta', 0.0)), 'baseline_boundary_useful_diversity': float(context.metrics.get('baseline_boundary_useful_diversity', 0.0)), 'candidate_boundary_useful_diversity': float(context.metrics.get('candidate_boundary_useful_diversity', 0.0)), 'boundary_useful_diversity_delta': float(context.metrics.get('boundary_useful_diversity_delta', 0.0)), 'boundary_diversity_gain': float(context.metrics.get('boundary_diversity_gain', 0.0)), 'reward_component_target_accuracy': float(context.metrics.get('reward_component_target_accuracy', 0.0)), 'reward_component_vote_delta': float(context.metrics.get('reward_component_vote_delta', 0.0)), 'reward_component_vote_margin': float(context.metrics.get('reward_component_vote_margin', 0.0)), 'reward_component_boundary_diversity': float(context.metrics.get('reward_component_boundary_diversity', 0.0)), 'reward_component_invalid_penalty': float(context.metrics.get('reward_component_invalid_penalty', 0.0)), 'reward_component_guard_penalty': float(context.metrics.get('reward_component_guard_penalty', 0.0)), 'baseline_oracle_acc': float(context.metrics.get('baseline_oracle_acc', 0.0)), 'candidate_oracle_acc': float(context.metrics.get('candidate_oracle_acc', 0.0)), 'coverage_delta': float(context.metrics.get('coverage_delta', 0.0)), 'coverage_gain_count': int(context.metrics.get('coverage_gain_count', 0)), 'coverage_gain_rate': float(context.metrics.get('coverage_gain_rate', 0.0)), 'coverage_loss_count': int(context.metrics.get('coverage_loss_count', 0)), 'coverage_loss_rate': float(context.metrics.get('coverage_loss_rate', 0.0)), 'net_coverage_count': int(context.metrics.get('net_coverage_count', 0)), 'net_coverage_delta': float(context.metrics.get('net_coverage_delta', 0.0)), **{context.key: context.metrics.get(context.key, 0) for context.depth in range(1, 4) for context.key in (f'baseline_coverage_depth_c{context.depth}', f'candidate_coverage_depth_c{context.depth}', f'depth{context.depth}_gain_count', f'depth{context.depth}_gain_rate', f'depth{context.depth}_loss_count', f'depth{context.depth}_loss_rate', f'depth{context.depth}_net_count', f'depth{context.depth}_net_delta')}, 'competence_reward_component': float(context.metrics.get('competence_reward_component', 0.0)), 'v7_reward_component': float(context.metrics.get('v7_reward_component', 0.0)), 'effective_reward_specialization_strength': float(context.metrics.get('effective_reward_specialization_strength', 0.0)), 'final_reward': float(context.metrics.get('final_reward', context.metrics.get('reward', 0.0))), 'stage_aux_objective': float(context.metrics.get('stage_aux_objective', 0.0)), 'stage_aux_depth2_component': float(context.metrics.get('stage_aux_depth2_component', 0.0)), 'stage_aux_boundary_component': float(context.metrics.get('stage_aux_boundary_component', 0.0)), 'baseline_target_accuracy': float(context.metrics.get('baseline_target_accuracy', 0.0)), 'candidate_target_accuracy': float(context.metrics.get('candidate_target_accuracy', context.metrics.get('target_agent_accuracy', 0.0))), 'rescue_rate': float(context.metrics.get('rescue_rate', 0.0)), 'useful_diversity': float(context.metrics.get('useful_diversity', 0.0)), 'rescue_useful_diversity': float(context.metrics.get('rescue_useful_diversity', 0.0)), 'baseline_embedding_diversity': float(context.metrics.get('baseline_embedding_diversity', 0.0)), 'candidate_embedding_diversity': float(context.metrics.get('candidate_embedding_diversity', context.metrics.get('embedding_diversity', 0.0))), 'diversity_delta': float(context.metrics.get('diversity_delta', 0.0)), 'baseline_invalid_rate': float(context.metrics.get('baseline_invalid_rate', 0.0)), 'candidate_invalid_rate': float(context.metrics.get('candidate_invalid_rate', context.metrics.get('invalid_rate', 0.0))), 'invalid_delta': float(context.metrics.get('invalid_delta', 0.0)), 'behavior_context_counts': context.metrics.get('behavior_context_counts', {}), 'candidate_transition_vector': context.metrics.get('candidate_transition_vector', {}), 'candidate_transition_support': context.metrics.get('candidate_transition_support', {}), **system._candidate_v7_log_fields(context.metrics), 'capability_profile_before': context.profile_before, 'capability_profile_after': dict(context.agent.capability_profile), 'prompt_hash': str(context.metrics.get('prompt_hash', '')), 'parent_prompt_hash': str(context.metrics.get('parent_prompt_hash', '')), 'prompt_change_ratio': float(context.metrics.get('prompt_change_ratio', 0.0) or 0.0), 'max_behavior_cycle_similarity': float(context.metrics.get('max_behavior_cycle_similarity', 0.0) or 0.0), 'behavior_cycle_overlap': int(context.metrics.get('behavior_cycle_overlap', 0) or 0), 'matched_behavior_state_id': str(context.metrics.get('matched_behavior_state_id', '')), 'exact_prompt_cycle': bool(context.metrics.get('exact_prompt_cycle', False)), 'behavior_cycle_guard_passed': bool(context.metrics.get('behavior_cycle_guard_passed', True)), 'prompt_trust_region_passed': bool(context.metrics.get('prompt_trust_region_passed', True)), 'rejection_reason': str(context.metrics.get('rejection_reason', '')), 'accuracy_guard_passed': bool(context.metrics.get('accuracy_guard_passed', True)), 'invalid_guard_passed': bool(context.metrics.get('invalid_guard_passed', True)), 'competence_depth1_guard_enabled': bool(context.metrics.get('competence_depth1_guard_enabled', context.candidate_guard_enabled)), 'competence_depth1_guard_epsilon': float(context.metrics.get('competence_depth1_guard_epsilon', 0.0) or 0.0), 'competence_depth1_guard_passed': bool(context.metrics.get('competence_depth1_guard_passed', True)), 'hard_guard_passed': bool(context.metrics.get('hard_guard_passed', True)), 'hard_rejection_reason': str(context.metrics.get('rejection_reason', '')), 'candidate_type': str(context.metrics.get('candidate_type', '')), 'archive_bucket': str(context.item.get('archive_bucket', '')), 'cheap_prescreen_reasons': list(context.item.get('cheap_prescreen_reasons', [])), 'refill_candidate': bool(context.item.get('refill_candidate', False)), 'mechanism_signature': context.metrics.get('mechanism_signature', []), 'parent_mechanism_signature': context.metrics.get('parent_mechanism_signature', []), 'peer_dominant_mechanism_signature': context.metrics.get('peer_dominant_mechanism_signature', []), 'mechanism_signature_distance': float(context.metrics.get('mechanism_signature_distance', 0.0) or 0.0), 'raw_reward': float(context.metrics.get('raw_reward', context.metrics.get('reward', 0.0)) or 0.0), 'penalized_reward': float(context.metrics.get('penalized_reward', context.metrics.get('reward', 0.0)) or 0.0), 'soft_guard_penalty': float(context.metrics.get('soft_guard_penalty', 0.0) or 0.0), 'soft_error_dependence_penalty': float(context.metrics.get('soft_error_dependence_penalty', 0.0) or 0.0), 'soft_cycle_penalty': float(context.metrics.get('soft_cycle_penalty', 0.0) or 0.0), 'soft_mechanism_shift_penalty': float(context.metrics.get('soft_mechanism_shift_penalty', 0.0) or 0.0), 'soft_accuracy_regression_penalty': float(context.metrics.get('soft_accuracy_regression_penalty', 0.0) or 0.0), 'soft_guard_reasons': context.metrics.get('soft_guard_reasons', []), 'beam_slot': str(context.item.get('beam_slot', 'not_retained')), 'pareto_feasible': context.item.get('pareto_feasible'), 'pareto_rank': context.item.get('pareto_rank'), 'pareto_crowding_distance': context.item.get('pareto_crowding_distance'), 'pareto_selected': context.item.get('pareto_selected'), 'active_selection_key': context.active_selection_key, 'effective_weight_target_accuracy': float(context.metrics.get('effective_weight_target_accuracy', 0.0)), 'effective_weight_div_delta': float(context.metrics.get('effective_weight_div_delta', 0.0)), 'effective_weight_vote_delta': float(context.metrics.get('effective_weight_vote_delta', 0.0)), 'effective_weight_vote_margin': float(context.metrics.get('effective_weight_vote_margin', 0.0)), 'effective_weight_boundary_diversity': float(context.metrics.get('effective_weight_boundary_diversity', 0.0)), 'effective_accuracy_guard_epsilon': float(context.metrics.get('effective_accuracy_guard_epsilon', 0.0)), 'reward_phase_progress': float(context.metrics.get('reward_phase_progress', 0.0)), 'reward_diversity_need': float(context.metrics.get('reward_diversity_need', 0.0)), 'reward_unique_prompt_ratio': float(context.metrics.get('reward_unique_prompt_ratio', 0.0)), 'reward_accepted_updates': float(context.metrics.get('reward_accepted_updates', 0.0)), 'solver_reuse_enabled': bool(context.metrics.get('solver_reuse_enabled', False)), 'solver_reuse_hits': int(context.metrics.get('solver_reuse_hits', 0)), 'solver_reuse_misses': int(context.metrics.get('solver_reuse_misses', 0)), 'solver_calls': int(context.metrics.get('solver_calls', 0)), 'solver_reuse_total': int(context.metrics.get('solver_reuse_total', 0)), 'solver_reuse_hit_rate': float(context.metrics.get('solver_reuse_hit_rate', 0.0)), 'accepted': bool(context.accepted), 'in_top_beam': bool(context.in_top_beam), 'is_top1': bool(context.is_top1), 'active_prompt_changed': bool(context.changed), 'top1_candidate_source': context.top1_candidate_source, 'top1_candidate_pool_source': context.top1_candidate_pool_source, 'rank_in_beam': context.rank, 'beam_rank': context.rank, 'prompt_preview': normalize_spaces(str(context.item.get('prompt', '')))[:220], 'optimizer_model': system.cfg.optimizer_model, 'evaluator_model': system.cfg.evaluator_model, 'candidate_source': system._candidate_generation_source(context.item), 'candidate_pool_source': system._candidate_pool_source(context.item), 'generation_batch_type': context.item.get('generation_batch_type', ''), 'generation_case_ids': context.item.get('generation_case_ids', []), 'target_error_pattern': context.item.get('target_error_pattern', ''), 'accuracy_repair_rule': context.item.get('accuracy_repair_rule', ''), 'expected_accuracy_effect': context.item.get('expected_accuracy_effect', ''), 'num_target_error_cases': int(context.num_target_error_cases), 'num_accuracy_repair_candidates': int(context.num_accuracy_repair_candidates), 'num_diversity_candidates': int(context.num_diversity_candidates), 'optimizer_fallback_mode': str(getattr(system.cfg, 'optimizer_fallback_mode', 'none')), 'optimizer_parent_concurrency': int(context.parent_concurrency), 'fallback_enabled': bool(context.fallback_enabled), 'optimizer_underfilled': bool(context.optimizer_underfilled), 'requested_optimizer_candidates': int(context.requested_optimizer_candidates), 'num_optimizer_candidates': int(context.num_optimizer_candidates), 'num_fallback_candidates': int(context.num_fallback_candidates), 'num_existing_beam_candidates': int(context.num_existing_beam_candidates), 'optimizer_architecture': str(context.item_diagnostics.get('optimizer_architecture', getattr(system.cfg, 'optimizer_architecture', 'one_shot'))), 'teacher_question': context.item_diagnostics.get('teacher_question', ''), 'teacher_question_approved': bool(context.item_diagnostics.get('teacher_question_approved', False)), 'teacher_question_forced_best_score': bool(context.item_diagnostics.get('teacher_question_forced_best_score', False)), 'teacher_question_forced_best_round': int(context.item_diagnostics.get('teacher_question_forced_best_round', 0) or 0), 'teacher_question_forced_below_threshold': bool(context.item_diagnostics.get('teacher_question_forced_below_threshold', False)), 'teacher_question_score': system._safe_float(context.item_diagnostics.get('teacher_question_score', 0.0), 0.0), 'teacher_critic_rounds': int(context.item_diagnostics.get('teacher_critic_rounds', 0) or 0), 'teacher_quality_critique': str(context.item_diagnostics.get('teacher_quality_critique', '')), 'teacher_specificity_critique': str(context.item_diagnostics.get('teacher_specificity_critique', '')), 'teacher_task_alignment_critique': str(context.item_diagnostics.get('teacher_task_alignment_critique', '')), 'teacher_error_alignment_critique': str(context.item_diagnostics.get('teacher_error_alignment_critique', '')), 'teacher_diversity_critique': str(context.item_diagnostics.get('teacher_diversity_critique', '')), 'teacher_rewrite_count': int(context.item_diagnostics.get('teacher_rewrite_count', 0) or 0), 'student_candidate_count_raw': int(context.item_diagnostics.get('student_candidate_count_raw', 0) or 0), 'student_candidate_count_final': int(context.item_diagnostics.get('student_candidate_count_final', 0) or 0), 'student_candidate_filtered_count': int(context.item_diagnostics.get('student_candidate_filtered_count', 0) or 0), 'student_candidate_filter_reasons': context.item_diagnostics.get('student_candidate_filter_reasons', []), 'student_all_candidates_filtered': bool(context.item_diagnostics.get('student_all_candidates_filtered', False)), 'student_missing_required_field_count': int(context.item_diagnostics.get('student_missing_required_field_count', 0) or 0), 'student_missing_required_fields': context.item_diagnostics.get('student_missing_required_fields', []), **system._student_failure_log_fields(context.item_diagnostics), 'tcs_metadata_applicable': context.is_tcs_metadata_applicable, 'tcs_metadata_valid': not context.tcs_metadata_errors if context.is_tcs_metadata_applicable else None, 'tcs_metadata_errors': context.tcs_metadata_errors, 'diversity_contribution': str(context.item.get('diversity_contribution', '')), 'error_correlation_reduction': str(context.item.get('error_correlation_reduction', '')), 'task_alignment_rule': str(context.item.get('task_alignment_rule', '')), 'peer_redundancy_avoidance': str(context.item.get('peer_redundancy_avoidance', '')), 'declared_mechanism': str(context.item.get('proposal', {}).get('modified_mechanism', context.item.get('proposal', {}).get('new_or_modified_mechanism', context.item.get('proposal', {}).get('mechanism_name', '')))) if isinstance(context.item.get('proposal', {}), dict) else '', 'candidate_prompt_char_count': int(context.item.get('candidate_prompt_char_count', len(str(context.item.get('prompt', '')))) or 0), 'candidate_prompt_over_soft_limit': bool(context.item.get('candidate_prompt_over_soft_limit', False)), 'candidate_prompt_over_hard_limit': bool(context.item.get('candidate_prompt_over_hard_limit', False)), 'candidate_prompt_overlength_rejected': bool(context.item.get('candidate_prompt_overlength_rejected', False)), 'candidate_prompt_ends_with_sentence_boundary': bool(context.item.get('candidate_prompt_ends_with_sentence_boundary', system._prompt_ends_with_sentence_boundary(str(context.item.get('prompt', ''))))), 'optimizer_raw_response_empty': int(context.item_diagnostics.get('optimizer_raw_response_empty', 0) or 0), 'optimizer_json_parse_failed': int(context.item_diagnostics.get('optimizer_json_parse_failed', 0) or 0), 'optimizer_raw_candidate_count': int(context.item_diagnostics.get('optimizer_raw_candidate_count', 0) or 0), 'optimizer_empty_prompt_count': int(context.item_diagnostics.get('optimizer_empty_prompt_count', 0) or 0), 'optimizer_sanitized_count': int(context.item_diagnostics.get('optimizer_sanitized_count', 0) or 0), 'optimizer_redundant_filtered_count': int(context.item_diagnostics.get('optimizer_redundant_filtered_count', 0) or 0), 'optimizer_schema_filtered_count': int(context.item_diagnostics.get('optimizer_schema_filtered_count', 0) or 0), 'optimizer_final_candidate_count': int(context.item_diagnostics.get('optimizer_final_candidate_count', 0) or 0), 'num_eval_samples': int(context.metrics.get('num_eval_samples', 0)), 'candidate_eval_strategy': str(context.metrics.get('candidate_eval_strategy', getattr(system.cfg, 'candidate_eval_strategy', 'random'))), 'candidate_eval_pool_size': int(context.metrics.get('candidate_eval_pool_size', getattr(system.cfg, 'candidate_eval_pool_size', 0))), 'candidate_eval_pool_actual_size': int(context.metrics.get('candidate_eval_pool_actual_size', getattr(system.cfg, 'candidate_eval_pool_actual_size', 0))), 'candidate_eval_batch_size': int(context.metrics.get('candidate_eval_batch_size', getattr(system.cfg, 'candidate_eval_batch_size', 0))), 'actual_eval_batch_size': int(context.metrics.get('actual_eval_batch_size', context.metrics.get('num_eval_samples', 0))), 'num_eval_repeats': int(context.metrics.get('num_eval_repeats', getattr(system.cfg, 'candidate_eval_repeats', 1))), 'candidate_eval_data_source': str(context.metrics.get('candidate_eval_data_source', getattr(system.cfg, 'candidate_eval_data_source', 'optimization_train'))), 'candidate_eval_total_count': int(context.metrics.get('candidate_eval_total_count', context.metrics.get('actual_eval_batch_size', 0))), 'candidate_eval_unique_question_count': int(context.metrics.get('candidate_eval_unique_question_count', context.metrics.get('actual_eval_batch_size', 0))), 'candidate_eval_repeat_count': int(context.metrics.get('candidate_eval_repeat_count', getattr(system.cfg, 'candidate_eval_repeats', 1))), **context.competence_log_fields})
        if system._is_state_conditioned_method():
            system.update_logs.append({
                **system._base_log_fields(),
                'event': 'candidate_batch_audit',
                'epoch': context.epoch_id,
                'step': context.step_id,
                'agent_id': context.agent_id,
                'optimization_routes': sorted({
                    str(batch.get('optimization_route', 'general_accuracy') or 'general_accuracy')
                    for batch in context.generation_batches if isinstance(batch, dict)
                }),
                'candidate_state_metrics': [
                    {
                        'candidate_id': str(item.get('candidate_id', '')),
                        'optimization_route': str(item.get('optimization_route', 'general_accuracy') or 'general_accuracy'),
                        'candidate_target_accuracy': float(item.get('metrics', {}).get('candidate_target_accuracy', 0.0) or 0.0),
                        'representative_target_accuracy': float(item.get('metrics', {}).get('representative_pool_candidate_target_accuracy', 0.0) or 0.0),
                        'accuracy_delta': float(item.get('metrics', {}).get('accuracy_delta', 0.0) or 0.0),
                        'representative_accuracy_delta': float(item.get('metrics', {}).get('representative_accuracy_delta', 0.0) or 0.0),
                        'all_pools_accuracy_delta': float(item.get('metrics', {}).get('all_pools_accuracy_delta', 0.0) or 0.0),
                        'invalid_delta': float(item.get('metrics', {}).get('invalid_delta', 0.0) or 0.0),
                        'representative_invalid_delta': float(item.get('metrics', {}).get('representative_invalid_delta', 0.0) or 0.0),
                        'all_pools_invalid_delta': float(item.get('metrics', {}).get('all_pools_invalid_delta', 0.0) or 0.0),
                        'coverage_c0_to_c1_count': int(item.get('metrics', {}).get('coverage_pool_c0_to_c1_count', 0) or 0),
                        'coverage_c1_to_c2_count': int(item.get('metrics', {}).get('coverage_pool_c1_to_c2_count', 0) or 0),
                        'conversion_c2_to_c3_count': int(item.get('metrics', {}).get('conversion_pool_c2_to_c3_count', 0) or 0),
                        'conversion_wrong_cluster_reduction': int(item.get('metrics', {}).get('conversion_pool_c2_wrong_cluster_reduction', 0) or 0),
                        'state_quality_guard_passed': bool(item.get('metrics', {}).get('state_quality_guard_passed', False)),
                    }
                    for item in context.evaluated
                ],
                **system._candidate_eval_audit_fields(context.eval_batch),
            })


class UpdateSummaryStage:

    @staticmethod
    async def run(system, context):
        system._append_prompt_history_event(context.agent_id, context.epoch_id, context.step_id, 'beam_accept' if context.changed else 'beam_keep', context.changed)
        if bool(getattr(system.cfg, 'candidate_eval_cache_logging', True)):
            if not hasattr(system, 'cost_summary'):
                system.cost_summary = system._empty_cost_summary()
            system.cost_summary['candidate_eval_solver_api_calls'] = int(system.cost_summary.get('candidate_eval_solver_api_calls', 0) or 0) + int(context.candidate_eval_cache_stats.get('candidate_eval_solver_api_call_count', 0) or 0)
            system.cost_summary['candidate_eval_cache_hits'] = int(system.cost_summary.get('candidate_eval_cache_hits', 0) or 0) + int(context.candidate_eval_cache_stats.get('candidate_eval_memory_cache_hit_count', 0) or 0) + int(context.candidate_eval_cache_stats.get('candidate_eval_persisted_cache_hit_count', 0) or 0)
            system.cost_summary['candidate_eval_inflight_reuses'] = int(system.cost_summary.get('candidate_eval_inflight_reuses', 0) or 0) + int(context.candidate_eval_cache_stats.get('candidate_eval_inflight_reuse_count', 0) or 0)
            system.cost_summary['candidate_eval_calls_saved_vs_naive'] = int(system.cost_summary.get('candidate_eval_calls_saved_vs_naive', 0) or 0) + int(context.candidate_eval_cache_stats.get('candidate_eval_calls_saved_vs_naive', 0) or 0)
            system.cost_summary['candidate_eval_prompt_dedup_savings'] = int(system.cost_summary.get('candidate_eval_prompt_dedup_savings', 0) or 0) + int(context.candidate_eval_cache_stats.get('candidate_eval_prompt_dedup_savings', 0) or 0)
        context.summary = {'agent_id': context.agent_id, 'execution_session_id': system._current_execution_session_id(), 'update_attempt_id': context.update_attempt_id, **context.competence_log_fields, 'updated': bool(context.changed), 'candidate_count': len(context.candidate_pool), 'depth1_guard_rejection_count': sum((str(context.item.get('metrics', {}).get('rejection_reason', '')) == 'competence_depth1_guard' for context.item in context.evaluated)), 'accuracy_guard_rejection_count': sum((not bool(context.item.get('metrics', {}).get('accuracy_guard_passed', True)) for context.item in context.evaluated)), 'invalid_guard_rejection_count': sum((not bool(context.item.get('metrics', {}).get('invalid_guard_passed', True)) for context.item in context.evaluated)), 'dependence_guard_rejection_count': sum((str(context.item.get('metrics', {}).get('rejection_reason', '')) in {'pivotal_loss_guard', 'shared_error_creation_guard'} for context.item in context.evaluated)), 'pareto_not_retained_count': sum((not bool(context.item.get('pareto_selected', False)) for context.item in context.evaluated)), 'retained_candidate_count': len(context.selected), 'active_prompt_changed_count': int(context.changed), 'catastrophic_accuracy_guard_rejection_count': sum((not bool(context.item.get('metrics', {}).get('accuracy_guard_passed', True)) for context.item in context.evaluated)), 'soft_error_dependence_penalty_count': sum((float(context.item.get('metrics', {}).get('soft_error_dependence_penalty', 0.0) or 0.0) > 0.0 for context.item in context.evaluated)), 'soft_cycle_penalty_count': sum((float(context.item.get('metrics', {}).get('soft_cycle_penalty', 0.0) or 0.0) > 0.0 for context.item in context.evaluated)), 'soft_mechanism_shift_penalty_count': sum((float(context.item.get('metrics', {}).get('soft_mechanism_shift_penalty', 0.0) or 0.0) > 0.0 for context.item in context.evaluated)), 'exploration_candidate_count': sum((system._candidate_pool_source(context.item) == 'optimizer' and float(context.item.get('metrics', {}).get('mechanism_signature_distance', 0.0) or 0.0) > 0.0 for context.item in context.evaluated)), 'exploration_slot_occupancy_count': sum((str(context.item.get('beam_slot', '')) == ('mechanism_niche' if system._is_stable_qd_lineage() else 'explore') for context.item in context.selected)), 'exploration_to_active_conversion_count': int(bool(context.selected and context.selected[0].get('beam_slot') == 'explore' and context.changed)), 'generation_batches': context.generation_batches, 'baseline_homogeneous_case_count': len(context.baseline_cases), 'num_target_error_cases': int(context.num_target_error_cases), 'num_accuracy_repair_candidates': int(context.num_accuracy_repair_candidates), 'num_diversity_candidates': int(context.num_diversity_candidates), 'optimizer_fallback_mode': str(getattr(system.cfg, 'optimizer_fallback_mode', 'none')), 'optimizer_parent_concurrency': int(context.parent_concurrency), 'parent_sources': list(context.parent_sources), 'per_niche_parent_count': dict(context.agent.per_niche_parent_count), 'probation_parent_count': int(context.agent.probation_parent_count), 'probation_to_safe_conversion_count': int(getattr(system, 'probation_to_safe_conversion_count', 0)), 'candidate_starvation': bool(context.requirements.get('safe_non_incumbent_count', 1) == 0) if system._is_stable_qd_lineage() else False, 'mechanism_starvation': bool(context.requirements.get('safe_distinct_mechanism_count', 1) == 0) if system._is_stable_qd_lineage() else False, 'search_branch_starvation': bool(context.requirements.get('safe_non_incumbent_count', 1) == 0 and (not getattr(context.agent, 'probation_archive', []))) if system._is_stable_qd_lineage() else False, 'candidate_starvation_count': int(getattr(system, 'candidate_starvation_count', 0)), 'mechanism_starvation_count': int(getattr(system, 'mechanism_starvation_count', 0)), 'search_branch_starvation_count': int(getattr(system, 'search_branch_starvation_count', 0)), 'refill_requirements_unmet_count': int(getattr(system, 'refill_requirements_unmet_count', 0)), 'fallback_enabled': bool(context.fallback_enabled), 'optimizer_underfilled': bool(context.optimizer_underfilled), 'requested_optimizer_candidates': int(context.requested_optimizer_candidates), 'num_optimizer_candidates': int(context.num_optimizer_candidates), 'num_fallback_candidates': int(context.num_fallback_candidates), 'num_existing_beam_candidates': int(context.num_existing_beam_candidates), 'num_tcs_optimizer_candidates': int(context.num_tcs_optimizer_candidates), 'num_tcs_metadata_valid_candidates': int(context.num_tcs_metadata_valid_candidates), 'num_tcs_metadata_invalid_candidates': int(context.num_tcs_metadata_invalid_candidates), 'tcs_execution_complete': context.tcs_execution_complete, 'tcs_call_group_ids': sorted({str(context.c.get('tcs_call_group_id', '')) for context.c in context.candidate_pool if str(context.c.get('tcs_call_group_id', ''))}), 'top1_candidate_source': context.top1_candidate_source, 'top1_candidate_pool_source': context.top1_candidate_pool_source, 'active_prompt_changed': bool(context.changed), **context.pareto_summary, 'top1_pareto_rank': context.selected[0].get('pareto_rank') if system._uses_vote_pareto_selection() and context.selected else None, 'top1_vote_gain_rate': float(context.selected[0].get('metrics', {}).get('vote_gain_rate', 0.0)) if system._uses_vote_pareto_selection() and context.selected else None, 'top1_vote_loss_rate': float(context.selected[0].get('metrics', {}).get('vote_loss_rate', 0.0)) if system._uses_vote_pareto_selection() and context.selected else None, 'top1_vote_delta': float(context.selected[0].get('metrics', {}).get('vote_delta', 0.0)) if system._uses_vote_pareto_selection() and context.selected else None, **context.optimizer_generation_summary, **system._student_failure_log_fields(context.optimizer_generation_summary), 'top_reward': float(context.agent.prompt_beam[0].get('score', 0.0) or 0.0), 'top_metrics': context.agent.prompt_beam[0].get('metrics', {}), **context.candidate_eval_cache_stats, 'execution_session_id': system._current_execution_session_id(), 'update_attempt_id': context.update_attempt_id}
        if system._is_state_conditioned_method():
            coverage_candidates = [
                item for item in context.evaluated
                if str(item.get('optimization_route', '')) == 'coverage_repair'
            ]
            for question_hash in sorted({
                question_hash
                for item in coverage_candidates
                for question_hash in item.get('generation_question_hashes', [])
            }):
                related = [
                    item for item in coverage_candidates
                    if question_hash in item.get('generation_question_hashes', [])
                ]
                successful = any(
                    bool(item.get('metrics', {}).get('state_quality_guard_passed', False))
                    and (
                        int(item.get('metrics', {}).get('c0_to_c1_count', 0) or 0) > 0
                        or int(item.get('metrics', {}).get('c1_to_c2_count', 0) or 0) > 0
                    )
                    for item in related
                )
                if successful:
                    system.coverage_resolved_by[question_hash] = int(context.agent_id)
                    system.coverage_resolved_epoch[question_hash] = int(context.epoch_id)
                    continue
                failed = list(system.coverage_failed_prompt_hashes.get(question_hash, []))
                before = len(set(failed))
                failed.extend(str(item.get('prompt_hash', '')) for item in related if item.get('prompt_hash'))
                system.coverage_failed_prompt_hashes[question_hash] = sorted(set(failed))
                if before < 2 <= len(system.coverage_failed_prompt_hashes[question_hash]):
                    system.coverage_rotation_count += 1
            exploration_descendants = [
                item for item in context.evaluated
                if bool(item.get('parent_was_exploration', False))
            ]
            safe_descendants = [
                item for item in exploration_descendants
                if bool(item.get('metrics', {}).get('state_quality_guard_passed', False))
            ]
            archive_hashes = {
                str(item.get('prompt_hash', ''))
                for item in getattr(context.agent, 'safe_qd_archive', [])
            }
            archived_descendants = [
                item for item in safe_descendants
                if str(item.get('prompt_hash', '')) in archive_hashes
            ]
            system.exploration_descendant_count += len(exploration_descendants)
            system.exploration_descendant_safe_count += len(safe_descendants)
            system.exploration_descendant_archive_count += len(archived_descendants)
            for field in (
                'vote_gain_count', 'c0_to_c1_count', 'c1_to_c2_count', 'c2_to_c3_count'
            ):
                system_field = f'exploration_descendant_{field}'
                setattr(system, system_field, int(getattr(system, system_field, 0)) + sum(
                    int(int(item.get('metrics', {}).get(field, 0) or 0) > 0)
                    for item in safe_descendants
                ))
            has_state_gain = any(
                any(int(item.get('metrics', {}).get(field, 0) or 0) > 0 for field in (
                    'vote_gain_count', 'c0_to_c1_count', 'c1_to_c2_count', 'c2_to_c3_count'
                ))
                for item in safe_descendants
            )
            agent_key = str(context.agent_id)
            system.state_no_gain_updates_per_agent[agent_key] = (
                0 if has_state_gain else int(
                    system.state_no_gain_updates_per_agent.get(agent_key, 0) or 0
                ) + 1
            )
            system.exploration_descendant_state_gain_count += sum(
                int(any(
                    int(item.get('metrics', {}).get(field, 0) or 0) > 0
                    for field in ('c0_to_c1_count', 'c1_to_c2_count', 'c2_to_c3_count')
                ))
                for item in safe_descendants
            )
            context.summary.update({
                'parent_selection_diagnostics': dict(
                    getattr(context, 'parent_selection_diagnostics', {}) or {}
                ),
                'state_no_gain_updates_per_agent': dict(system.state_no_gain_updates_per_agent),
                'exploration_parent_use_count': int(system.exploration_parent_use_count),
                'exploration_descendant_count': int(system.exploration_descendant_count),
                'exploration_descendant_safe_count': int(system.exploration_descendant_safe_count),
                'exploration_descendant_archive_count': int(system.exploration_descendant_archive_count),
            })
        if system._is_state_conditioned_method() and context.changed and context.selected:
            selected_metrics = context.selected[0].get('metrics', {})
            agent_key = str(context.agent_id)
            system.c0_rescue_count_per_agent[agent_key] = int(
                system.c0_rescue_count_per_agent.get(agent_key, 0) or 0
            ) + int(selected_metrics.get('c0_to_c1_count', 0) or 0)
            system.c1_deepening_count_per_agent[agent_key] = int(
                system.c1_deepening_count_per_agent.get(agent_key, 0) or 0
            ) + int(selected_metrics.get('c1_to_c2_count', 0) or 0)
            context.summary.update({
                'coverage_case_assignment_per_agent': dict(system.coverage_case_assignment_per_agent),
                'c0_rescue_count_per_agent': dict(system.c0_rescue_count_per_agent),
                'c1_deepening_count_per_agent': dict(system.c1_deepening_count_per_agent),
            })
        if system._is_state_conditioned_method():
            context.summary.update({
                'state_archive_slots': [
                    str(item.get('state_archive_slot', ''))
                    for item in getattr(context.agent, 'safe_qd_archive', [])
                ],
                'state_archive_prompt_hashes': [
                    str(item.get('prompt_hash', ''))
                    for item in getattr(context.agent, 'safe_qd_archive', [])
                ],
            })
        system.depth1_guard_rejection_count = int(getattr(system, 'depth1_guard_rejection_count', 0)) + int(context.summary['depth1_guard_rejection_count'])
        if system._is_stable_qd_lineage() or system._is_rollout_qd_method() or system._is_state_conditioned_method():
            system.total_agent_update_count += 1
        if system._is_stable_qd_lineage():
            system.task_repair_niche_occupancy_count += int(context.pareto_summary.get('task_repair_niche_occupancy', 0) or 0)
            system.mechanism_niche_occupancy_count += int(context.pareto_summary.get('mechanism_niche_occupancy', 0) or 0)
        for context.field in ('catastrophic_accuracy_guard_rejection_count', 'soft_error_dependence_penalty_count', 'soft_cycle_penalty_count', 'soft_mechanism_shift_penalty_count', 'exploration_candidate_count', 'exploration_slot_occupancy_count', 'exploration_to_active_conversion_count'):
            setattr(system, context.field, int(getattr(system, context.field, 0)) + int(context.summary.get(context.field, 0) or 0))
        if system._is_v82_hybrid():
            system.mechanism_signature_history.append({'epoch': int(context.epoch_id), 'step': int(context.step_id), 'agent_id': int(context.agent_id), 'retained': [list(context.item.get('metrics', {}).get('mechanism_signature', [])) for context.item in context.selected]})
            system.beam_slot_state[str(context.agent_id)] = [str(context.item.get('beam_slot', '')) for context.item in context.selected]
            system.exploration_slot_candidates = [{'agent_id': int(context.agent_id), 'candidate_id': str(context.item.get('candidate_id', '')), 'prompt': str(context.item.get('prompt', ''))} for context.item in context.selected if str(context.item.get('beam_slot', '')) == 'explore']
        system.update_logs.append({**system._base_log_fields(), 'event': 'beam_update_summary', 'epoch': context.epoch_id, 'step': context.step_id, 'agent_id': context.agent_id, 'execution_session_id': system._current_execution_session_id(), 'update_attempt_id': context.update_attempt_id, **context.competence_log_fields, 'search_mode': 'evolutionary_beam', 'beam_size': context.beam_size, 'active_prompt_changed': bool(context.changed), 'top1_candidate_source': context.top1_candidate_source, 'top1_candidate_pool_source': context.top1_candidate_pool_source, 'candidate_count': len(context.candidate_pool), 'depth1_guard_rejection_count': context.summary['depth1_guard_rejection_count'], 'accuracy_guard_rejection_count': context.summary['accuracy_guard_rejection_count'], 'invalid_guard_rejection_count': context.summary['invalid_guard_rejection_count'], 'dependence_guard_rejection_count': context.summary['dependence_guard_rejection_count'], 'pareto_not_retained_count': context.summary['pareto_not_retained_count'], 'retained_candidate_count': context.summary['retained_candidate_count'], 'active_prompt_changed_count': context.summary['active_prompt_changed_count'], 'generation_batches': context.generation_batches, 'general_error_case_count': sum((len(context.batch.get('cases', [])) for context.batch in context.generation_batches if str(context.batch.get('batch_type', '')) == 'general_error')), 'c1_creation_case_count': sum((sum((int(context.case.get('baseline_correct_count', -1)) == 0 for context.case in context.batch.get('cases', []))) for context.batch in context.generation_batches if str(context.batch.get('batch_type', '')) == 'c1_c2_creation')), 'c2_creation_case_count': sum((sum((int(context.case.get('baseline_correct_count', -1)) == 1 for context.case in context.batch.get('cases', []))) for context.batch in context.generation_batches if str(context.batch.get('batch_type', '')) == 'c1_c2_creation')), 'boundary_case_count': sum((len(context.batch.get('cases', [])) for context.batch in context.generation_batches if str(context.batch.get('batch_type', '')) == 'actual_plurality_boundary')), 'residual_case_count': sum((len(context.batch.get('cases', [])) for context.batch in context.generation_batches if str(context.batch.get('batch_type', '')) == 'residual_shared_error')), 'catastrophic_accuracy_guard_rejection_count': context.summary['catastrophic_accuracy_guard_rejection_count'], 'soft_error_dependence_penalty_count': context.summary['soft_error_dependence_penalty_count'], 'soft_cycle_penalty_count': context.summary['soft_cycle_penalty_count'], 'soft_mechanism_shift_penalty_count': context.summary['soft_mechanism_shift_penalty_count'], 'exploration_candidate_count': context.summary['exploration_candidate_count'], 'exploration_slot_occupancy_count': context.summary['exploration_slot_occupancy_count'], 'exploration_to_active_conversion_count': context.summary['exploration_to_active_conversion_count'], 'optimizer_fallback_mode': str(getattr(system.cfg, 'optimizer_fallback_mode', 'none')), 'optimizer_parent_concurrency': int(context.parent_concurrency), 'parent_sources': list(context.parent_sources), 'per_niche_parent_count': dict(context.agent.per_niche_parent_count), 'probation_parent_count': int(context.agent.probation_parent_count), 'probation_to_safe_conversion_count': int(getattr(system, 'probation_to_safe_conversion_count', 0)), 'fallback_enabled': bool(context.fallback_enabled), 'optimizer_underfilled': bool(context.optimizer_underfilled), 'requested_optimizer_candidates': int(context.requested_optimizer_candidates), 'num_optimizer_candidates': int(context.num_optimizer_candidates), 'num_fallback_candidates': int(context.num_fallback_candidates), 'num_existing_beam_candidates': int(context.num_existing_beam_candidates), 'num_teacher_calls': int(context.optimizer_generation_summary.get('num_teacher_calls', 0) or 0), 'num_critic_calls': int(context.optimizer_generation_summary.get('num_critic_calls', 0) or 0), 'num_teacher_rewrite_calls': int(context.optimizer_generation_summary.get('num_teacher_rewrite_calls', 0) or 0), 'num_student_calls': int(context.optimizer_generation_summary.get('num_student_calls', 0) or 0), 'num_student_retry_calls': int(context.optimizer_generation_summary.get('num_student_retry_calls', 0) or 0), 'num_student_repair_calls': int(context.optimizer_generation_summary.get('num_student_repair_calls', 0) or 0), 'num_tcs_optimizer_candidates': int(context.num_tcs_optimizer_candidates), 'num_tcs_metadata_valid_candidates': int(context.num_tcs_metadata_valid_candidates), 'num_tcs_metadata_invalid_candidates': int(context.num_tcs_metadata_invalid_candidates), 'tcs_execution_complete': context.tcs_execution_complete, 'tcs_call_group_ids': sorted({str(context.c.get('tcs_call_group_id', '')) for context.c in context.candidate_pool if str(context.c.get('tcs_call_group_id', ''))}), 'candidate_selection_mode': str(getattr(system.cfg, 'candidate_selection_mode', 'scalar_reward')), **context.candidate_eval_cache_stats, **context.pareto_summary, 'candidate_starvation': bool(context.requirements.get('safe_non_incumbent_count', 1) == 0) if system._is_stable_qd_lineage() else False, 'mechanism_starvation': bool(context.requirements.get('safe_distinct_mechanism_count', 1) == 0) if system._is_stable_qd_lineage() else False, 'search_branch_starvation': bool(context.requirements.get('safe_non_incumbent_count', 1) == 0 and (not getattr(context.agent, 'probation_archive', []))) if system._is_stable_qd_lineage() else False, 'candidate_starvation_count': int(getattr(system, 'candidate_starvation_count', 0)), 'mechanism_starvation_count': int(getattr(system, 'mechanism_starvation_count', 0)), 'search_branch_starvation_count': int(getattr(system, 'search_branch_starvation_count', 0)), 'refill_requirements_unmet_count': int(getattr(system, 'refill_requirements_unmet_count', 0)), 'top1_pareto_rank': context.selected[0].get('pareto_rank') if system._uses_vote_pareto_selection() and context.selected else None, 'top1_vote_gain_rate': float(context.selected[0].get('metrics', {}).get('vote_gain_rate', 0.0)) if system._uses_vote_pareto_selection() and context.selected else None, 'top1_vote_loss_rate': float(context.selected[0].get('metrics', {}).get('vote_loss_rate', 0.0)) if system._uses_vote_pareto_selection() and context.selected else None, 'top1_vote_delta': float(context.selected[0].get('metrics', {}).get('vote_delta', 0.0)) if system._uses_vote_pareto_selection() and context.selected else None, **context.optimizer_generation_summary, **system._student_failure_log_fields(context.optimizer_generation_summary), 'execution_session_id': system._current_execution_session_id(), 'update_attempt_id': context.update_attempt_id})
        context.agent.last_update_record = context.summary

class PromptUpdateMixin:

    async def _run_v9_sequential_stage_b(self, context) -> Tuple[bool, Dict[str, Any]]:
        probe_data = list(getattr(self, 'fixed_acceptance_probe_data', []) or [])
        if not probe_data:
            raise RuntimeError('V9 Stage B requires a non-empty fixed acceptance probe')
        if not getattr(self, 'current_sequential_profiles', []):
            await self.refresh_state_conditioned_fixed_probe_snapshot(probe_data, epoch=context.epoch_id)
        active_profiles = [dict(profile) for profile in self.current_sequential_profiles]
        previous_active_profile = dict(active_profiles[context.agent_id])
        question_hashes = [self._hash(str(example.get('question', ''))) for example in probe_data]
        gold_answers = [
            self.task_spec.parse_gold(example.get('answer'), str(example.get('question', '')))
            for example in probe_data
        ]
        active_team_metrics = sequential_team_metrics(
            active_profiles, gold_answers, question_hashes, context.agent_id, self.cfg,
            vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
        )
        active_hash = self._normalized_prompt_hash(context.agent.current_prompt)
        incumbent = self._make_beam_item(context.agent.current_prompt, None, {}, None, 0)
        incumbent.update({
            'candidate_id': f'incumbent_{active_hash}',
            'prompt_hash': active_hash,
            'source': 'incumbent',
            'candidate_pool_source': 'incumbent',
        })
        stage_a_ranked = sorted(
            context.evaluated,
            key=lambda item: (
                float(item.get('metrics', {}).get('candidate_target_accuracy', 0.0) or 0.0),
                float(item.get('metrics', {}).get('candidate_team_accuracy', 0.0) or 0.0),
                -float(item.get('metrics', {}).get('candidate_invalid_rate', 1.0) or 1.0),
                str(item.get('prompt_hash', '')),
            ),
            reverse=True,
        )[:max(1, int(self.cfg.state_full_probe_acceptance_candidates))]
        stage_b_pool = [incumbent, *stage_a_ranked, *list(context.agent.prompt_memory)]
        unique = {}
        for item in stage_b_pool:
            row = dict(item)
            prompt = str(row.get('prompt', context.agent.current_prompt))
            row['prompt'] = prompt
            row['prompt_hash'] = self._normalized_prompt_hash(prompt)
            unique.setdefault(row['prompt_hash'], row)
        before_missing = int(getattr(self, 'full_probe_missing_pair_evaluation_count', 0))
        before_cache_hits = int(getattr(self, 'full_probe_cache_hit_count', 0))
        stage_b_concurrency = max(1, min(
            int(getattr(self.cfg, 'candidate_eval_concurrency', 1) or 1),
            len(unique),
        ))
        stage_b_semaphore = asyncio.Semaphore(stage_b_concurrency)

        async def evaluate_full_probe(row):
            async with stage_b_semaphore:
                profile = await self._evaluate_prompt_on_stable_probe(
                    context.agent_id, str(row['prompt']), probe_data
                )
            team_profiles = [dict(value) for value in active_profiles]
            team_profiles[context.agent_id] = dict(profile)
            team_metrics = sequential_team_metrics(
                team_profiles, gold_answers, question_hashes, context.agent_id, self.cfg,
                vote_fn=plurality_vote_with_diagnostics, match_fn=self.task_spec.match_answer,
            )
            reward = state_vote_reward(
                active_team_metrics['correctness_rows'], team_metrics['correctness_rows'],
                active_team_metrics['vote_correct_vector'], team_metrics['vote_correct_vector'],
                self.cfg,
            )
            paired_safe_trace = paired_safe_trace_diversity_c4c5(
                active_profiles, team_profiles, context.agent_id, self.cfg
            )
            metrics = {
                **dict(row.get('metrics', {}) or {}),
                **team_metrics,
                **reward,
                **paired_safe_trace,
            }
            initial_metrics = (
                self.initial_sequential_team_metrics[context.agent_id]
                if context.agent_id < len(self.initial_sequential_team_metrics)
                else active_team_metrics
            )
            metrics.update(full_probe_constraints(metrics, active_team_metrics, initial_metrics, self.cfg))
            metrics['rollout_profile'] = dict(profile)
            metrics['outcome_signature_hash'] = outcome_signature(
                profile,
                self.cfg.state_outcome_signature_version,
                str(getattr(self, 'current_fixed_probe_hash', '')),
                question_hashes,
            )
            metrics['safe_trace_signature_hash'] = safe_trace_signature(
                team_profiles,
                context.agent_id,
                self.cfg.state_safe_trace_signature_version,
                str(getattr(self, 'current_fixed_probe_hash', '')),
                question_hashes,
            )
            row['metrics'] = metrics
            row['outcome_signature_hash'] = metrics['outcome_signature_hash']
            row['safe_trace_signature_hash'] = metrics['safe_trace_signature_hash']
            row['reward'] = float(metrics['state_reward_total'])
            row['stage_b_full_probe_evaluated'] = True
            return row

        profiled = await asyncio.gather(*[
            evaluate_full_probe(dict(row)) for row in unique.values()
        ])
        if not isinstance(getattr(self, 'state_search_diagnostics', None), dict):
            self.state_search_diagnostics = {}
        diagnostics = self.state_search_diagnostics
        for key in (
            'candidate_diversity_constraint_evaluated_count',
            'candidate_diversity_constraint_pass_count',
            'candidate_diversity_constraint_rejection_count',
            'candidate_correct_set_constraint_rejection_count',
            'candidate_safe_trace_constraint_rejection_count',
            'correct_set_constraint_binding_count', 'safe_trace_constraint_binding_count',
            'paired_safe_trace_available_count', 'paired_safe_trace_pair_count',
            'safe_diversity_parent_use_count', 'accepted_accuracy_regression_count',
        ):
            diagnostics.setdefault(key, 0)
        diagnostics['evaluated_candidate_count'] = int(
            diagnostics.get('evaluated_candidate_count', 0) or 0
        ) + len(profiled)
        for item in profiled:
            metrics = item.get('metrics', {})
            evaluated_diversity = bool(metrics.get('diversity_constraint_evaluated', False))
            correct_rejected = bool(metrics.get('correct_set_constraint_rejected', False))
            safe_rejected = bool(metrics.get('safe_trace_constraint_rejected', False))
            diagnostics['candidate_diversity_constraint_evaluated_count'] = int(
                diagnostics.get('candidate_diversity_constraint_evaluated_count', 0) or 0
            ) + int(evaluated_diversity)
            diagnostics['candidate_diversity_constraint_pass_count'] = int(
                diagnostics.get('candidate_diversity_constraint_pass_count', 0) or 0
            ) + int(evaluated_diversity and not correct_rejected and not safe_rejected)
            diagnostics['candidate_diversity_constraint_rejection_count'] = int(
                diagnostics.get('candidate_diversity_constraint_rejection_count', 0) or 0
            ) + int(correct_rejected or safe_rejected)
            diagnostics['candidate_correct_set_constraint_rejection_count'] = int(
                diagnostics.get('candidate_correct_set_constraint_rejection_count', 0) or 0
            ) + int(correct_rejected)
            diagnostics['candidate_safe_trace_constraint_rejection_count'] = int(
                diagnostics.get('candidate_safe_trace_constraint_rejection_count', 0) or 0
            ) + int(safe_rejected)
            diagnostics['correct_set_constraint_binding_count'] = int(
                diagnostics.get('correct_set_constraint_binding_count', 0) or 0
            ) + int(bool(metrics.get('correct_set_constraint_binding', False)))
            diagnostics['safe_trace_constraint_binding_count'] = int(
                diagnostics.get('safe_trace_constraint_binding_count', 0) or 0
            ) + int(bool(metrics.get('safe_trace_constraint_binding', False)))
            diagnostics['paired_safe_trace_available_count'] = int(
                diagnostics.get('paired_safe_trace_available_count', 0) or 0
            ) + int(bool(metrics.get('paired_safe_trace_constraint_available', False)))
            diagnostics['paired_safe_trace_pair_count'] = int(
                diagnostics.get('paired_safe_trace_pair_count', 0) or 0
            ) + int(metrics.get('paired_safe_trace_pair_count', 0) or 0)
        incumbent = next(item for item in profiled if item['prompt_hash'] == active_hash)
        feasible = [
            item for item in profiled
            if bool(item.get('metrics', {}).get('sequential_constraints_passed', False))
        ]
        recent = list(getattr(self, 'sequential_recent_accepted_prompt_hashes', []) or [])
        cooldown = max(0, int(self.cfg.state_prompt_reaccept_cooldown_updates))
        for item in feasible:
            if item['prompt_hash'] != active_hash and item['prompt_hash'] in recent[-cooldown:]:
                item['metrics']['sequential_constraints_passed'] = False
                item['metrics']['rejection_reason'] = 'prompt_reaccept_cooldown'
        feasible = [item for item in feasible if item['metrics']['sequential_constraints_passed']]
        selected = max(feasible or [incumbent], key=lambda item: accuracy_first_key(item, self.cfg))
        changed = (
            selected['prompt_hash'] != active_hash
            and candidate_strictly_beats_incumbent(selected, incumbent, self.cfg)
        )
        if not changed:
            selected = incumbent
        previous_active_item = dict(incumbent)
        previous_active_item['metrics'] = dict(incumbent.get('metrics', {}) or {})
        previous_active_item['metrics']['rollout_profile'] = previous_active_profile
        if changed:
            if int(selected['metrics'].get('candidate_target_correct_count', 0) or 0) < int(
                incumbent['metrics'].get('candidate_target_correct_count', 0) or 0
            ):
                diagnostics['accepted_accuracy_regression_count'] = int(
                    diagnostics.get('accepted_accuracy_regression_count', 0) or 0
                ) + 1
            context.agent.current_prompt = str(selected['prompt'])
            context.agent.history.append(context.agent.current_prompt)
            context.agent.accept_count += 1
            selected['accepted_update_index'] = len(self.sequential_update_history)
            selected['active_selection_count'] = int(selected.get('active_selection_count', 0) or 0) + 1
            self.sequential_recent_accepted_prompt_hashes.append(selected['prompt_hash'])
            self.sequential_recent_accepted_prompt_hashes = self.sequential_recent_accepted_prompt_hashes[
                -max(1, int(self.cfg.state_update_cycle_window)):
            ]
            self.current_sequential_profiles[context.agent_id] = dict(
                selected['metrics']['rollout_profile']
            )
            await self.refresh_state_conditioned_fixed_probe_snapshot(
                probe_data, epoch=context.epoch_id
            )
        else:
            context.agent.reject_count += 1
        memory_items = [item for item in profiled if item['metrics']['sequential_constraints_passed']]
        context.agent.prompt_memory, memory_diagnostics = rebuild_prompt_memory(
            memory_items,
            self._normalized_prompt_hash(context.agent.current_prompt),
            int(self.cfg.state_prompt_memory_capacity),
            config=self.cfg,
            previous_active_item=previous_active_item if changed else None,
            return_diagnostics=True,
        )
        context.agent.prompt_memory_diagnostics = dict(memory_diagnostics)
        if changed:
            for peer_id, peer in enumerate(self.agents):
                if peer_id == context.agent_id or not peer.prompt_memory:
                    continue
                peer_active = sequential_team_metrics(
                    self.current_sequential_profiles, gold_answers, question_hashes,
                    peer_id, self.cfg, vote_fn=plurality_vote_with_diagnostics,
                    match_fn=self.task_spec.match_answer,
                )
                refreshed_memory = []
                for memory_item in peer.prompt_memory:
                    memory_profile = dict(memory_item.get('metrics', {}).get('rollout_profile', {}) or {})
                    if not memory_profile:
                        continue
                    peer_profiles = [dict(value) for value in self.current_sequential_profiles]
                    peer_profiles[peer_id] = memory_profile
                    peer_metrics = sequential_team_metrics(
                        peer_profiles, gold_answers, question_hashes, peer_id, self.cfg,
                        vote_fn=plurality_vote_with_diagnostics,
                        match_fn=self.task_spec.match_answer,
                    )
                    peer_reward = state_vote_reward(
                        peer_active['correctness_rows'], peer_metrics['correctness_rows'],
                        peer_active['vote_correct_vector'], peer_metrics['vote_correct_vector'],
                        self.cfg,
                    )
                    merged = {**dict(memory_item.get('metrics', {}) or {}), **peer_metrics, **peer_reward}
                    merged.update(paired_safe_trace_diversity_c4c5(
                        self.current_sequential_profiles, peer_profiles, peer_id, self.cfg
                    ))
                    peer_initial = (
                        self.initial_sequential_team_metrics[peer_id]
                        if peer_id < len(self.initial_sequential_team_metrics) else peer_active
                    )
                    merged.update(full_probe_constraints(merged, peer_active, peer_initial, self.cfg))
                    row = dict(memory_item)
                    row['metrics'] = merged
                    row['safe_trace_signature_hash'] = safe_trace_signature(
                        peer_profiles,
                        peer_id,
                        self.cfg.state_safe_trace_signature_version,
                        str(getattr(self, 'current_fixed_probe_hash', '')),
                        question_hashes,
                    )
                    row['metrics']['safe_trace_signature_hash'] = row['safe_trace_signature_hash']
                    refreshed_memory.append(row)
                if refreshed_memory:
                    peer.prompt_memory, peer_memory_diagnostics = rebuild_prompt_memory(
                        refreshed_memory,
                        self._normalized_prompt_hash(peer.current_prompt),
                        int(self.cfg.state_prompt_memory_capacity),
                        config=self.cfg,
                        return_diagnostics=True,
                    )
                    peer.prompt_memory_diagnostics = dict(peer_memory_diagnostics)
                    peer.prompt_beam = [dict(item) for item in peer.prompt_memory]
        stage_a_candidate_count = len(context.evaluated)
        context.agent.prompt_beam = [dict(item) for item in context.agent.prompt_memory]
        context.selected = [selected]
        context.changed = changed
        context.evaluated = profiled
        record = {
            **self._base_log_fields(),
            'event': 'sequential_single_agent_update',
            'epoch': int(context.epoch_id),
            'step': int(context.step_id),
            'epoch_agent_order': epoch_agent_order(max(0, int(context.epoch_id) - 1), len(self.agents)),
            'current_agent_order_index': int(
                self.sequential_agent_order_index_by_epoch.get(str(context.epoch_id), 1)
            ) - 1,
            'target_agent_id': int(context.agent_id),
            'target_selection_reason': 'deterministic_rotating_order',
            'active_prompt_changed': bool(changed),
            'active_prompt_changed_count': int(changed),
            'previous_active_prompt_hash': active_hash,
            'selected_prompt_hash': str(selected['prompt_hash']),
            'rollback_prompt_hash': str(memory_diagnostics.get('rollback_prompt_hash', '')),
            'selected_accuracy_first_key': list(accuracy_first_key(selected, self.cfg)),
            'stage_a_candidate_count': stage_a_candidate_count,
            'stage_b_candidate_count': len(profiled),
            'stage_b_full_probe_solver_calls': int(
                getattr(self, 'full_probe_missing_pair_evaluation_count', 0)
            ) - before_missing,
            'stage_b_cache_hits': int(getattr(self, 'full_probe_cache_hit_count', 0)) - before_cache_hits,
            'prompt_memory_size': len(context.agent.prompt_memory),
            'memory_capacity': int(self.cfg.state_prompt_memory_capacity),
            'memory_occupancy': len(context.agent.prompt_memory),
            'prompt_memory_slots': [item.get('prompt_memory_slot', '') for item in context.agent.prompt_memory],
            'memory_underfilled': bool(memory_diagnostics.get('memory_underfilled', False)),
            'memory_underfilled_reason': str(memory_diagnostics.get('memory_underfilled_reason', '')),
            'memory_slot_candidate_count': dict(memory_diagnostics.get('slot_candidate_count', {})),
            'memory_slot_duplicate_skip_count': int(memory_diagnostics.get('slot_duplicate_skip_count', 0)),
            'memory_quality_fill_count': int(memory_diagnostics.get('quality_fill_count', 0)),
            'joint_team_combination_count': 0,
            'equal_vote_weighting': True,
            'raw_vote_accuracy_delta': float(selected['metrics'].get('vote_accuracy_delta', 0.0) or 0.0),
            'vote_gain_count': int(selected['metrics'].get('vote_gain_count', 0) or 0),
            'vote_loss_count': int(selected['metrics'].get('vote_loss_count', 0) or 0),
            'diversity_constraints_enabled': bool(self.cfg.state_diversity_constraints_enabled),
            'optimizer_generation_diagnostics': dict(
                getattr(context, 'optimizer_generation_summary', {}) or {}
            ),
            'selected_metrics': {
                key: value for key, value in selected['metrics'].items()
                if key not in {'rollout_profile', 'correctness_rows', 'vote_correct_vector'}
            },
            'candidate_decisions': [
                {
                    'candidate_id': str(item.get('candidate_id', '')),
                    'prompt_hash': str(item.get('prompt_hash', '')),
                    'selected': str(item.get('prompt_hash', '')) == str(selected.get('prompt_hash', '')),
                    'candidate_selected': str(item.get('prompt_hash', '')) == str(selected.get('prompt_hash', '')),
                    'candidate_accepted': bool(
                        changed
                        and str(item.get('prompt_hash', '')) == str(selected.get('prompt_hash', ''))
                    ),
                    'accuracy_first_key': list(accuracy_first_key(item, self.cfg)),
                    **{
                        key: item.get('metrics', {}).get(key)
                        for key in (
                            'candidate_target_correct_count', 'candidate_target_accuracy',
                            'state_reward_total', 'state_reward_distribution_component',
                            'state_reward_vote_component', 'state_reward_bottom2_component',
                            'active_target_correct_count', 'initial_target_correct_count',
                            'candidate_invalid_count', 'local_accuracy_loss_count',
                            'global_accuracy_loss_count', 'correct_set_diversity_mean',
                            'correct_set_diversity_min', 'safe_trace_diversity_c4c5',
                            'safe_trace_pair_count', 'safe_trace_constraint_available',
                            'active_paired_safe_trace_diversity',
                            'candidate_paired_safe_trace_diversity',
                            'paired_safe_trace_delta', 'paired_safe_trace_pair_count',
                            'paired_safe_trace_constraint_available',
                            'accuracy_constraint_passed', 'invalid_constraint_passed',
                            'correct_set_diversity_constraint_passed',
                            'safe_trace_constraint_passed',
                            'diversity_constraint_evaluated',
                            'correct_set_constraint_rejected',
                            'safe_trace_constraint_rejected',
                            'correct_set_constraint_binding',
                            'safe_trace_constraint_binding',
                            'vote_accuracy_delta', 'vote_gain_count', 'vote_loss_count',
                            'diversity_constraint_slack', 'sequential_constraints_passed',
                            'candidate_feasible',
                            'rejection_reason',
                        )
                    },
                }
                for item in profiled
            ],
        }
        if not hasattr(self, 'cost_summary'):
            self.cost_summary = self._empty_cost_summary()
        self.cost_summary['stage_b_full_probe_solver_calls'] = int(
            self.cost_summary.get('stage_b_full_probe_solver_calls', 0) or 0
        ) + int(record['stage_b_full_probe_solver_calls'])
        self.cost_summary['stage_b_cache_hits'] = int(
            self.cost_summary.get('stage_b_cache_hits', 0) or 0
        ) + int(record['stage_b_cache_hits'])
        stage_b_total = (
            int(self.cost_summary['stage_b_full_probe_solver_calls'])
            + int(self.cost_summary['stage_b_cache_hits'])
        )
        self.cost_summary['stage_b_cache_hit_rate'] = (
            float(self.cost_summary['stage_b_cache_hits']) / stage_b_total
            if stage_b_total else 0.0
        )
        self.sequential_update_history.append(dict(record))
        self._flush_jsonl('sequential_update_history.jsonl', [record])
        self.update_logs.append(dict(record))
        self._append_prompt_history_event(
            context.agent_id, context.epoch_id, context.step_id,
            'sequential_accept' if changed else 'sequential_keep', changed,
        )
        self.flush_prompt_history()
        requested = sum(
            int(row.get('optimizer_final_candidate_count', 0) or 0)
            for row in getattr(context, 'optimizer_generation_records', [])
        )
        summary = {
            **dict(getattr(context, 'optimizer_generation_summary', {}) or {}),
            'agent_id': context.agent_id,
            'updated': bool(changed),
            'active_prompt_changed': bool(changed),
            'active_prompt_changed_count': int(changed),
            'candidate_count': len(context.candidate_pool),
            'requested_optimizer_candidates': int(
                len(context.parent_jobs) * int(context.requested)
            ),
            'num_optimizer_candidates': int(sum(
                self._candidate_pool_source(item) == 'optimizer' for item in context.candidate_pool
            )),
            'num_fallback_candidates': int(sum(
                self._candidate_pool_source(item) == 'fallback' for item in context.candidate_pool
            )),
            'num_existing_beam_candidates': int(sum(
                self._candidate_pool_source(item) == 'existing_beam' for item in context.candidate_pool
            )),
            'optimizer_underfilled': bool(requested < len(context.parent_jobs) * int(context.requested)),
            'top_metrics': selected['metrics'],
            'prompt_memory_size': len(context.agent.prompt_memory),
            'stage_b_full_probe_solver_calls': record['stage_b_full_probe_solver_calls'],
            'stage_b_cache_hits': record['stage_b_cache_hits'],
            'memory_occupancy': record['memory_occupancy'],
            'memory_underfilled': record['memory_underfilled'],
            'rollback_prompt_hash': record['rollback_prompt_hash'],
            'joint_team_combination_count': 0,
            'epoch_agent_order': record['epoch_agent_order'],
            'current_agent_order_index': record['current_agent_order_index'],
            'target_selection_reason': record['target_selection_reason'],
        }
        self.total_agent_update_count = int(getattr(self, 'total_agent_update_count', 0)) + 1
        context.agent.last_update_record = dict(summary)
        return changed, summary

    async def update_prompt_with_beam(self, agent_id: int, overlap_diagnosis: Dict[str, Any], eval_batch: List[Dict[str, str]], step_id: int, epoch_id: int) -> Tuple[bool, Dict[str, Any]]:
        context = PromptUpdateContext(agent_id=agent_id, overlap_diagnosis=overlap_diagnosis, eval_batch=eval_batch, step_id=step_id, epoch_id=epoch_id)
        await CandidateGenerationStage.run(self, context)
        await CheapPrescreenStage.run(self, context)
        await CandidateEvaluationStage.run(self, context)
        if self._is_state_conditioned_method():
            return await self._run_v9_sequential_stage_b(context)
        await CandidateClassificationAndRefillStage.run(self, context)
        await ArchiveSelectionStage.run(self, context)
        await CandidateEventStage.run(self, context)
        await UpdateSummaryStage.run(self, context)
        return (bool(context.changed), context.summary)
