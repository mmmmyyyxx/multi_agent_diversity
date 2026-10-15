"""Full wrong-set discovery and production representative sampling, isolated adapter."""
from copy import deepcopy
from pathlib import Path

from ...benchmarks.data_freeze import digest, file_hash
from ...search.population_evidence import sample_pattern_support
from ...search.schemas import EvidenceItem, SearchContractError
from ..single_prompt import raw_error, batch_ids
from .formal_adapters import discover
from .interfaces import PreparedFeedback
from .pattern import LargestPatternModule, project


def sampling_context(context):
    return dict(cell_id=context.cell_id, prompt_sha256=context.prompt.prompt_hash,
        task_metadata={r['stable_example_id']:{k:str(r['content'][k]) for k in ('type','level')
            if k in r['content']} for r in context.rows})


def project_full(feedback, seed, ordinal, sampling):
    """Only the selected Pattern and up to three representative evidence rows leave."""
    projected, focus = project(feedback, seed, ordinal)
    errors = {e['example_id']:e for e in feedback['errors']}
    if (set(sampling['task_metadata']) != set(errors)
            or any(e['correct'] is not False for e in errors.values())):
        raise SearchContractError('ABLATION_FULL_DIAGNOSIS_SAMPLING_SOURCE')
    state_id = digest(['ABLATION_SINGLE_PROMPT_PATTERN_ROOT_V1',sampling['prompt_sha256']])
    rows = tuple(EvidenceItem(i,'optimize',frozenset({'pattern'}),dict(
        target_member_correct=False,task_metadata=sampling['task_metadata'][i])) for i in errors)
    if focus['fallback_to_B']:
        ids = batch_ids(tuple(errors),seed,ordinal,3)
        return dict(errors=[errors[i] for i in ids], structured_feedback=None), dict(focus,
            representative_example_ids=list(ids), sampling='B_random_wrong_batch_fallback',
            sampling_state_id=state_id)
    ids = [r.example_id for r in sample_pattern_support(rows,focus['selected_support_ids'],
        seed=seed,state_id=state_id,member_id=0,ordinal=ordinal-1)]
    sf = projected['structured_feedback']
    selected = set(ids)
    # Keep the chosen Pattern's true support membership, but no nonrepresentative
    # raw example, Gradient, diagnosis or partition audit enters the Optimizer.
    sf = dict(gradients=[g for g in sf['gradients'] if g['example_id'] in selected],
        diagnostics=[d for d in sf['diagnostics'] if d['example_id'] in selected],
        patterns=deepcopy(sf['patterns']),unassigned_ids=[],partition_audit=[],cluster_structural_errors=[])
    return dict(errors=[errors[i] for i in ids],structured_feedback=sf), dict(focus,
        representative_example_ids=ids,sampling='formal_seeded_task_type_round_robin_up_to_three',
        sampling_state_id=state_id)


class FullDiagnosisPatternModule(LargestPatternModule):
    def binding(self):
        value = super().binding()
        root = Path(__file__).resolve().parents[2]
        value['production_source_hashes']['search/population_evidence.py'] = file_hash(root/'search/population_evidence.py')
        value.update(identity='ABLATION_FULL_WRONG_GRADIENT_PATTERN_REPRESENTATIVES_V1',
            implementation='ablations.full_pattern.FullDiagnosisPatternModule',required_error_source='all_wrong',
            semantic_calls_per_opportunity=150*3+3,maximum_diagnosis_examples=150,
            diagnosis='all_current_optimize_errors_before_any_sampling',
            representative_count=3,representative_sampler='search.population_evidence.sample_pattern_support',
            representative_salt='seed_prompt_hash_member0_round_minus_one_no_repeat_salt',
            optimizer_evidence='selected_pattern_membership_plus_representative_errors_gradients_diagnostics_only',
            no_pattern='B_random_three_wrong_errors_structured_null')
        value['production_implementations'].append('search.population_evidence.sample_pattern_support')
        return value

    def prepare(self,context,state,provider):
        if len(context.rows)>150:
            raise SearchContractError('ABLATION_FULL_DIAGNOSIS_OPTIMIZE_BOUND')
        full = dict(errors=[raw_error(r,context.observations[r['stable_example_id']]) for r in context.rows],
            structured_feedback=discover(context,provider))
        sampling = sampling_context(context)
        projected,focus=project_full(full,context.seed,context.ordinal,sampling)
        return PreparedFeedback(projected['errors'],projected['structured_feedback'],None,
            dict(focus=focus,complete_diagnosis=full,sampling_context=sampling),None)
