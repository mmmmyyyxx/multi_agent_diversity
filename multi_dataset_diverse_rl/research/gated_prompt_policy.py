"""Closed independent V2: legal changed candidates and Validation100 admission."""
import random
from .single_prompt_policy import policy as original_policy
from ..benchmarks.data_freeze import digest

STUDY='single_prompt_validation_gated_v2'
IDENTITY='SINGLE_PROMPT_VALIDATION100_STRICT_GATE_RETRY3_V2'
MANIFEST='experiments/manifests/single_prompt_validation_gated_v2.json'
SOURCE_BASE='d1baffa3d065f386d7077789f140c288a7253c94'


def policy():
    value=original_policy()
    value.update(identity=IDENTITY,optimizer_structural_draws=4,optimizer_structural_retries=3,
        invalid_candidate='initial_plus_three_retries_then_abandon_opportunity_keep_recorded_parent',
        unchanged_candidate='record_skip_evaluation_no_regeneration',
        acceptance='validation_correct_count_strictly_greater_than_retained_profile',
        optimize_gate=False,final_selection='last_validation_retained_prompt',
        no_errors='generate_changed_candidate_with_empty_error_feedback_no_fabrication',
        validation='adaptive_gate_100_fixed_members_no_raw_optimizer_feedback',
        validation_count=100,validation_membership_seed=88,
        validation_sampling='uniform_without_replacement_preserve_frozen_order',
        validation_cache='same_seed_same_request_all_rounds_arms_one_realization',
        validation_incumbent='reuse_recorded_retained_profile_no_rescore_lottery',
        candidate_evaluation='complete_optimize150_and_validation100',
        round_definition='one_opportunity_accepted_rejected_or_explicitly_abandoned_after_generation_exhaustion',
        generation_budget='model_context_only_no_character_budget',
        provider_failure='timeout_retry_until_frozen_calls_other_failure_abort_no_fallback',
        timeout_recovery=timeout_policy())
    return value


def call_ceiling():
    # Initial generation plus three structural retries per opportunity. Typed
    # transport timeouts remain governed separately and count toward this ceiling.
    return 9*21*(150+100)*4 + 3*20*(3*3+3) + 9*20*4


def timeout_policy():
    return dict(identity='GATED_PROMPT_ACCOUNTED_TIMEOUT_RECOVERY_V1',
        retry_errors=['APITimeoutError'],backoff_seconds=[2,4,8,16,30,60],
        combined_physical_call_ceiling=call_ceiling(),cumulative_token_ceiling=None)


def gate_membership(reader):
    ids=[r['stable_example_id'] for r in reader.members if r['project_split']=='validation']
    chosen=set(random.Random(policy()['validation_membership_seed']).sample(ids,policy()['validation_count']))
    selected=[i for i in ids if i in chosen]
    return dict(identity='FROZEN_VALIDATION100_FROM300_UNIFORM_SEED88_V1',
        sample_ids=selected,membership_sha256=digest(selected),source_count=len(ids),count=len(selected))
