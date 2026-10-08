"""Offline preparation using production binding and governance constructors."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.benchmarks.gradient_pilot_contract import derive_current_pilot_contract
from multi_dataset_diverse_rl.benchmarks.math_gradient_pattern_binding import BASELINE_PATH, BASELINE_SHA
from multi_dataset_diverse_rl.benchmarks.math_optimizer_generation import pattern_cluster_generation_contract
from multi_dataset_diverse_rl.benchmarks.math_visible_binding import (
    derive_visible_trajectory_contract, visible_gradient_prompt_artifact, derive_visible_accounting_metadata)
from multi_dataset_diverse_rl.benchmarks.math_evidence_binding import derive_evidence_contract, evidence_gradient_prompt_artifact
from multi_dataset_diverse_rl.governance.matched_realization import paired_policy, group_scope
from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest, prepare_canary

STUDY = 'a4_v23_matched_seed81_20261008_attempt1'
PROTOCOL = 'experiments/protocols/a4_v23_matched_comparison_v1'
PARENT = 'experiments/execution_bindings/math_v2_2_gradient_pattern_seed81_pilot_v2.json'
PATHS = {cell: f'experiments/execution_bindings/{STUDY}_{cell.lower()}.json' for cell in ('A', 'B')}
ATTEMPTS = {cell: STUDY + '_' + cell.lower() for cell in ('A', 'B')}


def write(relative, value):
    p = ROOT / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    return sha256(p.read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_bytes())


def prepare(task_hash, source):
    protocol_hash = write(PROTOCOL + '/protocol.json', dict(
        schema_version='matched_a4_real_comparison_protocol_v1',
        experiment_id='a4_v23_matched_comparison_v1',
        status='PREPARED_NOT_EXECUTED', preparation_user_task_sha256=task_hash,
        scientific_method_changed=False,
        execution_policy='MATCHED_SOLVER_REALIZATION_EXECUTION_POLICY_V1',
        normative_contract='docs/design/MATCHED_SOLVER_REALIZATION_EXECUTION_V1.md',
        starting_plan='experiments/protocols/math_optimization_evidence_v23/next_comparison.json',
        cells=dict(A='unified_team_prompt_search_v2_2 with V6', B='unified_team_prompt_search_v2_3'),
        seed=81, max_opportunities_per_arm=5, charged_token_ceiling_per_arm=2_000_000,
        study_charged_token_ceiling=4_000_000, generations=6, exports=4, full_candidates=2,
        solver='qwen3-8b', optimizer_gradient_pattern='qwen3.7-flash',
        solver_interface='MATH_SOLVER_INTERFACE_V6', solver_decoding='SOLVER_DECODING_POLICY_V2',
        optimizer_decoding='OPTIMIZER_REFLECTION_GENERATION_POLICY_V3',
        cluster_decoding='PATTERN_CLUSTER_GENERATION_POLICY_V1',
        initial_team_path=read(PARENT)['initial_team_path'],
        initial_team_sha256=read(PARENT)['initial_team_sha256'],
        optimize_count=60, shadow_count=40, aggregation='equal_equivalence_plurality_consistency_v1',
        transition='initial_competence_target_or_team_progress_v3',
        baseline='fresh paired Optimize60 measurements of all five members before search; historical 22/60 forbidden',
        access=dict(optimize='search and internal transfer', shadow='winner-only frozen gate',
            validation='not_authorized', test='sealed'),
        roles=['solver', 'reflection', 'pattern_gradient', 'pattern_cluster'],
        stopping=['frozen scientific stopper', 'NO_FEASIBLE_OPPORTUNITY', 'five-opportunity ceiling',
            'hard charged-token/request ceiling', 'provider, identity, persistence, ledger or canary failure'],
        retry_of_experiment=False, efficacy_based_stopping=False,
        required_metrics=['opportunities', 'proposals', 'contract_valid', 'mutation_positive',
            'search_validation_positive', 'validation_transfer', 'full_member_gain', 'newly_fixed',
            'newly_broken', 'retained', 'preservation_rate', 'invalid_outputs', 'vote_delta', 'oracle_delta',
            'full_admissible', 'commits', 'physical_requests', 'logical_evaluations', 'cache_hits', 'charged_tokens'],
        failure_analysis=['actionable_gradient', 'local_gain', 'transfer', 'preservation', 'team_gain', 'admission_or_operation'],
        candidate_audit=['parent_prompt_hash', 'pattern_hash', 'gradient_provenance', 'mutation_hypothesis_hash',
            'actual_diff_hash', 'Mutation', 'SearchValidation', 'TeamProbe', 'Full60', 'fixed', 'broken',
            'invalid_output_change', 'admission_decision', 'memory_record', 'subsequent_same_member_retrieval'],
        memory_claim='retrieval is not efficacy; audit subsequent actual edits and effects',
        real_api_authorized=False, READY_TO_RUN=False))
    # The intermediary is a new bounded settings receipt. It is never executed.
    parent_path = f'experiments/execution_bindings/{STUDY}_settings.json'
    parent_attempt = 'math_v2_2_gradient_pattern_A4_seed81_pilot_attempt_matched_settings_20261008'
    scope_path = PROTOCOL + '/settings_preparation_scope.json'
    scope_hash = write(scope_path, dict(schema_version='math_v2_2_operational_pilot_user_scope_v1',
        attempt_id=parent_attempt, arm='A4', seed=81, user_authorized=True, real_api_authorized=False,
        one_attempt_only=True, max_opportunities=5, token_ceiling=2_000_000,
        scientific_method_changed=False, validation_authorized=False, test_authorized=False,
        push_authorized=True, initial_memory_entries=0, raw_diagnostic_authorized=False,
        llm_judge_authorized=False, canary_authorized=False, operational_retry_limit=0,
        exact_frozen_api_authorization_required=True, user_task_sha256=task_hash,
        receipt_only_not_an_executable_experiment=True))
    amendment_path = PROTOCOL + '/cluster_settings_provenance.json'
    amendment_hash = write(amendment_path, dict(schema_version='pattern_cluster_output_amendment_v1',
        policy=pattern_cluster_generation_contract(), user_task_sha256=task_hash,
        parent_binding_sha256=sha256((ROOT / PARENT).read_bytes()).hexdigest(),
        scientific_method_changed=False))
    parent = derive_current_pilot_contract(read(BASELINE_PATH), attempt=parent_attempt,
        binding_path=parent_path, parent_path=BASELINE_PATH, parent_sha256=BASELINE_SHA,
        authorization_path=scope_path, authorization_sha256=scope_hash,
        max_opportunities=5, token_ceiling=2_000_000,
        cluster_generation_amendment=dict(policy=pattern_cluster_generation_contract(),
            path=amendment_path, sha256=amendment_hash, parent_path=PARENT,
            parent_sha256=sha256((ROOT / PARENT).read_bytes()).hexdigest()))
    parent_hash = write(parent_path, parent)
    metadata_path = PROTOCOL + '/v6_accounting_metadata.json'
    contracts = {}
    for cell, derive, artifact in [('A', derive_visible_trajectory_contract, visible_gradient_prompt_artifact),
                                  ('B', derive_evidence_contract, evidence_gradient_prompt_artifact)]:
        gradient_path = PROTOCOL + '/gradient_' + cell.lower() + '.json'
        gradient_hash = write(gradient_path, artifact())
        authority_path = PROTOCOL + '/preparation_scope_' + cell.lower() + '.json'
        # Metadata and user-scope hashes are inserted after the deterministic
        # policy is derived, avoiding any self-reference in the binding.
        fresh = dict(attempt=ATTEMPTS[cell], binding_path=PATHS[cell],
            parent_path=parent_path, parent_sha256=parent_hash,
            authorization_path=authority_path, authorization_sha256='0' * 64,
            gradient_prompt_path=gradient_path, gradient_prompt_sha256=gradient_hash,
            validation_metadata_path=metadata_path, validation_metadata_sha256='0' * 64)
        c = derive(parent, **fresh)
        policy = paired_policy(c, group_id=STUDY, cell=cell, attempts=ATTEMPTS,
            binding_paths=PATHS, protocol_sha256=protocol_hash)
        metadata = derive_visible_accounting_metadata(read(parent['validation_accounting_metadata_path']), parent, c)
        metadata_hash = write(metadata_path, metadata)
        scope = dict(schema_version='math_matched_comparison_preparation_scope_v1',
            attempt_id=ATTEMPTS[cell], one_attempt_only=True, user_authorized=True,
            real_api_authorized=False, paired_realization_policy=policy,
            parent_binding_sha256=parent_hash, user_task_sha256=task_hash,
            validation_authorized=False, test_authorized=False,
            preparation_only=True, paid_execution_requires_new_explicit_approval=True)
        scope_hash = write(authority_path, scope)
        fresh.update(authorization_sha256=scope_hash, validation_metadata_sha256=metadata_hash,
                     paired_realization_policy=policy)
        c = derive(parent, **fresh)
        write(PATHS[cell], c)
        contracts[cell] = c
    manifests = {}
    for cell, c in contracts.items():
        ident = 'a4_v23_matched_comparison_v1_' + cell.lower()
        manifest = preexecution_manifest(ROOT, source_sha=source, frozen=source != '0' * 40,
            binding_path=PATHS[cell], experiment_id=ident)
        path = ROOT / ('experiments/manifests/' + ident + '.yaml')
        path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding='utf-8')
        manifests[cell] = manifest
    per_cell = {cell: dict(provider_bounds=c['provider_bounds'],
        first_attempt_solver_logical=500 + 5 * c['provider_bounds']['solver_per_opportunity'],
        optimizer_output_caps=dict(gradient=1800, reflection=1800, cluster=8192),
        solver_output_cap=3600, attempt_charged_token_ceiling=2_000_000)
        for cell, c in contracts.items()}
    write(PROTOCOL + '/budget.json', dict(schema_version='matched_real_budget_v1',
        per_cell=per_cell, study_charged_token_ceiling=4_000_000,
        physical_transport_ceiling=sum(c['provider_bounds']['transport_attempts'] for c in contracts.values()),
        gross_reservations_are_not_spending=True, guarantees_five_opportunities=False,
        reservation_rule='serialized UTF-8 request bytes + 4096 + role accounting output ceiling',
        missing_usage='charge full reservation', solver_semantic_attempts=4, transport_attempts_per_draw=21,
        gradient_semantic_attempts=dict(A=3, B=1), validation_calls=0, test_calls=0,
        canary_extra_provider_calls=0, canary_is_initial_optimize_profiling=True,
        paired_bootstrap_cost='charged once to A; both logical baselines reported',
        historical_cost_reference=dict(tokens=480991, completed_opportunities=7,
            interface='V5 answer-only; no empirical V6 cost estimate available')))
    if source != '0' * 40:
        preps = {}
        for cell, manifest in manifests.items():
            destination = ROOT / 'runs' / STUDY / ('prep_' + cell.lower())
            preps[cell] = prepare_canary(ROOT, manifest, destination=destination)
        write('runs/' + STUDY + '/paired_authorization.json', dict(explicit_user_authorized=False,
            single_use_per_cell=True, scope=group_scope(contracts['A']['paired_realization_policy']),
            approved_startups={ATTEMPTS[cell]: p['startup_identity_sha256'] for cell, p in preps.items()},
            source_sha=source))
        write(PROTOCOL + '/execution_freeze.json', dict(source_sha=source,
            preps={cell: dict(startup_identity_sha256=p['startup_identity_sha256'], scope=p['scope'])
                for cell, p in preps.items()}, ready_for_authorization=True,
            real_api_authorized=False, READY_TO_RUN=False,
            commands={cell: f'python scripts/run_experiment.py --prep runs/{STUDY}/prep_{cell.lower()} --run-root runs/{ATTEMPTS[cell]} --execute'
                for cell in ('A', 'B')}))
    print(json.dumps(dict(status='FROZEN_NOT_AUTHORIZED' if source != '0' * 40 else 'DRAFT',
        source_sha=source, max_opportunities_per_arm=5, token_ceiling_per_arm=2_000_000)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task-sha256', required=True)
    p.add_argument('--source-sha', default='0' * 40)
    args = p.parse_args()
    if len(args.task_sha256) != 64 or len(args.source_sha) != 40:
        p.error('full hash identities required')
    prepare(args.task_sha256, args.source_sha)
