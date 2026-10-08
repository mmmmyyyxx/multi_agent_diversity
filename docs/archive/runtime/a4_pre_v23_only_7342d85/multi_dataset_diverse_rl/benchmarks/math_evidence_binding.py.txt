"""Fresh V2.3 preparation; historical bindings never grant eligibility or API access."""
from copy import deepcopy
from dataclasses import asdict
import json

from ..search.current_layer1 import EvidenceLayer1Config
from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.optimization_evidence import POLICY, BINDING, METHOD, MEMORY, EVIDENCE, INPUT
from ..search.schemas import SearchContractError
from ..search.textual_gradients import REFERENCE_GRADIENT_PROMPT, pattern_policy_for_trajectory
from .math_visible_binding import derive_visible_trajectory_contract
from .math_gradient_pattern_binding import MATHGradientPatternBinding
from .data_freeze import file_hash


def evidence_gradient_prompt_artifact():
    return dict(identity='REFERENCE_SOLUTION_GRADIENT_PROMPT_V5',prompt=REFERENCE_GRADIENT_PROMPT)


def derive_evidence_contract(parent, **fresh):
    c=derive_visible_trajectory_contract(parent,**fresh)
    c.update(identity=BINDING,method_identity=METHOD,optimization_evidence_policy=deepcopy(POLICY),
        optimizer_input_schema=INPUT,layer1_search_policy=asdict(EvidenceLayer1Config()),
        memory_policy_identity=MEMORY,panel_evidence_policy=EVIDENCE,
        pattern_policy=pattern_policy_for_trajectory(c['solver_trajectory_policy'],POLICY))
    # One accepted independent diagnostic per wrong example. UNCERTAIN is valid,
    # not a contract retry or semantic regeneration.
    c.pop('gradient_recovery_policy',None)
    for key in tuple(c):
        if key.startswith('gradient_recovery_amendment_'):c.pop(key)
    bound=deepcopy(c['provider_bounds']);k=c['operational_pilot']['max_opportunities']
    proof=bound['bound_proof'];bootstrap=proof['bootstrap_logical_solver_calls']
    multiplier=proof['solver_semantic_multiplier'];transport=proof['transport_multiplier']
    per_op=dict(local=42,probe=12,full=120,shadow=40)
    logical=sum(per_op.values());solver=multiplier*(bootstrap+k*logical)
    gradient=60*k;cluster=k;reflection=6*k;successful=solver+gradient+cluster+reflection
    bound.update(solver_per_opportunity=logical,physical_solver_per_opportunity=multiplier*logical,
        solver_calls=solver,pattern_gradient_calls=gradient,pattern_cluster_calls=cluster,
        pattern_calls=gradient+cluster,reflection_calls=reflection,
        successful_provider_calls=successful,transport_attempts=successful*transport)
    proof.update(identity='V2_3_OPTIMIZE_EVIDENCE_RESOURCE_DERIVATION_V1',
        per_op_logical_solver=per_op,gradient_semantic_multiplier=1)
    c['provider_bounds']=bound
    return c


class MATHEvidenceBinding(MATHGradientPatternBinding):
    def __init__(self,root,contract):
        self.root=root.resolve();self.contract=contract

    def blockers(self):
        c=self.contract
        try:
            parent_path=c['trajectory_parent_binding_path']
            if file_hash(self.path(parent_path))!=c['trajectory_parent_binding_sha256']:
                raise SearchContractError('MATH_EVIDENCE_PARENT_HASH_MISMATCH')
            parent=json.loads(self.path(parent_path).read_bytes())
            blockers=MATHGradientPatternBinding(self.root,parent).blockers()
            if blockers:raise SearchContractError(blockers[0])
            expected=derive_evidence_contract(parent,attempt=c['execution_attempt_id'],
                binding_path=c['binding_path'],parent_path=parent_path,parent_sha256=c['trajectory_parent_binding_sha256'],
                authorization_path=c['current_user_scope_path'],authorization_sha256=c['current_user_scope_sha256'],
                gradient_prompt_path=c['gradient_prompt_path'],gradient_prompt_sha256=c['gradient_prompt_sha256'],
                validation_metadata_path=c['validation_accounting_metadata_path'],
                validation_metadata_sha256=c['validation_accounting_metadata_sha256'],
                paired_realization_policy=c.get('paired_realization_policy'))
            if c!=expected:raise SearchContractError('MATH_EVIDENCE_FROZEN_SETTINGS_CHANGED')
            for p,h in [('current_user_scope_path','current_user_scope_sha256'),
                    ('gradient_prompt_path','gradient_prompt_sha256'),
                    ('validation_accounting_metadata_path','validation_accounting_metadata_sha256')]:
                if file_hash(self.path(c[p]))!=c[h]:raise SearchContractError('MATH_EVIDENCE_DEPENDENCY_HASH_MISMATCH')
            if json.loads(self.path(c['gradient_prompt_path']).read_bytes())!=evidence_gradient_prompt_artifact():
                raise SearchContractError('MATH_EVIDENCE_GRADIENT_PROMPT_MISMATCH')
            scope=json.loads(self.path(c['current_user_scope_path']).read_bytes())
            required=dict(schema_version='math_optimization_evidence_user_scope_v1',
                attempt_id=c['execution_attempt_id'],one_attempt_only=True,user_authorized=True,
                real_api_authorized=False,optimization_evidence_policy=POLICY,
                parent_binding_sha256=c['trajectory_parent_binding_sha256'],
                validation_authorized=False,test_authorized=False)
            if c.get('paired_realization_policy'):
                required=dict(schema_version='math_matched_comparison_preparation_scope_v1',
                    attempt_id=c['execution_attempt_id'],one_attempt_only=True,
                    user_authorized=True,real_api_authorized=False,
                    paired_realization_policy=c['paired_realization_policy'],
                    parent_binding_sha256=c['trajectory_parent_binding_sha256'],
                    validation_authorized=False,test_authorized=False,preparation_only=True,
                    paid_execution_requires_new_explicit_approval=True)
            if (any(scope.get(k)!=v for k,v in required.items())
                    or not isinstance(scope.get('user_task_sha256'),str)
                    or len(scope['user_task_sha256'])!=64):
                raise SearchContractError('MATH_EVIDENCE_FRESH_USER_SCOPE_REQUIRED')
            CURRENT_POLICY_BUNDLE.validate_contract(c)
            from .current_math_dependencies import validate_effective_math_dependencies
            validate_effective_math_dependencies(self)
            return ()
        except (SearchContractError,KeyError,TypeError,ValueError,OSError) as exc:
            return (str(exc) if isinstance(exc,SearchContractError) else 'MATH_EVIDENCE_FRESH_FREEZE_REQUIRED',)
