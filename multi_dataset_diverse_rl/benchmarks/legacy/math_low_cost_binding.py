"""Fresh low-cost authority; parent contracts and memberships remain immutable."""
import json
from ..math_v21_binding import MATHV21Binding
from ..math_low_cost import read_subsets,protocol_contract
from ..math_prediction_validity import prediction_validity_contract,invalid_recovery_contract
from ..math_execution import provider_bounds
from ..data_freeze import file_hash
from ..access import DataPurpose
from ..protocols import protocol_input
from ...governance.token_accounting import POLICY_40M
from ...search.schemas import SearchContractError
from ... import versions

_DEFAULT_OPTIMIZER_POLICY = object()

def derive_contract(parent,*,phase,attempt,binding_path,subsets_path,subsets_sha256,subsets,
        parent_path,parent_sha256,accounting_path,accounting_sha256,metadata_path,metadata_sha256,
        authorization_sha256,optimizer_policy=_DEFAULT_OPTIMIZER_POLICY,optimizer_authorization_sha256=None,
        optimizer_authorization_path=None,optimizer_nonthinking_evidence_policy=None):
    if phase not in {'canary','pilot'}:raise SearchContractError('LOW_COST_PHASE_INVALID')
    count=12 if phase=='canary' else 60
    support=subsets['membership_hashes']['canary_optimize' if phase=='canary' else 'pilot_optimize']
    bounds=provider_bounds(optimize_count=count,shadow_count=40)
    if phase=='canary':
        bounds.update(max_opportunities=1,reflection_calls=bounds['max_proposals_per_opportunity'],
            pattern_calls=1,solver_calls=5*(count+40)+bounds['solver_per_opportunity'])
    else:
        # Each strict integer team gain consumes at least one of N possible
        # gains; at most N+1 parent states each permit two no-commit epochs.
        # Failure-discount selection need not visit members round-robin.
        bounds['max_opportunities'] *= 2*(count+1)
        bounds['solver_calls']=5*(count+40)+bounds['max_opportunities']*bounds['solver_per_opportunity']
        bounds['reflection_calls']=bounds['max_opportunities']*bounds['max_proposals_per_opportunity']
        bounds['pattern_calls']=bounds['max_opportunities']
    bounds['solver_calls']*=4
    bounds['successful_provider_calls']=bounds['solver_calls']+bounds['reflection_calls']+bounds['pattern_calls']
    bounds['transport_attempts']=bounds['successful_provider_calls']*21
    c=dict(parent,identity=versions.MATH_LOW_COST_EXECUTION_BINDING_VERSION,
        binding_path=binding_path,execution_phase=phase,execution_attempt_id=attempt,
        canary_attempt_id=attempt,cache_namespace=attempt,cache_policy=versions.DURABLE_EXACT_OUTPUT_CACHE_VERSION,
        canary_phase='one_complete_production_opportunity_v1',shadow_count=40,
        low_cost_protocol=protocol_contract(subsets),low_cost_subsets_path=subsets_path,
        low_cost_subsets_sha256=subsets_sha256,amendment_parent_binding_path=parent_path,
        amendment_parent_binding_sha256=parent_sha256,accounting_policy_path=accounting_path,
        accounting_policy_sha256=accounting_sha256,validation_accounting_metadata_path=metadata_path,
        validation_accounting_metadata_sha256=metadata_sha256,continuation_authorization_sha256=authorization_sha256,
        prediction_validity_policy=dict(prediction_validity_contract(),
            identity=versions.MATH_RECOVERY_PREDICTION_VERSION,invalid_response_retries=3),
        invalid_recovery_policy=invalid_recovery_contract(),provider_bounds=bounds,
        initial_competence_binding=dict(parent['initial_competence_binding'],
            identity='MATH_INITIAL_COMPETENCE_LOW_COST_V1',count=count,support_identity=support))
    c['decoding']=dict(parent['decoding'],invalid_response_retries=3)
    if optimizer_policy is _DEFAULT_OPTIMIZER_POLICY:
        from ..math_optimizer_generation import optimizer_generation_contract
        optimizer_policy=optimizer_generation_contract()
    if optimizer_policy is not None:
        from ..math_optimizer_generation import optimizer_generation_contract
        if (not isinstance(optimizer_policy,dict)
                or optimizer_policy != optimizer_generation_contract(optimizer_policy.get('identity'))
                or not isinstance(optimizer_authorization_sha256,str)
                or len(optimizer_authorization_sha256)!=64 or not isinstance(optimizer_authorization_path,str)):
            raise SearchContractError('OPTIMIZER_AMENDMENT_AUTHORITY_REQUIRED')
        c.update(identity=versions.MATH_LOW_COST_OPTIMIZER_EXECUTION_BINDING_VERSION,
            optimizer_generation_policy=optimizer_policy,
            optimizer_amendment_authorization_path=optimizer_authorization_path,
            optimizer_amendment_authorization_sha256=optimizer_authorization_sha256)
    if optimizer_nonthinking_evidence_policy is not None:
        from ..math_optimizer_diagnostics import nonthinking_evidence_contract
        if optimizer_policy is None or optimizer_policy['enable_thinking'] is not False or optimizer_nonthinking_evidence_policy!=nonthinking_evidence_contract():
            raise SearchContractError('OPTIMIZER_NONTHINKING_EVIDENCE_POLICY_MISMATCH')
        c['optimizer_nonthinking_evidence_policy']=optimizer_nonthinking_evidence_policy
    from ..math_v21_interface import MATHV21BenchmarkAdapter
    c['benchmark_protocol_sha256']=MATHV21BenchmarkAdapter(c).protocol.identity()
    from ...governance.math_paired_validation import validation_policy
    c['post_search_validation_policy']=validation_policy(c)
    return c


class MATHLowCostBinding(MATHV21Binding):
    def blockers(self):
        c=self.contract
        try:
            from ..math_optimizer_generation import frozen_optimizer_policy
            optimizer=frozen_optimizer_policy(c)
            if optimizer:
                approval_path=self.path(c['optimizer_amendment_authorization_path'])
                approval=json.loads(approval_path.read_bytes())
                if (file_hash(approval_path)!=c['optimizer_amendment_authorization_sha256']
                        or approval['explicit_human_approval'] is not True or approval['policy']!=optimizer
                        or approval['total_accounting_authorization']!=40000000
                        or approval['pattern_real_calls_A1']!=0 or approval['test_model_calls']!=0):
                    return ('OPTIMIZER_AMENDMENT_AUTHORITY_MISMATCH',)
                if c.get('optimizer_nonthinking_evidence_policy') is not None and approval.get('optimizer_nonthinking_evidence_policy')!=c['optimizer_nonthinking_evidence_policy']:
                    return ('OPTIMIZER_NONTHINKING_EVIDENCE_AUTHORITY_MISMATCH',)
            parent_path=self.path(c['amendment_parent_binding_path'])
            if file_hash(parent_path)!=c['amendment_parent_binding_sha256']:
                return ('LOW_COST_PARENT_BINDING_HASH_MISMATCH',)
            parent=json.loads(parent_path.read_bytes())
            if parent['identity']!=versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION or MATHV21Binding(self.root,parent).blockers():
                return ('LOW_COST_PARENT_NOT_CONFORMANT',)
            subsets=read_subsets(self.root,c)
            reader=self.reader()
            # Every safe metadata row is an exact member of its frozen superset.
            fields=('stable_example_id','source_split','source_index','content_sha256','input_sha256','project_split')
            original={r['stable_example_id']:r for r in reader.members}
            if any(any(r[k]!=original[r['stable_example_id']][k] for k in fields) for r in subsets['metadata_universe']):
                return ('LOW_COST_SUPERSET_MEMBERSHIP_MISMATCH',)
            if (json.loads(self.path(c['accounting_policy_path']).read_bytes())!=POLICY_40M
                    or file_hash(self.path(c['accounting_policy_path']))!=c['accounting_policy_sha256']):
                return ('LOW_COST_ACCOUNTING_POLICY_MISMATCH',)
            expected=derive_contract(parent,phase=c['execution_phase'],attempt=c['execution_attempt_id'],
                binding_path=c['binding_path'],subsets_path=c['low_cost_subsets_path'],
                subsets_sha256=c['low_cost_subsets_sha256'],subsets=subsets,
                parent_path=c['amendment_parent_binding_path'],parent_sha256=c['amendment_parent_binding_sha256'],
                accounting_path=c['accounting_policy_path'],accounting_sha256=c['accounting_policy_sha256'],
                metadata_path=c['validation_accounting_metadata_path'],metadata_sha256=c['validation_accounting_metadata_sha256'],
                authorization_sha256=c['continuation_authorization_sha256'],
                optimizer_policy=c.get('optimizer_generation_policy') if c['identity']==versions.MATH_LOW_COST_OPTIMIZER_EXECUTION_BINDING_VERSION else None,
                optimizer_authorization_sha256=c.get('optimizer_amendment_authorization_sha256'),
                optimizer_authorization_path=c.get('optimizer_amendment_authorization_path'),
                optimizer_nonthinking_evidence_policy=c.get('optimizer_nonthinking_evidence_policy'))
            if c!=expected or not c['execution_attempt_id'].startswith('math_v2_1_low_cost_A1_seed81_'+c['execution_phase']+'_attempt'):
                return ('LOW_COST_FROZEN_CONTRACT_MISMATCH',)
            p=self.path(c['validation_accounting_metadata_path']);metadata=json.loads(p.read_bytes())
            rows=subsets['memberships']['pilot_validation']
            if (file_hash(p)!=c['validation_accounting_metadata_sha256']
                    or [(r['example_id'],r['input_sha256']) for r in metadata['examples']]!=[(r['stable_example_id'],r['input_sha256']) for r in rows]
                    or metadata.get('invalid_recovery_policy')!=c['invalid_recovery_policy']
                    or metadata.get('prediction_validity_policy')!=c['prediction_validity_policy']
                    or metadata.get('solver_decoding_policy')!=c['solver_decoding_policy']
                    or metadata['decoding']!=c['decoding'] or metadata['solver_output_interface']!=c['solver_output_interface']
                    or metadata['low_cost_protocol']!=c['low_cost_protocol']
                    or any(type(r['blank_prompt_serialized_request_bytes']) is not int or r['blank_prompt_serialized_request_bytes']<=0 for r in metadata['examples'])):
                return ('LOW_COST_VALIDATION_METADATA_MISMATCH',)
            return ()
        except (KeyError,ValueError,TypeError,OSError,SearchContractError):
            return ('LOW_COST_BINDING_INVALID',)

    def examples(self,role):
        if role not in {'optimize','shadow'}:raise SearchContractError('HELDOUT_SEARCH_ACCESS_FORBIDDEN')
        from ..math_low_cost import read_subsets
        reader=self.reader();c=self.contract
        name='pilot_shadow' if role=='shadow' else 'canary_optimize' if c['execution_phase']=='canary' else 'pilot_optimize'
        members=read_subsets(self.root,c)['memberships'][name]
        wanted={r['stable_example_id'] for r in members}
        reader.members=[r for r in reader.members if r['project_split']!=role or r['stable_example_id'] in wanted]
        reader.manifest=dict(reader.manifest,counts={**reader.manifest['counts'],role:len(members)})
        rows=reader.rows(role,DataPurpose.ADAPTIVE_GATE if role=='shadow' else DataPurpose.EVIDENCE)
        b=self.benchmark()
        from ...search.binary_runtime import CorrectnessExample
        return tuple(CorrectnessExample(protocol_input('math',r['stable_example_id'],r['content'],b.output_contract,
            protocol=b.protocol),r['reference_final_answer']) for r in rows)
