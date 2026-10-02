"""Explicit scientific amendment over the immutable accounting/search parent."""
from copy import deepcopy
import hashlib
import json
from .math_autonomous import MATHAutonomousBinding
from .math_domain_v2 import SETTINGS,MATHBenchmarkAdapterV2
from .experiment_splits import ExperimentSplitReader,COUNTS,ROLES
from .protocols import MATH_PROTOCOL_V2
from .data_freeze import digest,file_hash
from .. import versions


class MATHDomainBinding(MATHAutonomousBinding):
    def benchmark(self):
        return MATHBenchmarkAdapterV2()

    def reader(self):
        c=self.contract
        return ExperimentSplitReader(self.path(c['canonical_root']),self.path(c['split_directory']),'math',
            expected_manifest_sha256=c['split_manifest_sha256'],expected_protocol=versions.MATH_SCORABLE_SPLIT_VERSION)

    def blockers(self):
        c=self.contract
        try:
            if c['identity']!=versions.MATH_DOMAIN_EXECUTION_BINDING_VERSION:
                return ('MATH_DOMAIN_BINDING_MISMATCH',)
            parent=self.path(c['amendment_parent_binding_path'])
            if file_hash(parent)!=c['amendment_parent_binding_sha256']:
                return ('MATH_AMENDMENT_PARENT_MISMATCH',)
            original=json.loads(parent.read_bytes())
            from ..governance.math_paired_validation import POLICY as VALIDATION_POLICY
            if c['post_search_validation_policy']!=VALIDATION_POLICY:
                return ('MATH_VALIDATION_POLICY_MISMATCH',)
            if (c['answer_domain']!='MATH_ANSWER_DOMAIN_V2' or c['reference_validity']!='MATH_SCORABLE_REFERENCE_V2'
                    or c['evaluator']!='MATH_EQUIVALENCE_V2' or c['payload_parser_identity']!='MATH_PAYLOAD_PARSER_V2'
                    or c['benchmark_protocol_sha256']!=MATH_PROTOCOL_V2.identity()
                    or c['verify_settings_sha256']!=digest(SETTINGS) or c['split_version']!=versions.MATH_SCORABLE_SPLIT_VERSION
                    or json.loads(self.path(c['verify_settings_path']).read_bytes())!=SETTINGS):
                return ('MATH_ANSWER_DOMAIN_CONTRACT_MISMATCH',)
            # Restore only the explicitly amended scientific/data fields for
            # comparison against the immutable prior operational binding.
            projected=deepcopy(c)
            for key in ('amendment_parent_binding_path','amendment_parent_binding_sha256','answer_domain',
                    'payload_parser_identity','verify_settings_sha256','verify_settings_path','amendment_authorization_sha256','post_search_validation_policy'):
                projected.pop(key,None)
            for key in ('identity','benchmark_protocol_sha256','reference_validity','evaluator','split_version',
                    'split_directory','split_manifest_sha256','membership_hashes',
                    'validation_accounting_metadata_path','validation_accounting_metadata_sha256'):
                projected[key]=original[key]
            if MATHAutonomousBinding(self.root,projected).blockers():
                return ('MATH_UNAUTHORIZED_NON_DOMAIN_CHANGE',)
            reader=self.reader()
            if reader.manifest['counts']!=COUNTS['math'] or reader.manifest['hashes']!=c['membership_hashes']:
                return ('MATH_SCORABLE_MEMBERSHIP_MISMATCH',)
            # Durable preparation proof is bound to frozen source hashes; no
            # held-out reference is reopened by runtime search readiness.
            for role in ROLES:
                members=[r for r in reader.members if r['project_split']==role]
                if len(members)!=COUNTS['math'][role] or any(r['source_split']!=('test' if role=='test' else 'train') for r in members):
                    return ('MATH_SOURCE_ROLE_MISMATCH',)
            metadata_path=self.path(c['validation_accounting_metadata_path'])
            if file_hash(metadata_path)!=c['validation_accounting_metadata_sha256']:
                return ('VALIDATION_ACCOUNTING_METADATA_MISMATCH',)
            metadata=json.loads(metadata_path.read_bytes())
            selected=[r for r in reader.members if r['project_split']=='validation']
            if (metadata['split_manifest_sha256']!=c['split_manifest_sha256']
                    or metadata['solver_output_interface']!=c['solver_output_interface']
                    or metadata['decoding']!=c['decoding'] or metadata['context']!='ACCOUNTING_DATA_PREP_CONTEXT'
                    or [(r['example_id'],r['input_sha256']) for r in metadata['examples']] != [(r['stable_example_id'],r['input_sha256']) for r in selected]):
                return ('VALIDATION_ACCOUNTING_MEMBERSHIP_MISMATCH',)
            return ()
        except (KeyError,OSError,ValueError,TypeError):
            return ('MATH_DOMAIN_BINDING_INVALID',)


def execution_binding(root,contract):
    if contract['identity'] in {versions.MATH_V2_1_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION}:
        from .math_v21_binding import MATHV21Binding
        return MATHV21Binding(root,contract)
    return MATHDomainBinding(root,contract) if contract['identity']==versions.MATH_DOMAIN_EXECUTION_BINDING_VERSION else MATHAutonomousBinding(root,contract)
