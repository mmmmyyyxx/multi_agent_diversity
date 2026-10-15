"""Fresh Test continuation with bounded exact negative certificates before native verification."""
from dataclasses import asdict
from copy import deepcopy
import json,sys
from ...benchmarks.data_freeze import digest
from ...benchmarks.math_domain_v2 import require_scorable
from ...benchmarks.math_interval_certificate import POLICY
from ...benchmarks.math_certified_domain import certified_matrix
from ...persistence.durable_io import read_json,append_jsonl
from ...governance.token_accounting import audit_journal,POLICY_V27_FORMAL_UNCAPPED
from ...search.schemas import SearchContractError
from . import test_deadline as previous,test_resume as skeleton,governance as gov

MANIFEST='experiments/manifests/single_prompt_c_test_certified_v10.json'
PROTOCOL='docs/design/SINGLE_PROMPT_C_TEST_CERTIFIED_V10.md'
SCHEMA='single_prompt_c_test_certified_manifest_v10'
PREP_SCHEMA='single_prompt_c_test_certified_prep_v10'
PARENT='runs/ablations/pattern_c_only_test_deadline_v9'
PIN='runs/ablations/pattern_c_only_test_certified_v10/parent_pin.json'
PIN_SHA256='cf479feca25a27f7238d7852e74e299c8791fcfcf8fb0b309e5de0a223e17fb0'
EXPERIMENT_ID='single_prompt_c_test_certified_v10'
PARENT_MANIFEST=previous.MANIFEST
VERSION='V10'
HANDOFF_NAME='C-only Test continuation with exact far-inequality certificates'


def operation_policy():
    value=previous.operation_policy()
    value.update(identity='TEST_ONLY_CERTIFIED_FAR_REJECTION_BEFORE_PINNED_VERIFY_V3',
        certified_rejection_policy=deepcopy(POLICY),scoring_worker='math_certified_worker',
        certificate_timeout_outcome='not_an_incorrect_score; retain_typed_local_timeout_recovery',
        inherited_partial_cell='V7_V8_paid_draws_and_V9_zero_API_terminal_hashes_preserved')
    return value


def _contract():return sys.modules[__name__]
def parent_pin(root):return previous.parent_pin(root,contract=_contract())
def manifest_template():return previous.manifest_template(contract=_contract())
def validate_manifest(root,value):return previous.validate_manifest(root,value,contract=_contract())
def handoff_template(root,prep,scope):return previous.handoff_template(root,prep,scope,contract=_contract())
def prepare(root,prep):return previous.prepare(root,prep,contract=_contract())
def validate_prep(root,prep,*,authorized=False):return previous.validate_prep(root,prep,authorized=authorized,contract=_contract())
authorization_template=previous.authorization_template


def audit_parent(root):
    pin=parent_pin(root);base=root/PARENT;run=base/'test_run'
    if gov.inventory(base)!=pin['parent_inventory']:raise SearchContractError('CERTIFIED_TEST_PARENT_CHANGED')
    scope=read_json(base/'prep/prep.json')['scope']
    if (scope['source_sha']!=pin['parent_source_sha'] or scope['scope_identity']!=pin['parent_scope']
            or digest({k:v for k,v in scope.items() if k!='scope_identity'})!=scope['scope_identity']
            or read_json(run/'scope.json')!=scope or not pin['no_inflight_at_stop']
            or pin['max_same_context_retry']<3 or not read_json(base/'prep/authorization.json')['approved']):
        raise SearchContractError('CERTIFIED_TEST_PARENT_IDENTITY')
    parent=previous.audit_parent(root)
    view,events=audit_journal(run/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
    if len(events)!=1 or view['charged_total']!=0:raise SearchContractError('CERTIFIED_TEST_PARENT_ZERO_API_CHANGED')
    used=set()
    for line in (run/'provider_receipts/solver_paid_prefix.jsonl').read_bytes().splitlines():
        row=json.loads(line);key=(row['cell_id'],row['stage'],row['provider_request_sha256'])
        entry=parent['paid_prefix'].get(key)
        if (key in used or entry is None or entry['reservation_id']!=row['parent_reservation_id']
                or entry['receipt_sha256']!=row['parent_receipt_sha256']):
            raise SearchContractError('CERTIFIED_TEST_PARENT_REPLAY_LINK')
        used.add(key)
    if len(used)!=356:raise SearchContractError('CERTIFIED_TEST_PARENT_REPLAY_COUNT')
    for line in (run/'provider_receipts/solver_cache_access.jsonl').read_bytes().splitlines():
        row=json.loads(line);key=(row['cell_id'],row['request_sha256'])
        if (row['split']!='test' or key not in parent['terminal_observations']
                or parent['terminal_observations'][key]!=row['observation_sha256']):
            raise SearchContractError('CERTIFIED_TEST_PARENT_TERMINAL_HASH')
    first=read_json(run/scope['plan']['cells'][0]['cell_id']/'result.json')
    if first['predictions']!=parent['first_result']['predictions']:
        raise SearchContractError('CERTIFIED_TEST_COMPLETE_CELL_CHANGED')
    return parent


class TestResumeBroker(previous.TestResumeBroker):
    def score_resolved(self,resolved,row):
        context=dict(operation='resolved_answer_scoring',example_id=row['stable_example_id'],
            resolved_prediction_sha256=digest(asdict(resolved)))
        def operation():
            require_scorable(row['reference_final_answer'])
            if not resolved.prediction_valid:return False
            value=certified_matrix((str(row['reference_final_answer']),resolved.answer))
            if not value['valid'][0]:raise SearchContractError('REFERENCE_UNSCORABLE')
            correct=bool(value['valid'][1] and value['equivalence'][0][1])
            with self.lock:
                append_jsonl(self.receipts.directory/'local_scoring_certificates.jsonl',dict(context,
                    cell_id=self.cell_id,policy=POLICY['identity'],
                    input_sha256=digest([str(row['reference_final_answer']),resolved.answer]),
                    certified_pair_rejections=value['certified_pair_rejections'],correct=correct))
            return correct
        return self.retry_local(operation,context)


def execute(root,prep,run):return skeleton.execute(root,prep,run,contract=_contract())
