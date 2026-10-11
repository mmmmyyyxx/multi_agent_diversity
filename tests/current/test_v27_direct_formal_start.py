"""Direct formal prerequisite integrity, no model or held-out access."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import pytest
from multi_dataset_diverse_rl.governance import direct_formal_start as direct,canary_progression
from multi_dataset_diverse_rl.governance.source_identity import hash_scope,current_scientific_files
from multi_dataset_diverse_rl.search.schemas import SearchContractError

def fixture(root):
    attempt='synthetic_v27_formal'
    path='experiments/protocols/synthetic_v27/offline_verification.json'
    r=dict(identity=direct.IDENTITY,method=direct.METHOD,attempt_id=attempt,
        user_instruction_sha256='a'*64,user_selected_direct_formal=True,
        canary_execution='NOT_RUN_CURRENT_VERSION',new_real_api_calls=0,
        validation_access='NOT_AUTHORIZED',test_access='SEALED',
        current_suite=dict(passed=743,failed=0,errors=0),network_attempts=0,
        compileall='PASS',governance='PASS',deterministic_output_feedback_smokes='PASS')
    p=root/path;p.parent.mkdir(parents=True);p.write_text(json.dumps(r),encoding='utf-8')
    policy=dict(identity=direct.IDENTITY,offline_verification_path=path,
        offline_verification_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),user_instruction_sha256='a'*64)
    source=root/'source.py';source.write_text('SYNTHETIC = True\n',encoding='utf-8')
    payload=dict(scope=dict(method=direct.METHOD,phase='formal_search_only',attempt_id=attempt),
        source_identity=dict(execution_closure=hash_scope(root,[source])))
    c=dict(method_identity=direct.METHOD,execution_phase='formal',execution_attempt_id=attempt,canary_prerequisite=policy)
    return c,payload,p,r

def test_direct_start_verifies_frozen_evidence_and_reports_no_real_canary(tmp_path):
    c,p,_,_=fixture(tmp_path)
    canary_progression.require_policy(c['canary_prerequisite'])
    result=canary_progression.verify(tmp_path,c,p)
    assert result['current_canary']=='NOT_RUN' and result['efficacy_gate'] is False
    assert 'canary_closure' not in result

@pytest.mark.parametrize('field,value',[
    ('method','old_method'),('attempt_id','other_attempt'),('user_selected_direct_formal',False),
    ('canary_execution','EXECUTION_COMPLETE'),('new_real_api_calls',1),('network_attempts',1),
    ('validation_access','AUTHORIZED'),('test_access','OPEN'),('compileall','FAIL'),
    ('governance','FAIL'),('deterministic_output_feedback_smokes','FAIL'),
    ('current_suite',dict(passed=742,failed=0,errors=0)),('current_suite',dict(passed=743,failed=1,errors=0))])
def test_bad_offline_record_cannot_admit_direct_start(tmp_path,field,value):
    c,p,path,r=fixture(tmp_path);r[field]=value;path.write_text(json.dumps(r),encoding='utf-8')
    c['canary_prerequisite']['offline_verification_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SearchContractError,match='OFFLINE_VERIFICATION_HOLD'):direct.verify(tmp_path,c,p)

def test_frozen_evidence_and_source_tampering_fail_closed(tmp_path):
    c,p,path,_=fixture(tmp_path);path.write_text('{}',encoding='utf-8')
    with pytest.raises(SearchContractError,match='HASH_MISMATCH'):direct.verify(tmp_path,c,p)

def test_tracked_source_tampering_fails_closed(tmp_path):
    c,p,_,_=fixture(tmp_path);(tmp_path/'source.py').write_text('CHANGED=True\n',encoding='utf-8')
    with pytest.raises(SearchContractError,match='SOURCE_IDENTITY_MISMATCH'):direct.verify(tmp_path,c,p)

@pytest.mark.parametrize('field,value',[('method_identity','old_method'),('execution_phase','pilot'),('execution_attempt_id','another')])
def test_only_exact_current_formal_attempt_admitted(tmp_path,field,value):
    c,p,_,_=fixture(tmp_path);c[field]=value
    with pytest.raises(SearchContractError,match='CURRENT_SCOPE_MISMATCH'):direct.verify(tmp_path,c,p)

@pytest.mark.parametrize('change',[dict(identity='UNKNOWN'),dict(offline_verification_sha256='bad'),
    dict(user_instruction_sha256='x'*64),dict(offline_verification_path='../outside.json'),dict(extra=True)])
def test_prerequisite_shape_is_closed(tmp_path,change):
    c,_,_,_=fixture(tmp_path);c['canary_prerequisite'].update(change)
    with pytest.raises(SearchContractError):canary_progression.require_policy(c['canary_prerequisite'])

def test_direct_start_spec_is_in_both_source_closures():
    root=Path(__file__).resolve().parents[2]
    for execution in (False,True):
        assert root/'docs/design/V27_DIRECT_FORMAL_START_V1.md' in current_scientific_files(root,execution_closure=execution)
