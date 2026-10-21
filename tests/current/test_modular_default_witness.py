"""Structural candidates have new witnesses; preserve old c938 evidence intact."""
import json
from pathlib import Path
import pytest
from tests.current.modular_witness import CASES,capture
from multi_dataset_diverse_rl.search.parent_population import digest

@pytest.mark.parametrize('name',CASES)
def test_default_matches_structural_complete_witness(name,tmp_path):
    from multi_dataset_diverse_rl.search.candidate_contract import POLICY
    fixture=json.loads(Path(__file__).with_name('modular_hendrycks_scoring_witness_v1.json').read_bytes())
    historical=json.loads(Path(__file__).with_name('modular_default_witness.json').read_bytes())
    assert fixture['candidate_policy']==POLICY
    observed=capture(name,tmp_path/name)
    assert digest(observed)==fixture['cases'][name]['canonical_sha256']
    assert observed['request_count']==fixture['cases'][name]['request_count']
    assert observed['request_count']==historical['cases'][name]['request_count']
