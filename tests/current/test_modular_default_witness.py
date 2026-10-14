"""Default method compared to pre-refactor c938 full synthetic witnesses."""
import json
from pathlib import Path
import pytest
from tests.current.modular_witness import CASES,capture
from multi_dataset_diverse_rl.search.parent_population import digest

@pytest.mark.parametrize('name',CASES)
def test_default_matches_original_complete_witness(name,tmp_path):
    fixture=json.loads(Path(__file__).with_name('modular_default_witness.json').read_bytes())
    observed=capture(name,tmp_path/name)
    assert digest(observed)==fixture['cases'][name]['canonical_sha256']
    assert observed['request_count']==fixture['cases'][name]['request_count']
