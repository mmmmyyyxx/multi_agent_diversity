from copy import deepcopy
from pathlib import Path
import pytest
from multi_dataset_diverse_rl.benchmarks.math_canary_inputs import (
 ARM_B_TEAM_VERSION,ARM_B_SEED,build_fresh_canary_subsets)
from multi_dataset_diverse_rl.benchmarks.math_structured_binding import initial_team_artifact,MATHStructuredBinding
from multi_dataset_diverse_rl.benchmarks.current_math_dependencies import validate_current_initial_team
from multi_dataset_diverse_rl.persistence.durable_io import read_json
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT=Path(__file__).resolve().parents[2]
BINDING='experiments/execution_bindings/a4_v25_arm_b_seed83_canary_attempt1.json'

def test_five_identical_arm_b_hashes_and_original_seed_immutable():
 c=read_json(ROOT/BINDING);team=initial_team_artifact(ARM_B_TEAM_VERSION)
 validate_current_initial_team(team,c)
 assert all(m['prompt']==ARM_B_SEED.to_dict() for m in team['members'])
 assert initial_team_artifact()['ordered_team_sha256']!=team['ordered_team_sha256']
 poisoned=deepcopy(team);poisoned['members'][0]['prompt']['strategy']='Changed strategy.'
 with pytest.raises(SearchContractError):validate_current_initial_team(poisoned,c)

def test_seed83_binding_broker_and_wrong_seed_rejected():
 c=read_json(ROOT/BINDING);b=MATHStructuredBinding(ROOT,c)
 assert not b.blockers()
 assert c['execution_seed']==83 and c['seeds']==[83]
 assert RequestBroker(contract=c,transport=lambda _:None,arm='A4',seed=83).seed==83
 with pytest.raises(SearchContractError):RequestBroker(contract=c,transport=lambda _:None,arm='A4',seed=81)
 poisoned=deepcopy(c);poisoned['execution_seed']=81
 assert MATHStructuredBinding(ROOT,poisoned).blockers()

def test_membership_replays_exclusions_without_any_outcome():
 import hashlib
 c=read_json(ROOT/BINDING);data=read_json(ROOT/c['low_cost_subsets_path'])
 assert data==build_fresh_canary_subsets(data['metadata_universe'],c['split_manifest_sha256'],
  seed=83,excluded_example_hashes=data['excluded_example_hashes'])
 for name,role in [('canary_optimize','optimize'),('pilot_shadow','shadow')]:
  for row in data['memberships'][name]:
   assert row['project_split']==role
   assert hashlib.sha256(row['stable_example_id'].encode()).hexdigest() not in data['excluded_example_hashes']
 with pytest.raises(SearchContractError):build_fresh_canary_subsets(data['metadata_universe'],c['split_manifest_sha256'],
  seed=83,excluded_example_hashes=['not-a-hash'])

def test_real_entrypoint_uses_arm_b_and_seed83_with_fake_transport(tmp_path,monkeypatch):
 from tests.current.test_structured_production_entrypoint import _entrypoint_fixture
 c=read_json(ROOT/BINDING)
 result,requests,_,_=_entrypoint_fixture(tmp_path,monkeypatch,False,phase='canary',all_correct=True,
  seed=83,team_version=ARM_B_TEAM_VERSION,development_subsets=dict(path=c['low_cost_subsets_path'],
    sha256=c['low_cost_subsets_sha256'],protocol=c['low_cost_protocol']))
 assert all(r['messages'][0]['content']==ARM_B_SEED.render() for r in requests if r['model']=='qwen3-8b')
