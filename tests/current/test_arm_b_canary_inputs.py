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

def test_seed84_binding_broker_wrong_seed_and_historical_execution_rejected():
 old=read_json(ROOT/BINDING)
 assert MATHStructuredBinding(ROOT,old).blockers()
 with pytest.raises(SearchContractError):RequestBroker(contract=old,transport=lambda _:None,arm='A4',seed=83)
 c=read_json(ROOT/'experiments/execution_bindings/a4_v25_arm_b_recovery_seed84_canary_attempt1.json');b=MATHStructuredBinding(ROOT,c)
 assert b.blockers()  # Frozen pre-migration source remains historical evidence.
 from tests.current.test_structured_optimization_evidence import contract
 c=contract();c['execution_seed']=84;c['seeds']=[84]
 assert c['execution_seed']==84 and c['seeds']==[84]
 assert RequestBroker(contract=c,transport=lambda _:None,arm='A4',seed=84).seed==84
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
 assert all(r['messages'][0]['content']==ARM_B_SEED.render() for r in requests if r['model']=='gpt-4o-mini')


def test_arm_b_complete_search_keeps_member_lanes_fixed_peers_and_shadow(tmp_path,monkeypatch):
 import asyncio
 from multi_dataset_diverse_rl.benchmarks.math_structured_interface import system_prompt_policy,system_interface_contract
 from tests.current import test_structured_optimization_evidence as fixture
 original_contract=fixture.contract
 def arm_b_contract():
  c=original_contract()
  c.update(initial_team_version=ARM_B_TEAM_VERSION,
   initial_team_sha256=initial_team_artifact(ARM_B_TEAM_VERSION)['ordered_team_sha256'],
   system_prompt_policy=system_prompt_policy(ARM_B_TEAM_VERSION),
   solver_output_interface=system_interface_contract(ARM_B_TEAM_VERSION))
  return c
 monkeypatch.setattr(fixture,'contract',arm_b_contract)
 monkeypatch.setattr(fixture,'BASE',ARM_B_SEED)
 bad=ARM_B_SEED.edit('strategy','Use case analysis to check signs.')
 good=ARM_B_SEED.edit('strategy','Check signs and substitute the result to verify constraints.')
 monkeypatch.setattr(fixture,'BAD',bad)
 monkeypatch.setattr(fixture,'GOOD',good)
 events=[];original_write=RequestBroker._write
 def write(broker,row):
  events.append(deepcopy(row));original_write(broker,row)
 monkeypatch.setattr(RequestBroker,'_write',write)
 run,broker,requests,packets,_=fixture.graph(tmp_path)
 original_profiles=deepcopy(run.state.profiles)
 assert run.state.prompts==(ARM_B_SEED,)*5
 for j in range(len(run.state.examples)):
  sources=[run.state.profiles[m][j]['solver_trajectory']['source'] for m in range(5)]
  assert [s['member_id'] for s in sources]==list(range(5))
  assert len({s['request_sha256'] for s in sources})==5
 result=asyncio.run(run.run(max_opportunities=1))
 target=result.trace[0].target_member
 assert result.trace[0].committed_candidate_id and len(result.transitions)==1
 assert run.state.prompts[target]==good
 for m in range(5):
  if m!=target:
   assert run.state.prompts[m]==ARM_B_SEED and run.state.profiles[m]==original_profiles[m]
 solver_requests=[r for r in requests if r['model']=='gpt-4o-mini']
 assert all(r['messages'][0]['content'] in {p.render() for p in (ARM_B_SEED,bad,good)} for r in solver_requests)
 assert len(solver_requests[:90])==90
 assert all(r['messages'][0]['content']==ARM_B_SEED.render() for r in solver_requests[:90])
 stages={e['stage'] for e in events if e['kind']=='SOLVER_REQUEST_CONTRACT'}
 assert {'initial','gepa_local','assigned_repair_check','independent_team_probe','full','adaptive_gate'}<=stages
 shadow=[e for e in events if e['kind']=='PREDICTION_VALIDITY' and e['split']=='shadow']
 assert {e['member_id'] for e in shadow}==set(range(5))
 assert len({e['request_sha256'] for e in shadow})==18  # Five original lanes plus one edited lane, three cases each.
 assert run.gate.winner_count==1 and packets
