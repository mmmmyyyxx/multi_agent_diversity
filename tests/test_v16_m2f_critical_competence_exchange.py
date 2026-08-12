from __future__ import annotations
import importlib.util
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load():
 spec=importlib.util.spec_from_file_location("critical_exchange",ROOT/"scripts/audit_v16_m2f_critical_competence_exchange.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def state(*,target_correct,vote_correct,peer_gold=0,peer_margin=0):
 return SimpleNamespace(team_correctness=(target_correct,False,False,False,False),vote_correct=vote_correct)
def test_role_policy_is_unique_before_pivotal(monkeypatch):
 m=load();monkeypatch.setattr(m,"build_peer_vote_context",lambda s,t:None);monkeypatch.setattr(m,"compute_member_aware_repair_opportunity",lambda **k:SimpleNamespace(unique_correct=True,pivotal_correct=True))
 assert m.exclusive_critical_role(state(target_correct=True,vote_correct=True),0)=="unique"
def test_exchange_counts_new_critical_and_old_loss(monkeypatch):
 m=load()
 parent={"loss":state(target_correct=True,vote_correct=True),"gain":state(target_correct=False,vote_correct=False)}
 candidate={"loss":state(target_correct=False,vote_correct=False),"gain":state(target_correct=True,vote_correct=True)}
 roles={(id(parent["loss"]),0):"pivotal",(id(parent["gain"]),0):"none",(id(candidate["loss"]),0):"none",(id(candidate["gain"]),0):"unique"}
 monkeypatch.setattr(m,"exclusive_critical_role",lambda s,t:roles[(id(s),t)])
 metrics,_=m.exchange_for_candidate(parent_states=parent,candidate_states_by_hash=candidate,target=0,responsibility={"gain"})
 assert metrics["critical_gain"]==metrics["critical_loss"]==1
 assert metrics["critical_net"]==0
 assert metrics["responsibility_to_unique"]==1
 assert metrics["responsibility_to_vote_conversion"]==1
def test_audit_is_explicitly_no_api():
 m=load();assert m.AUDIT_VERSION.endswith("_v1")

def test_oracle_and_vote_exchange_are_separate(monkeypatch):
 m=load()
 parent=SimpleNamespace(team_correctness=(False,False,False,False,False),vote_correct=False)
 candidate=SimpleNamespace(team_correctness=(True,False,False,False,False),vote_correct=False)
 monkeypatch.setattr(m,"exclusive_critical_role",lambda state,target:"unique" if state is candidate else "none")
 metrics,_=m.exchange_for_candidate(parent_states={"q":parent},candidate_states_by_hash={"q":candidate},target=0,responsibility=set())
 assert metrics["oracle_gain"]==1
 assert metrics.get("vote_gain",0)==0
 assert metrics["new_unique_gained"]==1

def test_only_parent_wrong_to_candidate_correct_is_new_critical(monkeypatch):
 m=load()
 parent=SimpleNamespace(team_correctness=(True,False,False,False,False),vote_correct=True)
 candidate=SimpleNamespace(team_correctness=(True,False,False,False,False),vote_correct=True)
 monkeypatch.setattr(m,"exclusive_critical_role",lambda state,target:"pivotal")
 metrics,_=m.exchange_for_candidate(parent_states={"q":parent},candidate_states_by_hash={"q":candidate},target=0,responsibility={"q"})
 assert metrics.get("new_pivotal_gained",0)==0
 assert metrics.get("responsibility_repair",0)==0
