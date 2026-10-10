"""Exact-source, zero-API replay of already sealed Optimize Gradient responses."""
import hashlib,importlib.util,io,json,subprocess,sys,tarfile
from pathlib import Path
from types import SimpleNamespace

import os,sitecustomize
assert os.environ.get('FORMAL_ZERO_API_GUARD_REQUIRED')=='1' and sitecustomize.network_attempt_count()==0
root=Path(__file__).resolve().parents[1]; out=root/'runs/pilot_semantic_alignment_20261011'; out.mkdir(exist_ok=True)
source='01306fbf9cde002291ab21e2f93a2c02399f7e9e'; published='30a57d82213fb7788581e3fe8fd9ba7903dfb302'
frozen=out/'frozen_source'; frozen.mkdir(exist_ok=True)
archive=subprocess.check_output(['git','archive',source,'multi_dataset_diverse_rl'])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    for member in tar.getmembers():
        assert (frozen/member.name).resolve().is_relative_to(frozen.resolve()) and not member.issym() and not member.islnk()
    tar.extractall(frozen,filter='data')
# Load current observation-only helper without placing current production on the
# replay import path. Every native validator below comes from the pinned source.
module_path=root/'multi_dataset_diverse_rl/diagnostics/pilot_observations.py'
spec=importlib.util.spec_from_file_location('observation_only',module_path); diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
sys.path.insert(0,str(frozen))
from multi_dataset_diverse_rl.search.textual_gradients import GradientExtractor,REFERENCE_GRADIENT_PROMPT
from multi_dataset_diverse_rl.search.optimization_evidence import POLICY
from multi_dataset_diverse_rl.search.system_prompt import SystemPrompt
from multi_dataset_diverse_rl.search.generation_failures import generation_failure
from multi_dataset_diverse_rl.search.policies import GlobalStopPolicy
from multi_dataset_diverse_rl import current_contract as versions

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
rows=lambda p:[json.loads(s) for s in p.read_bytes().splitlines() if s]
facts=dict(source_sha=source,published_report_commit=published,phases={},real_api_calls=0,validation_raw_reads=0,test_raw_reads=0,official_scores_modified=False)
facts['frozen_package_archive_sha256']=hashlib.sha256(archive).hexdigest()
facts['diagnostic_helper_sha256']=sha(module_path)
parent=root/'reports/a4_v25_arm_b_openlux_seed84_dual_scope_execution_20261010'
for seal in read(parent/'sha256_manifest.json')['files']:
    assert sha(parent/seal['file'])==seal['sha256']
eol=[]
for path in parent.iterdir():
    if path.is_file():
        blob=subprocess.check_output(['git','show',published+':'+path.relative_to(root).as_posix()])
        local=path.read_bytes()
        assert local==blob or local.replace(b'\r\n',b'\n')==blob
        if local!=blob:eol.append(dict(file=path.name,local_sha256=sha(path),git_blob_sha256=hashlib.sha256(blob).hexdigest(),difference='CRLF_TO_LF_ONLY'))
facts['historical_report_eol_projection']=eol
facts['parent_report_manifest_sha256']=sha(parent/'sha256_manifest.json')
for phase in ('canary','pilot'):
    run=root/('runs/a4_v25_arm_b_openlux_seed84_'+phase+'_attempt1/execution')
    inventory=read(run/'raw_evidence_inventory.json')
    assert sha(run/'raw_evidence_inventory.json')==read(parent/'summary.json')['phases'][phase]['lifecycle']['raw_inventory_sha256']
    for f in inventory['files']:
        path=(run/f['path']).resolve(); assert path.is_relative_to(run.resolve())
        assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
    trace=[r for r in rows(run/'provider_trace_private.jsonl') if r['role']=='pattern_gradient' and 'response' in r]
    cursor=0; groups=[]; native_category_counts={}; streak_inputs=[]; stopper=GlobalStopPolicy()
    trajectories=rows(run/'trajectory_private.jsonl')
    for op_row in [r for r in trajectories if r['stage']=='OPPORTUNITY']:
        op=op_row['opportunity']; diagnostics=op['pattern_context']['gradient_diagnostics']
        transition=next(r for r in trajectories if r['stage']=='TRANSITION' and r['opportunity_id']==op['opportunity_id'])
        stopped=stopper.observe_opportunity(parent_state_id=op_row['parent']['team_state_id'],
            eligible_members=op['objective']['eligible_members'],selected_member=op['target_member'],
            local_update=False,committed=transition['committed'] is not None)
        assert stopped is None
        streak_inputs.append(dict(committed=transition['committed'] is not None,
            parent_vote=op_row['parent']['team_scores']['vote_correct_count'],child_vote=transition['child']['team_scores']['vote_correct_count']))
        for expected in diagnostics:
            draws=trace[cursor:cursor+expected['structural_draws']]; cursor+=len(draws)
            assert len(draws)==expected['structural_draws']
            packets=[json.loads(d['request']['messages'][1]['content']) for d in draws]
            packet=packets[0]; ex=packet['example']
            assert ex['example_id']==expected['example_id'] and all(p==packet for p in packets)
            assert all(d['split']=='optimize' and d['response']['usage_reliable'] and d['request']['messages'][0]['content']==REFERENCE_GRADIENT_PROMPT for d in draws)
            evidence=SimpleNamespace(example_id=ex['example_id'],source_split='optimize',signals=dict(
                input_payload=ex['problem'],gold=ex['reference'],target_output=ex['prediction'],
                target_member_correct=False,target_member_valid=ex['valid'],solver_trajectory=ex['solver_trajectory'],
                reference_solution=ex['reference_solution']['text'],responsibility_labels=[],
                correctness_signal_identity=versions.TARGET_CORRECTNESS_SIGNAL_VERSION))
            replay_index=0
            def extract(actual_packet):
                global replay_index
                assert actual_packet==packet
                response=draws[replay_index]['response']; replay_index+=1
                if response['finish_reason']!='stop':raise generation_failure('GENERATED_OUTPUT_TRUNCATED')
                try:return json.loads(response['text'])
                except (ValueError,TypeError):raise generation_failure('PATTERN_GRADIENT_EXTRACTION_INVALID')
            provider=SimpleNamespace(extract=extract,optimization_evidence_policy=POLICY)
            extractor=GradientExtractor(provider)
            extractor.extract(SystemPrompt.from_dict(packet['current_system_prompt']),(evidence,))
            got=extractor.evidence_diagnostics[0]
            for key in ('disposition','structural_draws','structural_errors','exhausted','recovery_succeeded'):
                assert got[key]==expected[key],key
            assert replay_index==len(draws)
            group=[]
            for i,draw in enumerate(draws):
                accepted=i>=len(expected['structural_errors']); response=draw['response']
                annotation=diagnostic.gradient_shape_annotation(response['text'],response['finish_reason'])
                native=None if accepted else expected['structural_errors'][i]
                if native:native_category_counts[native]=native_category_counts.get(native,0)+1
                category=annotation['primary']
                if category=='REQUIRES_FROZEN_CONTENT_VALIDATION':category='ACCEPTED' if accepted else 'FROZEN_CONTENT_'+native
                assert not accepted or category=='ACCEPTED'
                group.append(dict(accepted=accepted,category=category,issues=annotation['issues'],finish_reason=response['finish_reason'],input_tokens=response['input_tokens'],output_tokens=response['output_tokens']))
            groups.append(group)
    assert cursor==len(trace)
    summary=diagnostic.gradient_cost_summary(groups)
    recovery=read(parent/'recovery_costs.json')[phase]; ledger=read(parent/'api_ledger_summary.json')['phases'][phase]
    assert summary['extra_draws']==recovery['gradient_structural_retry_responses'] and summary['extra_usage_tokens']==recovery['gradient_structural_retry_usage_tokens']
    assert summary['total_usage_tokens']==ledger['charged_by_role']['pattern_gradient']
    facts['phases'][phase]=dict(gradient=summary,native_rejection_categories=native_category_counts,
        opportunity_streaks=diagnostic.opportunity_streaks(streak_inputs),raw_inventory_sha256=sha(run/'raw_evidence_inventory.json'),
        raw_provider_trace_sha256=sha(run/'provider_trace_private.jsonl'),raw_trajectory_sha256=sha(run/'trajectory_private.jsonl'),
        native_validator_replay='EXACT_FROZEN_SOURCE_ALL_DRAWS_PASS',phase_usage_tokens=ledger['charged_tokens'])
    facts['phases'][phase]['frozen_epoch_stop_observation']=dict(completed_no_commit_epochs=stopper.no_commit_epochs,
        seen_members_in_current_epoch=len(stopper.seen_members),eligible_members_in_current_epoch=len(stopper.eligible_members),
        scientific_stop_reached=False,local_update='NOT_NEEDED_BY_FROZEN_STOP_DECISION')
facts['replay_import_source_verified']=all(Path(m.__file__).resolve().is_relative_to(frozen.resolve()) for name,m in sys.modules.items() if name.startswith('multi_dataset_diverse_rl') and getattr(m,'__file__',None))
assert facts['replay_import_source_verified']
assert sitecustomize.network_attempt_count()==0
(out/'diagnostic_facts.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='EXACT_SOURCE_DIAGNOSTIC_PASS',phases={k:dict(logical=v['gradient']['logical_gradients'],draws=v['gradient']['physical_draws']) for k,v in facts['phases'].items()},real_api_calls=0)))
