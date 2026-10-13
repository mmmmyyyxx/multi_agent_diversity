"""Guard-required synthetic nine-cell gated pipeline; no real dataset access."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    if os.environ.get('FORMAL_ZERO_API_GUARD_REQUIRED')!='1':
        raise SystemExit('Use tests/formal_zero_api_runner.py --offline-command')
    import socket,sitecustomize
    if socket.socket.connect.__module__!='sitecustomize' or sitecustomize.network_attempt_count():
        raise SystemExit('Pre-import network guard required')
    from multi_dataset_diverse_rl.research.gated_prompt import run_cell
    from multi_dataset_diverse_rl.research.gated_prompt_policy import policy
    from multi_dataset_diverse_rl.research.single_prompt_governance import private_path
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True,type=Path)
    destination=private_path(ROOT,parser.parse_args().destination)
    if destination.exists():raise SystemExit('Fresh synthetic destination required')
    destination.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('gated_smoke',ROOT/'tests/current/test_gated_prompt_study.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    class ExhaustFirstOpportunity(fixture.fixture.ToyTransport):
        def for_role(self,role,request):
            response=super().for_role(role,request)
            if role=='reflection' and self.reflections<=4:response['text']='synthetic malformed generation'
            return response
    broker=fixture.broker_for(destination,ExhaustFirstOpportunity());results=[]
    from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
    broker.receipts=ProviderResponseReceipts(destination/'provider_receipts',attempt_id='synthetic',startup_identity_sha256='a'*64)
    try:
        for arm in policy()['arms']:
            for seed in policy()['seeds']:
                cell=fixture.fixture.cell(arm,seed);directory=destination/cell['cell_id'];directory.mkdir()
                broker.cell_id=cell['cell_id'];broker.seed=seed;broker.attempt_id='SYNTHETIC_'+cell['cell_id']
                broker.transport.reflections=0
                results.append(run_cell(cell,fixture.fixture.toy_rows(5),fixture.fixture.toy_rows(3),broker,
                    lambda name,value:atomic_write_json(directory/(name+'.json'),value)))
        atomic_write_json(destination/'results_SYNTHETIC.json',results)
        # Exercise the real conditional Test wrapper with synthetic gold/problem
        # rows mapped to frozen metadata IDs. No actual held-out rows are opened.
        from multi_dataset_diverse_rl.research import gated_prompt_governance as gov
        from multi_dataset_diverse_rl.research import gated_prompt_test as final
        from multi_dataset_diverse_rl.governance import autonomous_math
        reader=gov.validate_manifest(ROOT,gov.manifest_template(ROOT))
        plan=final.formal_test_plan(ROOT,reader)
        synthetic_test_rows=tuple(dict(fixture.fixture.toy_rows(5)[i%5],stable_example_id=ident,
            content=dict(problem=f'Synthetic unique Test fixture {i}: {i%5}',solution=f'Synthetic reference value {i%5}.'))
            for i,ident in enumerate(plan['sample_ids']))
        class FakeReader:
            manifest_sha256=reader.manifest_sha256
            def heldout_rows(self,access):
                assert access.phase=='test';access.require_authorization();return synthetic_test_rows
        scope=dict(scope_identity='a'*64,source_sha='0'*40,source_hashes={},test_plan=plan,
            execution=dict(run_root=destination.relative_to(ROOT).as_posix()),
            cells=[dict(cell_id=r['cell_id'],arm=r['arm'],seed=r['seed'],attempt_id=r['cell_id']+'_train_fake') for r in results])
        prep=destination/'synthetic_prep'
        atomic_write_json(prep/'prep.json',dict(synthetic=True))
        atomic_write_json(prep/'authorization.json',dict(test_access=True))
        atomic_write_json(destination/'results.json',results)
        atomic_write_json(destination/'execution_summary.json',dict(usage=broker.usage_view(),token_ledger=broker.ledger.view(),cells_completed=9))
        broker.ledger.close()
        atomic_write_json(destination/'lifecycle.json',dict(status='EXECUTION_COMPLETE',phase='train'))
        atomic_write_json(destination/'terminal.json',dict(status='EXECUTION_COMPLETE',phase='train',scope_identity='a'*64,
            source_sha='0'*40,artifacts=gov.shared.inventory(destination)))
        # A fresh nested parent gives the Test wrapper its expected sibling output.
        # Private scope markers are synthetic and use the production single-use guard.
        (ROOT/'runs/gated_prompt_consumed').mkdir(parents=True,exist_ok=True)
        old_validate,old_manifest,old_plan=gov.validate_prep,gov.validate_manifest,final.formal_test_plan
        old_transport=autonomous_math.create_transport
        class Clients:
            def close(self):pass
        try:
            gov.validate_prep=lambda *args,**kwargs:scope
            gov.validate_manifest=lambda *args:FakeReader()
            final.formal_test_plan=lambda *args:plan
            autonomous_math.create_transport=lambda *args:(fixture.fixture.ToyTransport(),Clients())
            # The parent directory is fresh for this smoke and Test is a sibling.
            final_result=final.execute_final_test(ROOT,prep,destination)
            assert final_result['status']=='TRAIN_AND_TEST_EXECUTION_COMPLETE'
        finally:
            gov.validate_prep,gov.validate_manifest,final.formal_test_plan=old_validate,old_manifest,old_plan
            autonomous_math.create_transport=old_transport
        summary=dict(status='DRY_RUN_SYNTHETIC_COMPLETE_NOT_REAL_RESULTS',cells_completed=len(results),
            rounds_completed=sum(r['rounds_completed'] for r in results),
            prompt_checkpoints=sum(len(r['prompt_history']) for r in results),
            strict_gate_accepts=sum(m['accepted'] for r in results for m in r['metrics']),
            legal_gate_rejections=sum(m['candidate_evaluated'] and not m['accepted'] for r in results for m in r['metrics']),
            opportunities_abandoned=sum(r['opportunities_abandoned'] for r in results),
            synthetic_test_final_prompts=final_result['test_final_prompts'],
            fake_provider_usage=broker.usage_view(),unresolved_reservations=len(broker.ledger.inflight),
            real_api_calls=0,actual_validation_content_reads=0,actual_test_content_reads=0,
            network_attempts=sitecustomize.network_attempt_count())
        assert summary['cells_completed']==9 and summary['rounds_completed']==180 and summary['prompt_checkpoints']==189
        assert summary['opportunities_abandoned']==9 and summary['legal_gate_rejections']==171
        atomic_write_json(destination.parent/'dry_run_summary.json',summary);print(json.dumps(summary,indent=2))
    finally:broker.ledger.close()


if __name__=='__main__':main()
