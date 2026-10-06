"""Negative controls for post-execution wire audit, without runtime artifacts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from multi_dataset_diverse_rl.governance.gradient_pilot_audit import verify_request_firewall,text_hash
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.textual_gradients import validate_gradient
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT=Path(__file__).resolve().parents[2]


class PilotAuditControls(unittest.TestCase):
    def wire(self):
        c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v2.json').read_text(encoding='utf-8'))
        benchmark=NS(output_contract='immutable output framing',solver_user_content=lambda p,item:p+'\n'+item.input_payload)
        prompt='Check assumptions before algebra.';item=NS(input_id='synthetic',input_payload='public synthetic input')
        messages=[dict(role='system',content=benchmark.output_contract),dict(role='user',content=benchmark.solver_user_content(prompt,item))]
        broker=RequestBroker(contract=c,transport=None,arm='A4',seed=81)
        request,key=broker._request_identity(role='solver',split='optimize',messages=messages,member_slot=1)
        row=dict(role='solver',split='optimize',physical_attempt_no=1,request=request,request_sha256=key)
        validity=dict(request_sha256=key,member_id=1,input_id='synthetic',split='optimize',mutable_prompt_sha256=text_hash(prompt))
        args=([row],[validity],[dict(attempt=1,member_realization_lane=1)],c,benchmark,
            dict(optimize=dict(synthetic=NS(item=item)),shadow={}),{text_hash(prompt):prompt})
        return args

    def test_exact_member_lane_and_messages_pass_without_transport(self):
        result=verify_request_firewall(*self.wire())
        self.assertEqual(result['request_identity_rows_checked'],1)
        self.assertEqual(result['solver_wire_rows_checked'],1)
        self.assertEqual(result['audit_provider_calls'],0)

    def test_context_lane_identity_and_wire_mutations_fail(self):
        for mutation in ('optimizer_context','system_interface','member_lane','request_hash','model','extra_field'):
            with self.subTest(mutation=mutation):
                args=list(self.wire());args[:3]=deepcopy(args[:3]);row=args[0][0]
                if mutation=='optimizer_context':row['request']['messages'][1]['content']+='\nprivate optimizer feedback'
                elif mutation=='system_interface':row['request']['messages'][0]['content']='altered interface'
                elif mutation=='member_lane':args[2][0]['member_realization_lane']=2
                elif mutation=='request_hash':row['request_sha256']='0'*64
                elif mutation=='model':row['request']['model']='other-model'
                elif mutation=='extra_field':row['request']['unexpected']='unfrozen'
                with self.assertRaises(AssertionError):verify_request_firewall(*args)

    def test_current_numeric_rule_admits_generic_mathematical_digit(self):
        row=NS(example_id='synthetic',signals=dict(input_payload='Simplify a rational expression.',gold='x',target_output='y'))
        gradient='Check that integer coefficients have greatest common divisor 1.'
        self.assertEqual(validate_gradient(gradient,(row,)),gradient)

    def test_synthetic_abstract_behavior_passes_existing_contract(self):
        row=NS(example_id='synthetic',signals=dict(input_payload='Simplify a rational expression.',gold='x',target_output='y'))
        gradient='Check integer coefficients for a common factor before declaring an expression reduced.'
        self.assertEqual(validate_gradient(gradient,(row,)),gradient)


if __name__=='__main__':unittest.main()
