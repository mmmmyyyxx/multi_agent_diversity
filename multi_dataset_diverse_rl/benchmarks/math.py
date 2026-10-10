"""Shared current mathematical evaluator for benchmark-level offline checks."""
from .math_domain_v2 import MATHBenchmarkAdapterV2
from ..search.schemas import ParsedOutput
from .math_flexible_answer import IDENTITY, classify_prediction, extract_answer
from .protocols import MATH_PROTOCOL_FLEXIBLE_V7
from ..versions import MATH_SOLVER_INTERFACE_V9_VERSION

class MATHBenchmarkAdapter(MATHBenchmarkAdapterV2):
    parser_identity = IDENTITY
    output_contract = MATH_SOLVER_INTERFACE_V9_VERSION
    protocol = MATH_PROTOCOL_FLEXIBLE_V7
    final_payload = staticmethod(lambda raw: extract_answer(raw)[0] or None)
    def parse_member_output(self,raw,item):
        if item.benchmark_id!=self.benchmark_id:return ParsedOutput('',False)
        return classify_prediction(raw).parsed()
    parse_output=parse_member_output
