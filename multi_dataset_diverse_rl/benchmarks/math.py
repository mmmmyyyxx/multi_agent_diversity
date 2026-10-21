"""Shared current mathematical evaluator for benchmark-level offline checks."""
from .math_domain_v2 import MATHBenchmarkAdapterV2
from ..search.schemas import ParsedOutput
from .math_flexible_answer import IDENTITY, classify_prediction, extract_answer
from .protocols import MATH_PROTOCOL_HENDRYCKS
from ..versions import MATH_HENDRYCKS_INTERFACE_VERSION

class MATHBenchmarkAdapter(MATHBenchmarkAdapterV2):
    parser_identity = IDENTITY
    output_contract = MATH_HENDRYCKS_INTERFACE_VERSION
    protocol = MATH_PROTOCOL_HENDRYCKS
    final_payload = staticmethod(lambda raw: extract_answer(raw)[0] or None)
    def parse_member_output(self,raw,item):
        if item.benchmark_id!=self.benchmark_id:return ParsedOutput('',False)
        return classify_prediction(raw).parsed()
    parse_output=parse_member_output
