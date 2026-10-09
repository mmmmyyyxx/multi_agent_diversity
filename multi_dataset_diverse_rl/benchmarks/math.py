"""Shared current mathematical evaluator for benchmark-level offline checks."""
from .math_domain_v2 import MATHBenchmarkAdapterV2
from ..search.schemas import ParsedOutput

class MATHBenchmarkAdapter(MATHBenchmarkAdapterV2):
    def parse_member_output(self,raw,item):
        from .math_structured_answer import classify_prediction
        if item.benchmark_id!=self.benchmark_id:return ParsedOutput('',False)
        return classify_prediction(raw).parsed()
    parse_output=parse_member_output
