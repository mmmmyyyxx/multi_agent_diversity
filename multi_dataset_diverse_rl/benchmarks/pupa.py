"""PAPILLON's three-stage member contract; team privacy semantics remain HOLD."""
from dataclasses import dataclass
from typing import Callable

from ..search.schemas import BenchmarkCapabilities, SearchContractError
from .scorer_contracts import PUPAJudgeOutputs, pupa_contract_score


@dataclass(frozen=True)
class PAPILLONArtifacts:
    llm_request: str
    llm_response: str
    response: str
    pipeline_status: str = "COMPLETE"


class PAPILLONMemberPipeline:
    def __init__(self, *, trusted_redactor: Callable, untrusted_model: Callable,
                 trusted_synthesis: Callable) -> None:
        self.redact = trusted_redactor
        self.untrusted = untrusted_model
        self.synthesize = trusted_synthesis

    def run(self, private_query: str) -> PAPILLONArtifacts:
        try:
            request = self.redact(private_query)
            response = self.untrusted(request)
            final = self.synthesize(private_query, request, response)
            return PAPILLONArtifacts(request, response, final)
        except Exception:
            # Match pinned PAPILLON's empty artifact tuple on pipeline failure,
            # with an additional factual status, never raw error/private text.
            return PAPILLONArtifacts("", "", "", "INVALID_PIPELINE_EXCEPTION")


class PUPABenchmarkAdapter:
    benchmark_id = "pupa"
    preferred_aggregation = "unfrozen"
    capabilities = BenchmarkCapabilities(False, False, False, False, False)
    output_contract = "PAPILLON llm_request, llm_response, response artifacts; team contract HOLD."

    def _hold(self) -> None:
        raise SearchContractError("PUPA_TEAM_AGGREGATION_POLICY_NOT_FROZEN; PUPA_RESPONSIBILITY_POLICY_NOT_FROZEN")

    def format_input(self, item):
        self._hold()

    def parse_member_output(self, raw, item):
        self._hold()

    parse_output = parse_member_output

    def score_member_output(self, parsed, gold):
        self._hold()

    def build_task_feedback(self, parsed, gold):
        self._hold()


def fake_pupa_judge_score(artifacts: PAPILLONArtifacts, *, reference_response: str,
                         pii_str: str, quality_judge: Callable, leakage_judge: Callable,
                         trace: bool = False) -> float | bool:
    """Injected fake-only contract oracle. No real judge/client is constructed."""
    forward = bool(quality_judge(artifacts.response, reference_response))
    reverse = bool(quality_judge(reference_response, artifacts.response))
    leaked = int(leakage_judge(artifacts.llm_request, pii_str))
    return pupa_contract_score(PUPAJudgeOutputs(forward, reverse, leaked), pii_str, trace=trace)
