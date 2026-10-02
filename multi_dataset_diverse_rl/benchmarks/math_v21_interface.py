"""Versioned output-only repair; mathematical parsing and scoring are unchanged."""
import hashlib

from .math_interface import MATH_SOLVER_INTERFACE_V2, solver_interface_contract
from .math_domain_v2 import MATHBenchmarkAdapterV2
from ..search.schemas import SearchContractError
from .. import versions


MATH_SOLVER_INTERFACE_V3 = (
    "Solve the mathematical problem using the supplied decision procedure. "
    "Return exactly one nonempty line in this literal format:\n"
    "FINAL_ANSWER: <answer>\n"
    "Replace <answer> with the mathematical answer. The literal prefix "
    "FINAL_ANSWER: is mandatory. Do not output reasoning, headings, commentary, "
    "Markdown formatting, code fences, or additional lines. Do not put quotation "
    "marks, bold markers, or backticks around the line. Keep the entire answer "
    "payload on the same line as the prefix."
)


def v3_interface_contract():
    return dict(identity=versions.MATH_SOLVER_INTERFACE_V3_VERSION,
        sha256=hashlib.sha256(MATH_SOLVER_INTERFACE_V3.encode()).hexdigest(),
        parser_identity="math_verify_no_fallback_v1")


def interface_for_contract(contract):
    frozen = contract["solver_output_interface"]
    if frozen == solver_interface_contract():
        return MATH_SOLVER_INTERFACE_V2, solver_interface_contract()
    if frozen == v3_interface_contract():
        return MATH_SOLVER_INTERFACE_V3, v3_interface_contract()
    raise SearchContractError("MATH_SOLVER_INTERFACE_BINDING_MISMATCH")


class MATHV21BenchmarkAdapter(MATHBenchmarkAdapterV2):
    def __init__(self, contract):
        self.output_contract, self._interface_contract = interface_for_contract(contract)

    def solver_interface_contract(self):
        return dict(self._interface_contract)
