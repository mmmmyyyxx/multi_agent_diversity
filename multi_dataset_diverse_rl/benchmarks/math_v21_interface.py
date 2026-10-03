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


MATH_SOLVER_INTERFACE_V4_USER_SUFFIX = (
    "\n\nMandatory response format: reply with exactly one line beginning with "
    "the literal prefix FINAL_ANSWER: followed by the mathematical answer. "
    "Write the complete answer on that same line. Do not output reasoning, "
    "Markdown, or any text before or after that line."
)


def v4_interface_contract():
    return dict(identity=versions.MATH_SOLVER_INTERFACE_V4_VERSION,
        sha256=hashlib.sha256((MATH_SOLVER_INTERFACE_V3 + MATH_SOLVER_INTERFACE_V4_USER_SUFFIX).encode()).hexdigest(),
        system_sha256=hashlib.sha256(MATH_SOLVER_INTERFACE_V3.encode()).hexdigest(),
        user_suffix_sha256=hashlib.sha256(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX.encode()).hexdigest(),
        parser_identity="math_verify_no_fallback_v1")


def v5_interface_contract():
    return dict(v4_interface_contract(), identity=versions.MATH_SOLVER_INTERFACE_V5_VERSION,
        solver_max_output_tokens=3600, reflection_max_output_tokens=1800)


def interface_for_contract(contract):
    frozen = contract["solver_output_interface"]
    if frozen == solver_interface_contract():
        return MATH_SOLVER_INTERFACE_V2, solver_interface_contract()
    if frozen == v3_interface_contract():
        return MATH_SOLVER_INTERFACE_V3, v3_interface_contract()
    if frozen == v4_interface_contract():
        return MATH_SOLVER_INTERFACE_V3, v4_interface_contract()
    if frozen == v5_interface_contract():
        return MATH_SOLVER_INTERFACE_V3, v5_interface_contract()
    raise SearchContractError("MATH_SOLVER_INTERFACE_BINDING_MISMATCH")


def solver_user_content(contract, prompt, problem):
    interface_for_contract(contract)
    suffix = MATH_SOLVER_INTERFACE_V4_USER_SUFFIX if contract["solver_output_interface"] in (v4_interface_contract(), v5_interface_contract()) else ""
    return prompt + "\n\n" + problem + suffix


class MATHV21BenchmarkAdapter(MATHBenchmarkAdapterV2):
    def __init__(self, contract):
        self._contract = contract
        self.output_contract, self._interface_contract = interface_for_contract(contract)
        from .math_prediction_validity import frozen_prediction_policy
        self.prediction_validity_policy = frozen_prediction_policy(contract)
        self.invalid_predictions_are_incorrect = self.prediction_validity_policy is not None
        if self.invalid_predictions_are_incorrect:
            from .protocols import MATH_PROTOCOL_V3
            self.protocol = MATH_PROTOCOL_V3

    def prediction_result(self, result):
        from .math_prediction_validity import classify_prediction
        return classify_prediction(result["text"], result.get("finish_reason"))

    def parse_member_output(self, raw, item):
        if not self.invalid_predictions_are_incorrect:
            return super().parse_member_output(raw, item)
        from .math_prediction_validity import classify_prediction, prediction_from_persisted
        from ..search.schemas import ParsedOutput
        if item.benchmark_id != self.benchmark_id:
            return ParsedOutput("", False)
        result = classify_prediction(raw) if isinstance(raw, str) else prediction_from_persisted(raw)
        return result.parsed()

    parse_output = parse_member_output

    def solver_interface_contract(self):
        return dict(self._interface_contract)

    def solver_user_content(self, prompt, item):
        return solver_user_content(self._contract, prompt, self.format_input(item))
