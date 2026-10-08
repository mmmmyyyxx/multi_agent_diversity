"""Current V6 visible-solution interface; shared mathematical scoring is unchanged."""
from .math_interface import MATH_SOLVER_INTERFACE_V6, v6_interface_contract

from .math_domain_v2 import MATHBenchmarkAdapterV2
from ..search.schemas import SearchContractError
from .. import versions


def interface_for_contract(contract):
    from ..search.current_policy import require_current_contract
    require_current_contract(contract)
    if contract.get('solver_output_interface')!=v6_interface_contract():
        raise SearchContractError('MATH_SOLVER_INTERFACE_BINDING_MISMATCH')
    return MATH_SOLVER_INTERFACE_V6,v6_interface_contract()

def solver_user_content(contract,prompt,problem):
    interface_for_contract(contract)
    return prompt+'\n\n'+problem


class MATHV21BenchmarkAdapter(MATHBenchmarkAdapterV2):
    def __init__(self, contract):
        self._contract = contract
        self.output_contract, self._interface_contract = interface_for_contract(contract)
        from .math_visible_trajectory import frozen_trajectory_policy
        self.solver_trajectory_policy = frozen_trajectory_policy(contract)
        from .math_prediction_validity import frozen_prediction_policy
        self.prediction_validity_policy = frozen_prediction_policy(contract)
        self.invalid_predictions_are_incorrect = self.prediction_validity_policy is not None
        if self.invalid_predictions_are_incorrect:
            from .protocols import MATH_PROTOCOL_V3
            self.protocol = MATH_PROTOCOL_V3
    def prediction_result(self, result):
        from .math_prediction_validity import classify_prediction, prediction_from_persisted
        if 'resolved_prediction' in result:
            return prediction_from_persisted(result['resolved_prediction'])
        return classify_prediction(result["text"], result.get("finish_reason"))

    def parse_member_output(self, raw, item):
        if not self.invalid_predictions_are_incorrect:
            return super().parse_member_output(raw, item)
        from .math_prediction_validity import classify_prediction, prediction_from_persisted
        from ..search.schemas import ParsedOutput
        if item.benchmark_id != self.benchmark_id:
            return ParsedOutput("", False)
        if isinstance(raw, dict) and raw.get('schema') == versions.MATH_SOLVER_PROFILE_VERSION:
            if self.solver_trajectory_policy is None:
                raise SearchContractError('MATH_VISIBLE_PROFILE_REQUIRES_V6')
            from .math_visible_trajectory import profile_prediction
            return profile_prediction(raw, example_id=item.input_id).parsed()
        result = classify_prediction(raw) if isinstance(raw, str) else prediction_from_persisted(raw)
        return result.parsed()

    parse_output = parse_member_output

    def solver_interface_contract(self):
        return dict(self._interface_contract)

    def solver_user_content(self, prompt, item):
        return solver_user_content(self._contract, prompt, self.format_input(item))
