"""Complete editable System Message; benchmark formatting adds no instruction."""
from .math_domain_v2 import MATHBenchmarkAdapterV2
from ..search.system_prompt import SEED, solver_messages, IDENTITY, CONTRACT
from ..search.schemas import ParsedOutput, SearchContractError
from .. import versions
from .math_flexible_answer import POLICY, IDENTITY as PARSER, classify_prediction, prediction_from_persisted, extract_answer

SYSTEM_PROMPT_POLICY = dict(identity=IDENTITY, blocks=['role','strategy','answer'],
    serialization='ordered_role_strategy_answer_ascii_json_compact',
    rendering='role_double_newline_strategy_double_newline_answer',
    mutation=CONTRACT, single_block_per_generation=True, max_rendered_characters=3000,
    seed=SEED.to_dict(), user_message='raw_problem_only', extra_solver_instructions=False,
    deduplication='canonical_structured_bytes', block_hash='sha256_utf8',
    full_hash='sha256_canonical_structured_bytes', system_hash='sha256_rendered_utf8')


def system_prompt_policy(team_version=None):
    from copy import deepcopy
    from ..current_contract import MATH_INITIAL_TEAM_VERSION
    from .math_canary_inputs import initial_prompt
    policy = deepcopy(SYSTEM_PROMPT_POLICY)
    policy['seed'] = initial_prompt(team_version or MATH_INITIAL_TEAM_VERSION).to_dict()
    return policy


def system_interface_contract(team_version=None):
    return dict(identity=versions.MATH_SOLVER_INTERFACE_V9_VERSION,
        parser_identity=PARSER, system_prompt_policy=system_prompt_policy(team_version),
        answer_extraction_policy=POLICY, solver_max_output_tokens=3600, reflection_max_output_tokens=1800)


class MATHStructuredSystemBenchmark(MATHBenchmarkAdapterV2):
    parser_identity = PARSER
    invalid_predictions_are_incorrect = True
    final_payload = staticmethod(lambda raw: extract_answer(raw)[0] or None)

    def __init__(self, contract):
        from ..search.current_policy import require_current_contract
        from .math_response_evidence import frozen_trajectory_policy
        from .math_prediction_validity import frozen_prediction_policy
        from .protocols import MATH_PROTOCOL_FLEXIBLE_V7
        require_current_contract(contract)
        if contract.get('solver_output_interface') != system_interface_contract(contract.get('initial_team_version')):
            raise SearchContractError('MATH_STRUCTURED_INTERFACE_BINDING_MISMATCH')
        self._contract = contract
        # Metadata identity only; it is never sent as a Solver instruction.
        self.output_contract = versions.MATH_SOLVER_INTERFACE_V9_VERSION
        self.protocol = MATH_PROTOCOL_FLEXIBLE_V7
        self.solver_trajectory_policy = frozen_trajectory_policy(contract)
        self.prediction_validity_policy = frozen_prediction_policy(contract)

    def solver_interface_contract(self):
        return system_interface_contract(self._contract.get('initial_team_version'))

    def solver_user_content(self, prompt, item):
        return solver_messages(prompt, self.format_input(item))[1]['content']

    def prediction_result(self, result):
        if 'resolved_prediction' in result:
            return prediction_from_persisted(result['resolved_prediction'])
        return classify_prediction(result['text'], result.get('finish_reason'))

    def parse_member_output(self, raw, item):
        if item.benchmark_id != self.benchmark_id:
            return ParsedOutput('', False)
        if isinstance(raw, dict) and raw.get('schema') == versions.MATH_FLEXIBLE_PROFILE_VERSION:
            from .math_response_evidence import profile_prediction
            return profile_prediction(raw, example_id=item.input_id).parsed()
        prediction = classify_prediction(raw) if isinstance(raw, str) else prediction_from_persisted(raw)
        return prediction.parsed()

    parse_output = parse_member_output
