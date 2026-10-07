"""Unfrozen V2.2 profile; no prior binding or authority is inherited."""
from ..current_contract import MATH_V2_2_EXECUTION_BINDING_VERSION
from ..search.schemas import SearchContractError


class MATHGradientPatternBinding:
    def __init__(self, root, contract):
        if contract.get('identity') != MATH_V2_2_EXECUTION_BINDING_VERSION:
            raise SearchContractError('CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN')
        self.root, self.contract = root.resolve(), contract

    def blockers(self):
        return ('CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN',)

    def method(self, arm):
        raise SearchContractError(self.blockers()[0])

    def compose(self, **kwargs):
        raise SearchContractError(self.blockers()[0])

    def examples(self, role):
        raise SearchContractError(self.blockers()[0])
