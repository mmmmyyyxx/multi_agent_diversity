"""Pinned checker registry admitted only after offline resource lock verification."""


def load_pinned_checkers():
    from .evaluator_resources import verify_ifbench_runtime
    verify_ifbench_runtime()
    from ._vendor.ifbench.instructions_registry import INSTRUCTION_DICT
    return INSTRUCTION_DICT
