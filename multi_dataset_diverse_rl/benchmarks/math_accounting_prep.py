"""Isolated held-out accounting preparation: lengths only, no model or scorer."""
import hashlib
import json

from .data_freeze import file_hash
from .experiment_splits import ExperimentSplitReader
from .math_v21_interface import interface_for_contract, solver_user_content
from ..governance.token_accounting import serialized_request
from ..search.schemas import SearchContractError
from .math_solver_decoding import generation_request_fields, frozen_solver_policy
from .. import versions


def validation_rows(root,contract,*,context,search_complete_receipt=None):
    raise SearchContractError('CURRENT_HELDOUT_ACCOUNTING_READ_NOT_AUTHORIZED')


def solver_request(contract,prompt,problem):
    return dict(model=contract['models']['solver'],**generation_request_fields(contract,'solver'),
        messages=[dict(role='system',content=interface_for_contract(contract)[0]),
            dict(role='user',content=solver_user_content(contract,prompt,problem))])

def prepare_metadata(root,contract):
    raise SearchContractError('CURRENT_REUSE_FROZEN_LENGTH_METADATA_ONLY')


def escaped_prompt_bytes(prompt):
    return len(json.dumps(prompt, ensure_ascii=False)[1:-1].encode("utf-8"))


class ValidationReserve:
    def __init__(self, metadata, initial_prompts):
        self.metadata = metadata
        self.initial_sizes = tuple(escaped_prompt_bytes(p) for p in initial_prompts)
        self.max_prompt_size = max(self.initial_sizes)

    def observe(self, prompt):
        self.max_prompt_size = max(self.max_prompt_size, escaped_prompt_bytes(prompt))

    def cost(self, sizes):
        cap = self.metadata["decoding"].get("solver_max_output_tokens", self.metadata["decoding"]["max_output_tokens"])
        return sum(r["blank_prompt_serialized_request_bytes"] + n + 4096 + cap
                   for r in self.metadata["examples"] for n in sizes)

    def remaining(self):
        return self.cost(self.initial_sizes) + self.cost((self.max_prompt_size,) * 5)
