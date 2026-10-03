"""Isolated held-out accounting preparation: lengths only, no model or scorer."""
import hashlib
import json

from .data_freeze import file_hash
from .experiment_splits import ExperimentSplitReader
from .math_interface import MATH_SOLVER_INTERFACE_V2 as MATH_SOLVER_INTERFACE, solver_interface_contract
from ..governance.token_accounting import serialized_request
from ..search.schemas import SearchContractError
from .math_solver_decoding import generation_request_fields, frozen_solver_policy
from .. import versions


def validation_rows(root, contract, *, context, search_complete_receipt=None):
    if context not in {"ACCOUNTING_DATA_PREP_CONTEXT", "POST_SEARCH_VALIDATION_CONTEXT"}:
        raise SearchContractError("VALIDATION_CONTEXT_FORBIDDEN")
    if context == "POST_SEARCH_VALIDATION_CONTEXT" and not search_complete_receipt:
        raise SearchContractError("SEARCH_COMPLETE_RECEIPT_REQUIRED")
    reader = ExperimentSplitReader(root / contract["canonical_root"], root / contract["split_directory"], "math",
        expected_manifest_sha256=contract["split_manifest_sha256"], expected_protocol=contract["split_version"])
    selected = [r for r in reader.members if r["project_split"] == "validation"]
    if len(selected) != 300 or any(r["source_split"] != "train" for r in selected):
        raise SearchContractError("VALIDATION_SOURCE_ROLE_MISMATCH")
    if contract['identity']==versions.MATH_LOW_COST_EXECUTION_BINDING_VERSION:
        from .math_low_cost import read_subsets
        ids={r['stable_example_id'] for r in read_subsets(root,contract)['memberships']['pilot_validation']}
        selected=[r for r in selected if r['stable_example_id'] in ids]
    source = root / contract["canonical_root"] / "raw/math/train.jsonl"
    expected = next(r for r in reader.manifest["canonical_sources"] if r["name"] == "train")
    if file_hash(source) != expected["canonical_sha256"]:
        raise SearchContractError("CANONICAL_SOURCE_HASH_MISMATCH")
    wanted = {r["source_index"]: r for r in selected}
    found = {}
    with source.open("rb") as stream:
        for index, line in enumerate(stream):
            if index not in wanted:
                continue
            row = json.loads(line)
            if any(row[k] != wanted[index][k] for k in ("stable_example_id", "content_sha256", "input_sha256")):
                raise SearchContractError("VALIDATION_ROW_IDENTITY_MISMATCH")
            # Accounting context cannot return gold, solutions or other fields.
            value = {"stable_example_id": row["stable_example_id"], "input_sha256": row["input_sha256"],
                     "problem": row["content"]["problem"]}
            if context == "POST_SEARCH_VALIDATION_CONTEXT":
                value["reference"] = row["reference_final_answer"]
            found[row["stable_example_id"]] = value
    if len(found) != len(selected):
        raise SearchContractError("VALIDATION_COUNT_MISMATCH")
    return tuple(found[r["stable_example_id"]] for r in selected)


def solver_request(contract, prompt, problem):
    interface = MATH_SOLVER_INTERFACE
    user_content = prompt + "\n\n" + problem
    if contract["identity"] in {versions.MATH_V2_1_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION, versions.MATH_LOW_COST_EXECUTION_BINDING_VERSION}:
        from .math_v21_interface import interface_for_contract, solver_user_content
        interface = interface_for_contract(contract)[0]
        user_content = solver_user_content(contract, prompt, problem)
    return dict(model=contract["models"]["solver"], **generation_request_fields(contract, "solver"),
        messages=[{"role":"system", "content":interface},
                  {"role":"user", "content":user_content}])


def prepare_metadata(root, contract):
    interface = solver_interface_contract()
    if contract["identity"] in {versions.MATH_V2_1_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_DECODING_EXECUTION_BINDING_VERSION, versions.MATH_V2_1_PREDICTION_EXECUTION_BINDING_VERSION, versions.MATH_LOW_COST_EXECUTION_BINDING_VERSION}:
        from .math_v21_interface import interface_for_contract
        interface = interface_for_contract(contract)[1]
    examples = []
    for row in validation_rows(root, contract, context="ACCOUNTING_DATA_PREP_CONTEXT"):
        examples.append(dict(example_id=row["stable_example_id"], input_sha256=row["input_sha256"],
            blank_prompt_serialized_request_bytes=len(serialized_request(solver_request(contract, "", row["problem"])))))
    metadata = dict(identity="MATH_VALIDATION_ACCOUNTING_METADATA_V2", context="ACCOUNTING_DATA_PREP_CONTEXT",
        split_manifest_sha256=contract["split_manifest_sha256"], solver_output_interface=interface,
        decoding=contract["decoding"], validation_raw_rows_read=len(examples),
        model_calls=0, correctness_evaluations=0, content_exposed_to_search=False, examples=examples)
    if frozen_solver_policy(contract) is not None:
        metadata["solver_decoding_policy"] = frozen_solver_policy(contract)
    from .math_prediction_validity import frozen_prediction_policy
    if frozen_prediction_policy(contract):
        metadata["prediction_validity_policy"] = frozen_prediction_policy(contract)
    from .math_prediction_validity import frozen_recovery_policy
    if frozen_recovery_policy(contract):
        metadata['invalid_recovery_policy']=frozen_recovery_policy(contract)
        metadata['low_cost_protocol']=contract['low_cost_protocol']
    return metadata


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
