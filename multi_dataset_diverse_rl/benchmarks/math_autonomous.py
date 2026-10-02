"""Execution-only accounting binding over the immutable MATH V1.2 science."""
from copy import deepcopy
import hashlib
import json

from .math_execution import MATHExecutionBinding
from ..governance.token_accounting import POLICY
from .. import versions


class MATHAutonomousBinding(MATHExecutionBinding):
    def blockers(self):
        c = self.contract
        try:
            if c["identity"] != versions.MATH_AUTONOMOUS_EXECUTION_BINDING_VERSION:
                return ("MATH_AUTONOMOUS_BINDING_MISMATCH",)
            parent = self.path(c["parent_binding_path"])
            if hashlib.sha256(parent.read_bytes()).hexdigest() != c["parent_binding_sha256"]:
                return ("MATH_PARENT_BINDING_MISMATCH",)
            original = json.loads(parent.read_bytes())
            if MATHExecutionBinding(self.root, original).blockers():
                return ("MATH_PARENT_SCIENCE_NOT_READY",)
            policy = self.path(c["accounting_policy_path"])
            if (hashlib.sha256(policy.read_bytes()).hexdigest() != c["accounting_policy_sha256"]
                    or json.loads(policy.read_bytes()) != POLICY):
                return ("MATH_ACCOUNTING_POLICY_MISMATCH",)
            metadata = self.path(c["validation_accounting_metadata_path"])
            if hashlib.sha256(metadata.read_bytes()).hexdigest() != c["validation_accounting_metadata_sha256"]:
                return ("VALIDATION_ACCOUNTING_METADATA_MISMATCH",)
            value = json.loads(metadata.read_bytes())
            if (value["split_manifest_sha256"] != c["split_manifest_sha256"]
                    or value["solver_output_interface"] != c["solver_output_interface"]
                    or value["decoding"] != c["decoding"] or len(value["examples"]) != 300
                    or value["context"] != "ACCOUNTING_DATA_PREP_CONTEXT"):
                return ("VALIDATION_ACCOUNTING_METADATA_BINDING_MISMATCH",)
            selected = [r for r in self.reader().members if r["project_split"] == "validation"]
            if ([(r["example_id"],r["input_sha256"]) for r in value["examples"]] !=
                    [(r["stable_example_id"],r["input_sha256"]) for r in selected]
                    or any(type(r["blank_prompt_serialized_request_bytes"]) is not int or
                           r["blank_prompt_serialized_request_bytes"] <= 0 for r in value["examples"])):
                return ("VALIDATION_ACCOUNTING_MEMBERSHIP_MISMATCH",)
            if c["execution_phase"] not in {"canary", "pilot"}:
                return ("MATH_AUTONOMOUS_PHASE_MISMATCH",)
            if not c["execution_attempt_id"].startswith("math_unified_v2_A1_seed81_"):
                return ("MATH_AUTONOMOUS_ATTEMPT_MISMATCH",)
            if c["cache_namespace"] != c["execution_attempt_id"]:
                return ("MATH_AUTONOMOUS_FRESH_CACHE_MISMATCH",)
            # Only execution identity, accounting and operational phase/ceilings
            # may differ. Initial prompts, evaluator, models and search physics
            # are compared directly against the immutable parent bytes.
            projected = deepcopy(c)
            extras = {"binding_path", "parent_binding_path", "parent_binding_sha256",
                "accounting_policy_path", "accounting_policy_sha256", "token_ledger_directory",
                "validation_accounting_metadata_path", "validation_accounting_metadata_sha256",
                "execution_phase", "execution_attempt_id", "task_authorization_sha256"}
            for key in extras:
                projected.pop(key, None)
            for key in ("identity", "cache_namespace", "canary_attempt_id", "provider_bounds"):
                projected[key] = original[key]
            if projected != original:
                return ("MATH_AUTONOMOUS_SCIENTIFIC_PAYLOAD_CHANGED",)
            if c["execution_phase"] == "canary" and c["provider_bounds"] != original["provider_bounds"]:
                return ("CANARY_OPERATIONAL_CEILING_MISMATCH",)
            if c["execution_phase"] == "pilot" and c["provider_bounds"] != {
                **original["provider_bounds"], "max_opportunities":30_000_000,
                "solver_calls":30_000_000, "reflection_calls":30_000_000,
                "pattern_calls":0, "successful_provider_calls":30_000_000,
                "transport_attempts":30_000_000}:
                return ("PILOT_OPERATIONAL_CEILING_MISMATCH",)
            ledger = self.path(c["token_ledger_directory"])
            if not ledger.is_relative_to(self.root / "runs") or len(c["task_authorization_sha256"]) != 64:
                return ("MATH_TOKEN_LEDGER_SCOPE_MISMATCH",)
            return ()
        except (OSError, KeyError, ValueError, TypeError):
            return ("MATH_AUTONOMOUS_BINDING_INVALID",)
