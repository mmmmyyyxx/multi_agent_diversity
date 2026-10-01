"""Offline package and resource identity verification for pinned IFBench checks."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

from .data_freeze import digest, file_hash
from ..search.schemas import SearchContractError

LOCK = Path(__file__).with_name("ifbench_runtime_lock.json")


def tree_identity(root):
    rows = [{"path": p.relative_to(root).as_posix(), "sha256": file_hash(p)}
            for p in sorted(root.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")]
    if not rows:
        raise ValueError("EVALUATOR_RESOURCE_EMPTY")
    return digest(rows)


def verify_ifbench_runtime():
    try:
        lock = json.loads(LOCK.read_bytes())
        if lock["identity"] != "ifbench_checkers_runtime_v1":
            raise ValueError("EVALUATOR_LOCK_IDENTITY_MISMATCH")
        if ".".join(map(str, sys.version_info[:2])) != lock["python_minor"]:
            raise ValueError("EVALUATOR_PYTHON_IDENTITY_MISMATCH")
        for package, expected in lock["packages"].items():
            if importlib.metadata.version(package) != expected:
                raise ValueError("EVALUATOR_PACKAGE_IDENTITY_MISMATCH")
        spec = importlib.util.find_spec("en_core_web_sm")
        if spec is None or not spec.submodule_search_locations:
            raise ValueError("IFBENCH_LOCAL_SPACY_RESOURCE_MISSING")
        if tree_identity(Path(next(iter(spec.submodule_search_locations)))) != lock["spacy_model_sha256"]:
            raise ValueError("EVALUATOR_SPACY_RESOURCE_IDENTITY_MISMATCH")
        # An explicit locator prevents ambient user caches becoming authoritative.
        root = Path(os.environ["NLTK_DATA"])
        for resource in lock["nltk_resources"]:
            if tree_identity(root / resource["relative_path"]) != resource["tree_sha256"]:
                raise ValueError("EVALUATOR_NLTK_RESOURCE_IDENTITY_MISMATCH")
        import nltk
        nltk.data.path[:] = [str(root)]
        from ._vendor.ifbench.offline import require_nltk_resource, require_spacy_model
        require_spacy_model("en_core_web_sm")
        for resource in lock["nltk_resources"]:
            require_nltk_resource(resource["name"])
        return {"verified": True, "identity": lock["identity"], "lock_sha256": file_hash(LOCK), "provider_attempts": 0}
    except (OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError, RuntimeError) as exc:
        raise SearchContractError("IFBENCH_LOCAL_EVALUATOR_DEPENDENCIES_NOT_FROZEN: RUNTIME_IDENTITY_MISMATCH") from exc
