from pathlib import Path

import yaml

from multi_dataset_diverse_rl.governance.manifest import preregistration_hash
from scripts.run_v18_hybrid_online_accumulation import _config
from scripts.v18_no_semantic_critic_online import _registry_model


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/manifests/v18_qwen3_8b_no_semantic_critic_light_replication.yaml"


def test_model_override_is_explicit_and_defaults_remain_historical(tmp_path):
    class Task:
        task_type = "multiple_choice"
        task_id = "disambiguation_qa"
        benchmark = "bbh"
        answer_format = "choice"
        train_path = "data/train.jsonl"
        val_path = "data/val.jsonl"
        test_path = "data/test.jsonl"

    default = _config(task=Task(), seed=71, run_dir=tmp_path / "a", cache_path=tmp_path / "a.sqlite")
    assert default.models.agent_model == "qwen3-14b"
    assert default.models.optimizer_model == "qwen3-14b"
    assert default.models.evaluator_model == "qwen3-14b"

    switched = _config(
        task=Task(),
        seed=71,
        run_dir=tmp_path / "b",
        cache_path=tmp_path / "b.sqlite",
        agent_model="qwen3-8b",
        optimizer_model="qwen3.7-flash",
        evaluator_model="qwen3.7-flash",
    )
    assert switched.models.agent_model == "qwen3-8b"
    assert switched.models.optimizer_model == "qwen3.7-flash"
    assert switched.models.evaluator_model == "qwen3.7-flash"


def test_light_replication_manifest_is_one_seed_two_arm_and_api_disabled():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["seeds"] == [71]
    assert manifest["budget"]["limit"] == {"updates_per_seed_arm": 8, "trajectories": 2}
    assert manifest["model"]["solver"] == "qwen3-8b"
    assert set(manifest["model"]["optimizer_roles"].values()) == {"qwen3.7-flash"}
    assert manifest["api_authorization"]["authorized"] is False
    assert manifest["data"]["test_policy"].startswith("prohibited")


def test_registry_model_keeps_legacy_fallback_and_new_mapping():
    assert _registry_model({}) == {
        "solver": "qwen3-14b",
        "teacher": "qwen3-14b",
        "critic": "qwen3-14b",
        "student": "qwen3-14b",
        "thinking": False,
    }
    expected = {
        "solver": "qwen3-8b",
        "teacher": "qwen3.7-flash",
        "critic": "qwen3.7-flash",
        "student": "qwen3.7-flash",
        "thinking": False,
    }
    assert _registry_model({"model": expected}) == expected


def test_preregistration_hash_placeholder_is_replaced():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["preregistration"]["sha256"] == preregistration_hash(manifest)
