import pytest

from multi_dataset_diverse_rl.cli import build_dataset
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.utils import load_jsonl


def test_legacy_dataset_format_reads_old_fields_and_metadata():
    rows = build_dataset([{"input": "q", "output": "a", "subject": "s"}], "legacy")
    assert rows == [{"question": "q", "answer": "a", "task": "s", "subject": "s"}]


def test_mars_dataset_format_reads_prompt_gold_task():
    rows = build_dataset([{"prompt": "q", "gold": "yes", "task_name": "boolean_expressions"}], "mars")
    assert rows[0] == {
        "question": "q", "answer": "yes", "task": "boolean_expressions",
        "task_name": "boolean_expressions",
    }


def test_dataset_format_error_includes_record_index():
    with pytest.raises(ValueError, match="record 1"):
        build_dataset([{"question": "q", "answer": "a"}, {"question": "missing answer"}], "legacy")


def test_fixed_optimization_probe_contains_only_supplied_train_rows(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path), candidate_eval_pool_size=2)
    system = PromptEnsembleOptimizationSystem(cfg, solver=lambda *_args: None)
    train = [{"question": "train-1", "answer": "a"}, {"question": "train-2", "answer": "b"}]
    probe = system.build_probe(train)
    assert [row.question for row in probe.examples] == ["train-1", "train-2"]


def test_stage_a_indices_are_one_shared_deterministic_pool(tmp_path):
    cfg = Config.from_flat(out_dir=str(tmp_path))
    system = PromptEnsembleOptimizationSystem(cfg, solver=lambda *_args: None)
    system.fixed_probe = system.build_probe([{"question": f"q{i}", "answer": "A"} for i in range(3)])
    # Active profiles are populated directly because this test checks only pool identity.
    from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
    system.active_profiles = [tuple(PromptAnswer("B", "trace", True) for _ in range(3)) for _ in range(5)]
    system.initial_profiles = list(system.active_profiles)
    assigned = {system.fixed_probe.examples[0].question_hash}
    assert system.stage_a_indices(0, assigned) == system.stage_a_indices(0, assigned)


def test_round_robin_peer_state_ablation_builds_global_c0_pool(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        experiment_setting="shared_peer_state_vote_first",
        stage_a_representative_size=0,
        stage_a_coverage_size=2,
        stage_a_conversion_size=0,
        stage_a_preservation_size=0,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=lambda *_args: None)
    system.fixed_probe = system.build_probe([{"question": f"q{i}", "answer": "A"} for i in range(3)])
    from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
    system.active_profiles = [tuple(PromptAnswer("B", "trace", True) for _ in range(3)) for _ in range(5)]
    system.initial_profiles = list(system.active_profiles)
    assert len(system.stage_a_indices(0, set())) == 2


def test_representative_pool_uses_seeded_question_hash_not_file_order(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        experiment_setting="shared_peer_state_vote_first",
        stage_a_representative_size=2,
        stage_a_coverage_size=0,
        stage_a_conversion_size=0,
        stage_a_preservation_size=0,
        seed=43,
    )
    system = PromptEnsembleOptimizationSystem(cfg, solver=lambda *_args: None)
    system.fixed_probe = system.build_probe([{"question": f"q{i}", "answer": "A"} for i in range(5)])
    from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
    system.active_profiles = [tuple(PromptAnswer("B", "trace", True) for _ in range(5)) for _ in range(5)]
    system.initial_profiles = list(system.active_profiles)
    selected = system.stage_a_indices(0, set())
    assert selected == system._representative_indices(2)
    assert selected != [0, 1]


def test_dataset_format_csv_can_be_loaded_directly():
    raw = load_jsonl("Dataset_format/BBH/boolean_expressions.csv", limit=2)
    rows = build_dataset(raw, "legacy")
    assert len(rows) == 2
    assert rows[0]["question"]
    assert rows[0]["answer"] in {"True", "False"}
