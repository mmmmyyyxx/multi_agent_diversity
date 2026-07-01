import json
from argparse import Namespace

from scripts.compute_experiment_metrics import analyze_run
from scripts.experiment_config import DEFAULT_SEED_BASELINES, DEFAULT_EXPERIMENT_SETTINGS, dataset_paths_from_args, select_settings, setting_names
from scripts.run_experiments import SETTINGS, _selected_settings


def test_run_experiments_default_settings_include_baselines_and_guarded_beams():
    names = [setting.name for setting in SETTINGS]
    assert names == [
        "shared_baseline",
        "bank_baseline",
        "shared_guarded_beam",
        "bank_guarded_beam",
    ]
    assert {setting.name: setting.reward_mode for setting in SETTINGS}["shared_guarded_beam"] == "guarded_diversity"
    assert {setting.name: setting.reward_mode for setting in SETTINGS}["bank_guarded_beam"] == "guarded_diversity"
    assert SETTINGS == DEFAULT_EXPERIMENT_SETTINGS


def test_run_experiments_parser_seeds_baselines_by_default():
    assert DEFAULT_SEED_BASELINES == 1
    assert any(setting.baseline_only for setting in SETTINGS)


def test_selected_settings_filters_by_name():
    selected = _selected_settings("shared_baseline,bank_guarded_beam")
    assert [setting.name for setting in selected] == ["shared_baseline", "bank_guarded_beam"]
    assert select_settings("shared_baseline,bank_guarded_beam") == selected


def test_dataset_paths_use_dataset_specific_defaults():
    args = Namespace(
        mmlu_train_path="mmlu_train.jsonl",
        mmlu_val_path="mmlu_val.jsonl",
        mmlu_test_path="mmlu_test.jsonl",
        bbh_train_path="bbh_train.jsonl",
        bbh_val_path="bbh_val.jsonl",
        bbh_test_path="bbh_test.jsonl",
        task_type="auto",
        train_path="train.jsonl",
        val_path="val.jsonl",
        test_path="test.jsonl",
    )
    assert dataset_paths_from_args(args, "mmlu") == {
        "task_type": "mmlu",
        "train": "mmlu_train.jsonl",
        "val": "mmlu_val.jsonl",
        "test": "mmlu_test.jsonl",
    }
    assert dataset_paths_from_args(args, "bbh") == {
        "task_type": "bbh",
        "train": "bbh_train.jsonl",
        "val": "bbh_val.jsonl",
        "test": "bbh_test.jsonl",
    }


def test_analyze_experiments_uses_shared_setting_order():
    import scripts.analyze_experiments as analyze_experiments

    assert analyze_experiments.SETTINGS == setting_names(DEFAULT_EXPERIMENT_SETTINGS)


def test_compute_metrics_reads_vote_tie_rate_and_mars_delta(tmp_path):
    run_dir = tmp_path / "mmlu" / "shared_guarded_beam_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "reward_mode": "guarded_diversity",
                    "baseline_only": False,
                    "init_mode": "shared",
                    "agents": 5,
                    "epochs": 1,
                    "train_size": 2,
                    "test_size": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "test": {
                        "vote_acc": 0.75,
                        "vote_tie_rate": 0.25,
                        "mean_embedding_diversity": 0.4,
                        "mean_invalid_rate": 0.1,
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    row = analyze_run(run_dir, {"mmlu": {"vote_acc": 0.7, "embedding_diversity": 0.3}})
    assert row["dataset"] == "mmlu"
    assert row["setting"] == "shared_guarded_beam"
    assert row["vote_tie_rate"] == 0.25
    assert round(row["vs_mars_delta_acc"], 6) == 0.05
    assert round(row["vs_mars_delta_diversity"], 6) == 0.1
