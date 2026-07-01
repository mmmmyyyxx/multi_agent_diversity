import pytest

from multi_dataset_diverse_rl.cli import build_candidate_eval_pool, build_dataset, select_candidate_eval_batch
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.utils import load_jsonl


def test_legacy_dataset_format_reads_old_fields():
    rows = build_dataset([{"input": "q", "output": "a", "subject": "s"}], "legacy")
    assert rows == [{"question": "q", "answer": "a", "task": "s", "subject": "s"}]


def test_mars_dataset_format_reads_prompt_gold_task():
    rows = build_dataset([{"prompt": "q", "gold": "yes", "task_name": "boolean_expressions"}], "mars")
    assert rows[0]["question"] == "q"
    assert rows[0]["answer"] == "yes"
    assert rows[0]["task"] == "boolean_expressions"
    assert rows[0]["task_name"] == "boolean_expressions"


def test_dataset_format_error_includes_record_index():
    with pytest.raises(ValueError, match="record index 1"):
        build_dataset([{"question": "q", "answer": "a"}, {"question": "missing answer"}], "legacy")


def test_fixed_pool_candidate_eval_is_reproducible():
    cfg = Config(candidate_eval_strategy="fixed_pool", candidate_eval_batch_size=3, candidate_eval_pool_size=5, candidate_eval_repeats=2, seed=7)
    train = [{"question": f"q{i}", "answer": "a"} for i in range(10)]
    pool = build_candidate_eval_pool(train, [], cfg)
    left = select_candidate_eval_batch(train, pool, cfg, epoch=1, step=2)
    right = select_candidate_eval_batch(train, pool, cfg, epoch=1, step=2)
    assert left == right
    assert len(left) == 6
    assert len(pool) == 5


def test_stratified_candidate_eval_samples_multiple_tasks():
    cfg = Config(candidate_eval_strategy="stratified", candidate_eval_batch_size=4, candidate_eval_pool_size=6, candidate_eval_repeats=1, seed=3)
    train = [
        {"question": "a1", "answer": "x", "task": "a"},
        {"question": "a2", "answer": "x", "task": "a"},
        {"question": "b1", "answer": "x", "task": "b"},
        {"question": "b2", "answer": "x", "task": "b"},
        {"question": "c1", "answer": "x", "task": "c"},
        {"question": "c2", "answer": "x", "task": "c"},
    ]
    batch = select_candidate_eval_batch(train, train, cfg, epoch=1, step=1)
    assert len(batch) == 4
    assert len({row["task"] for row in batch}) >= 3


def test_dataset_format_csv_can_be_loaded_directly():
    raw = load_jsonl("Dataset_format/BBH/boolean_expressions.csv", limit=2)
    rows = build_dataset(raw, "legacy")
    assert len(rows) == 2
    assert rows[0]["question"]
    assert rows[0]["answer"] in {"True", "False"}
