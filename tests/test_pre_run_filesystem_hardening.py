"""Native-Windows long-path rehearsal of all active artifact operations."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil

import pytest

from multi_dataset_diverse_rl.persistence.artifacts import ArtifactWriter
from multi_dataset_diverse_rl.persistence.durable_io import io_path


def _padded_root(tmp_path: Path, desired_target_length: int) -> Path:
    root = tmp_path / "sp ace Ω"
    relative = Path("nested") / ("a" * 64 + ".json")
    while len(str(root / relative)) < desired_target_length:
        remaining = desired_target_length - len(str(root / relative))
        root /= "p" * min(80, max(1, remaining - 1))
    return root


@pytest.mark.skipif(os.name != "nt", reason="native Windows MAX_PATH matrix")
@pytest.mark.parametrize("target_length", [259, 260, 261, 320, 512])
def test_all_artifact_writer_operations_at_realistic_long_paths(tmp_path, target_length):
    root = _padded_root(tmp_path, target_length)
    relative = "nested/" + "a" * 64 + ".json"
    assert len(str(root / relative)) >= target_length
    try:
        writer = ArtifactWriter(root)
        writer.write_json(relative, {"value": 1})
        writer.write_json(relative, {"value": 2})
        with open(io_path(root / relative), encoding="utf-8") as handle:
            assert json.load(handle) == {"value": 2}

        writer.write_jsonl("nested/rows.jsonl", [{"row": 1}])
        writer.write_jsonl("nested/rows.jsonl", [{"row": 2}])
        writer.append_jsonl("nested/rows.jsonl", [{"row": 3}])
        with open(io_path(root / "nested/rows.jsonl"), encoding="utf-8") as handle:
            assert [json.loads(line) for line in handle] == [{"row": 2}, {"row": 3}]

        writer.write_csv("nested/rows.csv", [{"row": 1}], ["row"])
        writer.write_csv("nested/rows.csv", [{"row": 2}], ["row"])
        with open(io_path(root / "nested/rows.csv"), newline="", encoding="utf-8") as handle:
            assert list(csv.DictReader(handle)) == [{"row": "2"}]
        with os.scandir(io_path(root / "nested")) as entries:
            assert all(not entry.name.endswith(".tmp") for entry in entries)
    finally:
        if os.path.exists(io_path(root)):
            shutil.rmtree(io_path(root))
    assert not os.path.exists(io_path(root))
