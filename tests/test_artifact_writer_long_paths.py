"""Zero-provider regression for the aborted transfer diagnostic persistence path."""

from __future__ import annotations

import json
import os

import pytest

from multi_dataset_diverse_rl.persistence.artifacts import ArtifactWriter


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH regression")
def test_atomic_json_write_to_long_evaluation_identity_path(tmp_path):
    root = tmp_path / ("a" * max(12, 170 - len(str(tmp_path))))
    writer = ArtifactWriter(root)
    relative = "team_full_categorical_profiles/" + "b" * 64 + ".json"
    target = root / relative
    assert len(str(target.resolve())) > 260
    writer.write_json(relative, {"evaluation_identity": "b" * 64})
    long_target = "\\\\?\\" + str(target.resolve())
    with open(long_target, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload == {
        "evaluation_identity": "b" * 64,
    }
