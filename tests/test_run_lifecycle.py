from __future__ import annotations

import asyncio
import json

import pytest

from multi_dataset_diverse_rl.governance.run_lifecycle import run_with_lifecycle


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_fake_provider_transitions_running_to_complete(tmp_path):
    lifecycle = tmp_path / "run_lifecycle.json"
    observed = []

    async def fake_provider():
        observed.append(_read(lifecycle)["status"])
        return "ok"

    result = asyncio.run(run_with_lifecycle(
        lifecycle, identity={"experiment_id": "fake"}, operation=fake_provider,
    ))
    terminal = _read(lifecycle)
    assert result == "ok"
    assert observed == ["RUNNING"]
    assert terminal["status"] == "EXECUTION_COMPLETE"
    assert [row["status"] for row in terminal["history"]] == ["RUNNING", "EXECUTION_COMPLETE"]


def test_fake_provider_transitions_running_to_aborted(tmp_path):
    lifecycle = tmp_path / "run_lifecycle.json"
    observed = []

    async def fake_provider():
        observed.append(_read(lifecycle)["status"])
        raise ConnectionError("fake transport failure")

    with pytest.raises(ConnectionError, match="fake transport failure"):
        asyncio.run(run_with_lifecycle(
            lifecycle, identity={"experiment_id": "fake"}, operation=fake_provider,
        ))
    terminal = _read(lifecycle)
    assert observed == ["RUNNING"]
    assert terminal["status"] == "EXECUTION_ABORTED"
    assert terminal["error_type"] == "ConnectionError"
    assert [row["status"] for row in terminal["history"]] == ["RUNNING", "EXECUTION_ABORTED"]
