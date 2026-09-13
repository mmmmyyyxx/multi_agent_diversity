from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_agents_defines_two_layer_architecture_and_role_boundary() -> None:
    agents = _read("AGENTS.md")
    assert "## Current active research architecture" in agents
    assert "Layer 1 — Local Prompt Optimizer" in agents
    assert "Layer 2 — Team-Level Responsibility/Search Controller" in agents
    assert "The active research direction" in agents
    assert "canonical historical" in agents
    assert "## Agent execution model" in agents
    assert "GPT-5.6 Sol" in agents
    assert "GPT-5.6 Luna" in agents
    assert "Luna reports what happened; Sol decides what it means." in agents
    assert "LUNA_DISPATCH_UNAVAILABLE" in agents


def test_workflow_assigns_every_phase_and_keeps_scientific_work_with_sol() -> None:
    workflow = _read("docs/workflows/CODEX_EXPERIMENT_WORKFLOW.md")
    for phase in range(1, 10):
        assert f"## Phase {phase}" in workflow
    assert "Phase 3 — Implement [Sol]" in workflow
    assert "Phase 5 — Freeze / Handoff [Sol]" in workflow
    assert "Phase 6 — Execute frozen experiment [Luna]" in workflow
    assert "Phase 7 — Integrity audit [Sol]" in workflow
    assert "Phase 8 — Scientific analysis [Sol]" in workflow
    assert "Only Sol commits" in workflow


def test_handoff_template_is_fail_closed_and_not_runnable_by_default() -> None:
    handoff = _read("docs/workflows/EXPERIMENT_HANDOFF.md")
    required = (
        "repository_commit",
        "protocol_sha256",
        "manifest_sha256",
        "preregistration_sha256",
        "split_identity",
        "local_optimizer_backend",
        "opportunity_budget",
        "early_stop_rule",
        "api_scope",
        "validation_access_policy",
        "test_access_policy",
        "exact_runner_command",
        "expected_checkpoint_path",
        "expected_ledger_path",
        "allowed_retries",
        "fail_closed_conditions",
    )
    for field in required:
        assert field in handoff
    assert "READY_TO_RUN: false" in handoff
    assert "Only Sol may change `READY_TO_RUN` to `true`" in handoff
    assert "Luna must not improvise, repair source, amend the" in handoff
