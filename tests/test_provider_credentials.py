from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.config import Config, add_config_arguments, config_from_args
from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient
from multi_dataset_diverse_rl.provider_credentials import (
    DASHSCOPE_API_KEY_ENV,
    DASHSCOPE_BASE_URL_ENV,
    DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL,
    LWJ_DASHSCOPE_API_KEY_ENV,
    LWJ_DASHSCOPE_BASE_URL_ENV,
    PROVIDER_PROFILE_ENV,
    resolve_api_key,
    resolve_base_url,
    resolve_provider_profile,
)


def test_dashscope_defaults_resolve_key_env_and_project_endpoint(monkeypatch):
    monkeypatch.setenv(DASHSCOPE_API_KEY_ENV, "test-key")
    monkeypatch.delenv(DASHSCOPE_BASE_URL_ENV, raising=False)

    assert resolve_api_key(DASHSCOPE_API_KEY_ENV) == (
        DASHSCOPE_API_KEY_ENV,
        "test-key",
    )
    assert resolve_base_url(DASHSCOPE_BASE_URL_ENV) == (
        DASHSCOPE_BASE_URL_ENV,
        DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL,
    )
    assert resolve_api_key("") == (DASHSCOPE_API_KEY_ENV, "test-key")
    assert resolve_base_url("") == (
        DASHSCOPE_BASE_URL_ENV,
        DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL,
    )


def test_dashscope_base_url_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv(DASHSCOPE_BASE_URL_ENV, "https://override.invalid/v1")
    assert resolve_base_url(DASHSCOPE_BASE_URL_ENV) == (
        DASHSCOPE_BASE_URL_ENV,
        "https://override.invalid/v1",
    )


def test_role_client_uses_dashscope_defaults_without_exposing_key(
    monkeypatch,
):
    captured = {}
    monkeypatch.setenv(DASHSCOPE_API_KEY_ENV, "test-key")
    monkeypatch.delenv(DASHSCOPE_BASE_URL_ENV, raising=False)

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "multi_dataset_diverse_rl.llm_client.AsyncOpenAI",
        fake_openai,
    )
    client = RoleAwareLLMClient(Config())
    assert client._client_or_raise("solver") is not None
    assert captured == {
        "api_key": "test-key",
        "base_url": DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL,
    }


def test_lwj_profile_uses_isolated_credentials_and_endpoint(monkeypatch):
    monkeypatch.setenv(LWJ_DASHSCOPE_API_KEY_ENV, "lwj-test-key")
    monkeypatch.setenv(LWJ_DASHSCOPE_BASE_URL_ENV, "https://lwj.invalid/v1")

    assert resolve_api_key(DASHSCOPE_API_KEY_ENV, "lwj") == (
        LWJ_DASHSCOPE_API_KEY_ENV,
        "lwj-test-key",
    )
    assert resolve_base_url(DASHSCOPE_BASE_URL_ENV, "lwj") == (
        LWJ_DASHSCOPE_BASE_URL_ENV,
        "https://lwj.invalid/v1",
    )


def test_explicit_role_environment_override_wins_over_profile(monkeypatch):
    monkeypatch.setenv("CUSTOM_SOLVER_KEY", "custom-test-key")
    monkeypatch.setenv("CUSTOM_SOLVER_BASE", "https://custom.invalid/v1")

    assert resolve_api_key("CUSTOM_SOLVER_KEY", "lwj") == (
        "CUSTOM_SOLVER_KEY",
        "custom-test-key",
    )
    assert resolve_base_url("CUSTOM_SOLVER_BASE", "lwj") == (
        "CUSTOM_SOLVER_BASE",
        "https://custom.invalid/v1",
    )


def test_environment_profile_switch_supports_legacy_callers(monkeypatch):
    monkeypatch.setenv(PROVIDER_PROFILE_ENV, "lwj")
    monkeypatch.setenv(LWJ_DASHSCOPE_API_KEY_ENV, "lwj-test-key")

    assert resolve_provider_profile().name == "lwj"
    assert resolve_api_key(DASHSCOPE_API_KEY_ENV)[0] == LWJ_DASHSCOPE_API_KEY_ENV
    assert resolve_base_url(DASHSCOPE_BASE_URL_ENV)[0] == LWJ_DASHSCOPE_BASE_URL_ENV


def test_lwj_config_selects_lwj_client_without_exposing_key(monkeypatch):
    captured = {}
    monkeypatch.setenv(LWJ_DASHSCOPE_API_KEY_ENV, "lwj-test-key")
    monkeypatch.setenv(LWJ_DASHSCOPE_BASE_URL_ENV, "https://lwj.invalid/v1")

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "multi_dataset_diverse_rl.llm_client.AsyncOpenAI",
        fake_openai,
    )
    cfg = Config.from_flat(provider_profile="lwj")
    client = RoleAwareLLMClient(cfg)

    assert client._client_or_raise("solver") is not None
    assert cfg.to_flat_dict()["provider_profile"] == "lwj"
    assert captured == {
        "api_key": "lwj-test-key",
        "base_url": "https://lwj.invalid/v1",
    }
    assert client.cost_summary()["provider_profile"] == "lwj"


def test_unknown_provider_profile_fails_closed():
    with pytest.raises(ValueError, match="Unknown provider profile"):
        resolve_provider_profile("unknown")
    with pytest.raises(ValueError, match="Unknown provider_profile"):
        Config.from_flat(provider_profile="unknown")


def test_cli_switches_provider_profile_without_serializing_secret(monkeypatch):
    import argparse
    import json

    monkeypatch.setenv(LWJ_DASHSCOPE_API_KEY_ENV, "lwj-test-key")
    parser = add_config_arguments(argparse.ArgumentParser())
    cfg = config_from_args(parser.parse_args(["--provider_profile", "lwj"]))
    client = RoleAwareLLMClient(cfg)
    client.record_override_solver(started=0.0)

    assert cfg.models.provider_profile == "lwj"
    assert client.calls[0]["provider_profile"] == "lwj"
    assert client.cost_summary()["provider_profile"] == "lwj"
    assert "lwj-test-key" not in json.dumps(client.calls, sort_keys=True)
