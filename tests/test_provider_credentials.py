from types import SimpleNamespace

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient
from multi_dataset_diverse_rl.provider_credentials import (
    DASHSCOPE_API_KEY_ENV,
    DASHSCOPE_BASE_URL_ENV,
    DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL,
    resolve_api_key,
    resolve_base_url,
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
