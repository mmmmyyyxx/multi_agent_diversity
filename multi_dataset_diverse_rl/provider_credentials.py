from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass


DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
PROVIDER_PROFILE_ENV = "DASHSCOPE_PROVIDER_PROFILE"
DEFAULT_PROVIDER_PROFILE = "myx"
LWJ_DASHSCOPE_API_KEY_ENV = "LWJ_DASHSCOPE_API_KEY"
LWJ_DASHSCOPE_BASE_URL_ENV = "LWJ_DASHSCOPE_BASE_URL"
OPENLUX_API_KEY_ENV = "OPENLUX_API_KEY"
OPENLUX_BASE_URL_ENV = "OPENLUX_BASE_URL"
PRIVATE_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'runs/provider_credentials.local.json'
DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL = (
    "https://ws-tbeq6fj4ndibcz5p.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    api_key_env: str
    base_url_env: str
    default_base_url: str


PROVIDER_PROFILES = {
    "openlux": ProviderProfile("openlux", OPENLUX_API_KEY_ENV, OPENLUX_BASE_URL_ENV, ""),
    "myx": ProviderProfile(
        name="myx",
        api_key_env=DASHSCOPE_API_KEY_ENV,
        base_url_env=DASHSCOPE_BASE_URL_ENV,
        default_base_url=DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL,
    ),
    "lwj": ProviderProfile(
        name="lwj",
        api_key_env=LWJ_DASHSCOPE_API_KEY_ENV,
        base_url_env=LWJ_DASHSCOPE_BASE_URL_ENV,
        default_base_url="",
    ),
}


def _deployment_value(name):
    value = os.getenv(name, "")
    if value or name not in {OPENLUX_API_KEY_ENV, OPENLUX_BASE_URL_ENV}:
        return value
    if os.getenv('FORMAL_ZERO_API_GUARD_REQUIRED') == '1' or not PRIVATE_CONFIG_PATH.exists():
        return ""
    config = json.loads(PRIVATE_CONFIG_PATH.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not isinstance(config.get(name, ''), str):
        raise ValueError('PRIVATE_PROVIDER_CONFIG_INVALID')
    return config.get(name, '')


def resolve_provider_profile(configured_profile: str | None = None) -> ProviderProfile:
    name = str(configured_profile or "").strip().lower()
    if not name:
        name = os.getenv(PROVIDER_PROFILE_ENV, DEFAULT_PROVIDER_PROFILE).strip().lower()
    try:
        return PROVIDER_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROVIDER_PROFILES))
        raise ValueError(
            f"Unknown provider profile {name!r}; expected one of: {choices}"
        ) from exc


def resolve_api_key(
    configured_env: str = "", provider_profile: str | None = None,
) -> tuple[str, str]:
    profile = resolve_provider_profile(provider_profile)
    env_name = str(configured_env or "").strip()
    if not env_name or env_name == DASHSCOPE_API_KEY_ENV:
        env_name = profile.api_key_env
    return env_name, _deployment_value(env_name)


def resolve_base_url(
    configured_env: str = "", provider_profile: str | None = None,
) -> tuple[str, str]:
    profile = resolve_provider_profile(provider_profile)
    env_name = str(configured_env or "").strip()
    if not env_name or env_name == DASHSCOPE_BASE_URL_ENV:
        env_name = profile.base_url_env
    configured = _deployment_value(env_name)
    if configured:
        return env_name, configured
    if env_name == profile.base_url_env:
        return env_name, profile.default_base_url
    return env_name, ""
