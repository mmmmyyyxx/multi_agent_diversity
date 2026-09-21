from __future__ import annotations

import os
from dataclasses import dataclass


DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
PROVIDER_PROFILE_ENV = "DASHSCOPE_PROVIDER_PROFILE"
DEFAULT_PROVIDER_PROFILE = "myx"
LWJ_DASHSCOPE_API_KEY_ENV = "LWJ_DASHSCOPE_API_KEY"
LWJ_DASHSCOPE_BASE_URL_ENV = "LWJ_DASHSCOPE_BASE_URL"
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
    return env_name, os.getenv(env_name, "")


def resolve_base_url(
    configured_env: str = "", provider_profile: str | None = None,
) -> tuple[str, str]:
    profile = resolve_provider_profile(provider_profile)
    env_name = str(configured_env or "").strip()
    if not env_name or env_name == DASHSCOPE_BASE_URL_ENV:
        env_name = profile.base_url_env
    configured = os.getenv(env_name, "")
    if configured:
        return env_name, configured
    if env_name == profile.base_url_env:
        return env_name, profile.default_base_url
    return env_name, ""
