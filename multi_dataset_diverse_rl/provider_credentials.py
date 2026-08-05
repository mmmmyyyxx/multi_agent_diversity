from __future__ import annotations

import os


DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
DASHSCOPE_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL = (
    "https://ws-tbeq6fj4ndibcz5p.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)


def resolve_api_key(configured_env: str) -> tuple[str, str]:
    env_name = str(configured_env or "").strip() or DASHSCOPE_API_KEY_ENV
    return env_name, os.getenv(env_name, "")


def resolve_base_url(configured_env: str) -> tuple[str, str]:
    env_name = str(configured_env or "").strip()
    if not env_name:
        env_name = DASHSCOPE_BASE_URL_ENV
    configured = os.getenv(env_name, "")
    if configured:
        return env_name, configured
    if env_name == DASHSCOPE_BASE_URL_ENV:
        return env_name, DASHSCOPE_OPENAI_COMPATIBLE_BASE_URL
    return env_name, ""
