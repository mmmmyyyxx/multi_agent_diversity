"""Single provider-client construction boundary.

Scientific modules receive clients through execution services. Compatibility
runtime code may resolve environment-backed credentials here; raw values never
enter scientific specs or identities.
"""

from __future__ import annotations

from typing import Any, Callable

from openai import AsyncOpenAI

from .provider_credentials import resolve_api_key, resolve_base_url


class ProviderConfigurationError(ValueError):
    pass


class ProviderClientFactory:
    @staticmethod
    def create(
        *,
        api_key: str,
        base_url: str,
        client_type: Callable[..., AsyncOpenAI] = AsyncOpenAI,
        **kwargs: Any,
    ) -> AsyncOpenAI:
        if not api_key:
            raise ProviderConfigurationError("provider API key is unavailable")
        if not base_url:
            raise ProviderConfigurationError("provider base URL is unavailable")
        return client_type(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def from_environment(
        cls,
        *,
        provider_profile: str,
        api_key_env: str = "",
        base_url_env: str = "",
        **kwargs: Any,
    ) -> AsyncOpenAI:
        resolved_key_env, key = resolve_api_key(api_key_env, provider_profile)
        resolved_base_env, base = resolve_base_url(base_url_env, provider_profile)
        if not key:
            raise ProviderConfigurationError(
                f"provider API key is unavailable via {resolved_key_env}"
            )
        if not base:
            raise ProviderConfigurationError(
                f"provider base URL is unavailable via {resolved_base_env}"
            )
        return cls.create(api_key=key, base_url=base, **kwargs)
