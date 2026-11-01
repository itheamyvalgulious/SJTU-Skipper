"""Plugin that wires the LLM service into the runtime.

Registers settings items and constructs ``LLMService`` using a lazy
configuration callback so every request picks up the latest settings values.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from core.plugin import Plugin, PluginContext
from llm.errors import LLMError
from llm.openai_compat import OpenAICompatibleProvider
from llm.provider import ProviderRegistry
from llm.service import LLMService
from llm.types import ProviderConfig

logger = logging.getLogger("lecture_agent.plugin.llm")

_ENV_BASE_URL_KEYS = ("LLM_BASE_URL", "OPENAI_BASE_URL")
_ENV_API_KEY_KEYS = ("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-chat"
_DEFAULT_VLM_MODEL = "deepseek-chat"
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TEMPERATURE: float | None = None
_DEFAULT_MAX_TOKENS: int | None = None


def _resolve_env(key: str, env_keys: tuple[str, ...], default: str) -> str:
    for name in env_keys:
        val = os.environ.get(name)
        if val:
            return val
    return default


class LLMPlugin(Plugin):
    id = "llm"
    requires = ("settings",)
    provides = ("llm",)

    def setup(self, ctx: PluginContext) -> None:
        # --- discover settings service ---
        settings = ctx.services.require("settings")

        # --- register settings ---
        from plugins.settings.spec import SettingsSpec

        specs = [
            SettingsSpec(
                key="llm.base_url",
                label="LLM Base URL",
                type="string",
                default=_resolve_env("LLM_BASE_URL", _ENV_BASE_URL_KEYS, _DEFAULT_BASE_URL),
                group="LLM",
                description="OpenAI-compatible base URL. Append /v1 if missing.",
            ),
            SettingsSpec(
                key="llm.api_key",
                label="LLM API Key",
                type="secret",
                default=_resolve_env("LLM_API_KEY", _ENV_API_KEY_KEYS, ""),
                group="LLM",
                description="API key for the LLM provider.",
            ),
            SettingsSpec(
                key="llm.model",
                label="Default Model",
                type="string",
                default=_DEFAULT_MODEL,
                group="LLM",
                description="Model name for agent/completion calls.",
            ),
            SettingsSpec(
                key="llm.vlm_model",
                label="VLM Model",
                type="string",
                default=_DEFAULT_VLM_MODEL,
                group="LLM",
                description="Model name for vision-language calls (overrides llm.model when profile=vlm).",
            ),
            SettingsSpec(
                key="llm.temperature",
                label="Temperature",
                type="float",
                default=_DEFAULT_TEMPERATURE,
                group="LLM",
                description="Sampling temperature (omit for model default).",
            ),
            SettingsSpec(
                key="llm.max_tokens",
                label="Max Tokens",
                type="int",
                default=_DEFAULT_MAX_TOKENS,
                group="LLM",
                description="Maximum completion tokens (omit for model default).",
            ),
            SettingsSpec(
                key="llm.timeout",
                label="Timeout (s)",
                type="float",
                default=_DEFAULT_TIMEOUT,
                group="LLM",
                description="HTTP request timeout in seconds.",
            ),
            SettingsSpec(
                key="llm.max_retries",
                label="Max Retries",
                type="int",
                default=_DEFAULT_MAX_RETRIES,
                group="LLM",
                description="Automatic retry attempts for transient failures.",
            ),
            SettingsSpec(
                key="llm.extra_headers",
                label="Extra Headers (JSON)",
                type="json",
                default={},
                group="LLM",
                description='Additional HTTP headers as JSON object, e.g. {"X-Custom": "val"}.',
            ),
            SettingsSpec(
                key="llm.extra_body",
                label="Extra Body (JSON)",
                type="json",
                default={},
                group="LLM",
                description='Extra JSON body fields merged into every request, e.g. {"thinking": "disabled"}.',
            ),
        ]
        for spec in specs:
            try:
                settings.register(spec)
            except Exception:
                logger.warning("settings register failed for %s", spec.key)

        # --- build config callback ---
        def _config_provider(profile: str) -> ProviderConfig:
            try:
                base_url = settings.get("llm.base_url", _DEFAULT_BASE_URL)
                api_key = settings.get("llm.api_key", "")
                model = settings.get("llm.model", _DEFAULT_MODEL)
                temperature = settings.get("llm.temperature", _DEFAULT_TEMPERATURE)
                max_tokens = settings.get("llm.max_tokens", _DEFAULT_MAX_TOKENS)
                timeout = settings.get("llm.timeout", _DEFAULT_TIMEOUT)
                max_retries = settings.get("llm.max_retries", _DEFAULT_MAX_RETRIES)

                extra_headers = settings.get("llm.extra_headers", {})
                if isinstance(extra_headers, str):
                    try:
                        extra_headers = json.loads(extra_headers)
                    except json.JSONDecodeError:
                        extra_headers = {}
                extra_body = settings.get("llm.extra_body", {})
                if isinstance(extra_body, str):
                    try:
                        extra_body = json.loads(extra_body)
                    except json.JSONDecodeError:
                        extra_body = {}
            except Exception:
                logger.exception("failed to read settings for profile %s", profile)
                return ProviderConfig()

            if profile == "vlm":
                vlm_model = settings.get("llm.vlm_model", "")
                if vlm_model:
                    model = vlm_model

            return ProviderConfig(
                base_url=base_url or _DEFAULT_BASE_URL,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(timeout) if timeout else _DEFAULT_TIMEOUT,
                max_retries=int(max_retries) if max_retries else _DEFAULT_MAX_RETRIES,
                extra_headers=dict(extra_headers),
                extra_body=dict(extra_body),
            )

        # --- build service ---
        service = LLMService(
            config_provider=_config_provider,
            on_event=_on_llm_event,
        )

        # --- register the OpenAI provider ---
        config = _config_provider("agent")
        openai_provider = OpenAICompatibleProvider(config=config)
        service.register_provider("openai", openai_provider)
        service.register_provider("openai-compatible", openai_provider)

        # --- publish ---
        ctx.register_service("llm", service)
        logger.info("LLM plugin setup complete; providers: %s", service.registry.names())


def _on_llm_event(event_type: str, payload: dict[str, Any]) -> None:
    """Bridge LLM observability events to the event bus.

    Note: this runs inside the callers thread (not the dispatcher thread)
    because we do not have the bus reference here.  The *real* emission
    happens in the service tier via the runtime LLMProxy events bridge.
    This is a lightweight stub that logs for now.
    """
    logger.debug("llm event %s: %s", event_type, payload)