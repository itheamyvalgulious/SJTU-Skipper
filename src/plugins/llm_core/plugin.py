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
                default=_DEFAULT_BASE_URL,
                group="LLM",
                description="OpenAI 兼容 Base URL；缺少 /v1 时会自动补全。",
                env=_ENV_BASE_URL_KEYS,
                group_note="这里只配置连接、鉴权与重试；具体模型在 Agent / VLM / Summarize / STT 分组中选择。环境变量只在首次注册时写入设置，之后以设置页为准。",
            ),
            SettingsSpec(
                key="llm.api_key",
                label="LLM API Key",
                type="secret",
                default="",
                group="LLM",
                description="Agent / VLM / Summarize 共用的 API Key。",
                env=_ENV_API_KEY_KEYS,
            ),
            SettingsSpec(
                key="llm.temperature",
                label="采样温度",
                type="float",
                default=_DEFAULT_TEMPERATURE,
                group="LLM",
                description="采样温度；留空使用模型默认。",
            ),
            SettingsSpec(
                key="llm.max_tokens",
                label="最大 Token 数",
                type="int",
                default=_DEFAULT_MAX_TOKENS,
                group="LLM",
                description="最大生成 Token 数；留空使用模型默认。",
            ),
            SettingsSpec(
                key="llm.timeout",
                label="超时（秒）",
                type="float",
                default=_DEFAULT_TIMEOUT,
                group="LLM",
                description="HTTP 请求超时秒数。",
            ),
            SettingsSpec(
                key="llm.max_retries",
                label="最大重试次数",
                type="int",
                default=_DEFAULT_MAX_RETRIES,
                group="LLM",
                description="瞬时错误自动重试次数。",
            ),
            SettingsSpec(
                key="llm.extra_headers",
                label="额外请求头（JSON）",
                type="json",
                default={},
                group="LLM",
                description='附加 HTTP 头 JSON，例如 {"X-Custom": "val"}。',
            ),
            SettingsSpec(
                key="llm.extra_body",
                label="额外请求体（JSON）",
                type="json",
                default={},
                group="LLM",
                description='合并进每次请求的额外 JSON，例如 {"thinking": "disabled"}。',
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

            return ProviderConfig(
                base_url=base_url or _DEFAULT_BASE_URL,
                api_key=api_key,
                # Model selection belongs to the consuming plugin (Agent / VLM
                # / Summarize / STT) and is passed per request.
                model="",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(timeout) if timeout else _DEFAULT_TIMEOUT,
                max_retries=(
                    int(max_retries)
                    if max_retries is not None
                    else _DEFAULT_MAX_RETRIES
                ),
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