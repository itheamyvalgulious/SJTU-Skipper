"""LLM plugin: provider wiring and two authentication modes.

Settings expose two explicit choices:

* ``openai`` — a generic OpenAI-compatible endpoint with a base URL and key.
* ``opencode`` — an OpenCode subscription linked through OAuth device flow.

OpenCode credentials are stored in this application's settings (write-only
secrets) and refreshed here.  The runtime does not read a local OpenCode
installation or database.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
import webbrowser
from typing import Any

from core.plugin import Plugin, PluginContext
from integrations.opencode_go import (
    OPENCODE_GO_BASE_URL,
    OpenCodeGoError,
    poll_device_token,
    refresh_access_token,
    start_device_flow,
)
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

_MODE_OPENAI = "openai"
_MODE_OPENCODE = "opencode"
_OPENCODE_SESSION_ID = uuid.uuid4().hex
_REFRESH_SKEW_MS = 60_000


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

    def __init__(self) -> None:
        self._settings: Any | None = None
        self._login_lock = threading.RLock()
        self._refresh_lock = threading.RLock()
        self._login_thread: threading.Thread | None = None

    def setup(self, ctx: PluginContext) -> None:
        settings = ctx.services.require("settings")
        self._settings = settings

        from plugins.settings.spec import SettingsSpec

        specs = [
            SettingsSpec(
                key="llm.mode",
                label="连接方式",
                type="select",
                default=_MODE_OPENAI,
                group="LLM",
                options=(
                    (_MODE_OPENAI, "OpenAI 兼容接口"),
                    (_MODE_OPENCODE, "OpenCode 套餐"),
                ),
                description="选择独立的 OpenAI-compatible 配置，或关联 OpenCode 套餐。",
                group_note=(
                    "两种方式互相独立：OpenAI 兼容使用本页的 Base URL / API Key；"
                    "OpenCode 套餐通过设备授权单独关联，凭据保存在本应用设置中，"
                    "不读取本机 OpenCode 安装。"
                ),
                order=0,
            ),
            SettingsSpec(
                key="llm.base_url",
                label="OpenAI Base URL",
                type="string",
                default=_DEFAULT_BASE_URL,
                group="LLM",
                description="OpenAI 兼容 Base URL；缺少 /v1 时会自动补全。",
                env=_ENV_BASE_URL_KEYS,
                visible_when=("llm.mode", (_MODE_OPENAI,)),
                order=1,
            ),
            SettingsSpec(
                key="llm.api_key",
                label="OpenAI API Key",
                type="secret",
                default="",
                group="LLM",
                description="OpenAI 兼容接口的 API Key。",
                env=_ENV_API_KEY_KEYS,
                visible_when=("llm.mode", (_MODE_OPENAI,)),
                order=2,
            ),
            SettingsSpec(
                key="llm.extra_headers",
                label="额外请求头（JSON）",
                type="json",
                default={},
                group="LLM",
                description='附加 HTTP 头 JSON，例如 {"X-Custom": "val"}。',
                visible_when=("llm.mode", (_MODE_OPENAI,)),
                order=3,
            ),
            SettingsSpec(
                key="llm.opencode_status",
                label="OpenCode 套餐状态",
                type="string",
                default="未关联",
                group="LLM",
                description="通过下方按钮关联；授权成功后凭据会自动刷新。",
                visible_when=("llm.mode", (_MODE_OPENCODE,)),
                readonly=True,
                order=4,
            ),
            SettingsSpec(
                key="llm.opencode_login",
                label="关联 OpenCode 套餐",
                type="action",
                action="llm.opencode_login",
                button_label="关联 OpenCode 套餐",
                group="LLM",
                description="打开 OpenCode 授权页面；完成后凭据自动保存。",
                visible_when=("llm.mode", (_MODE_OPENCODE,)),
                order=5,
            ),
            SettingsSpec(
                key="llm.opencode_logout",
                label="解除关联",
                type="action",
                action="llm.opencode_logout",
                button_label="解除关联",
                group="LLM",
                description="删除本应用保存的 OpenCode 套餐凭据。",
                visible_when=("llm.mode", (_MODE_OPENCODE,)),
                order=6,
            ),
            SettingsSpec(
                key="llm.opencode_access_token",
                label="OpenCode Access Token",
                type="secret",
                default="",
                group="LLM",
                hidden=True,
            ),
            SettingsSpec(
                key="llm.opencode_refresh_token",
                label="OpenCode Refresh Token",
                type="secret",
                default="",
                group="LLM",
                hidden=True,
            ),
            SettingsSpec(
                key="llm.opencode_expires_at",
                label="OpenCode Token 到期时间",
                type="int",
                default=0,
                group="LLM",
                hidden=True,
            ),
            SettingsSpec(
                key="llm.opencode_org_id",
                label="OpenCode Org ID",
                type="string",
                default="",
                group="LLM",
                hidden=True,
            ),
            SettingsSpec(
                key="llm.temperature",
                label="采样温度",
                type="float",
                default=_DEFAULT_TEMPERATURE,
                group="LLM",
                description="采样温度；留空使用模型默认。",
                order=10,
            ),
            SettingsSpec(
                key="llm.max_tokens",
                label="最大 Token 数",
                type="int",
                default=_DEFAULT_MAX_TOKENS,
                group="LLM",
                description="最大生成 Token 数；留空使用模型默认。",
                order=11,
            ),
            SettingsSpec(
                key="llm.timeout",
                label="超时（秒）",
                type="float",
                default=_DEFAULT_TIMEOUT,
                group="LLM",
                description="HTTP 请求超时秒数。",
                order=12,
            ),
            SettingsSpec(
                key="llm.max_retries",
                label="最大重试次数",
                type="int",
                default=_DEFAULT_MAX_RETRIES,
                group="LLM",
                description="瞬时错误自动重试次数。",
                order=13,
            ),
            SettingsSpec(
                key="llm.extra_body",
                label="额外请求体（JSON）",
                type="json",
                default={},
                group="LLM",
                description='合并进每次请求的额外 JSON，例如 {"thinking": "disabled"}。',
                order=14,
            ),
        ]
        for spec in specs:
            try:
                settings.register(spec)
            except Exception:
                logger.warning("settings register failed for %s", spec.key)

        # Actions are separate from the form; the backend writes secrets.
        ctx.ui.register_action(
            "llm.opencode_login", self._handle_opencode_login, plugin_id=self.id
        )
        ctx.ui.register_action(
            "llm.opencode_logout", self._handle_opencode_logout, plugin_id=self.id
        )

        def _config_provider(profile: str) -> ProviderConfig:
            try:
                mode = str(settings.get("llm.mode", _MODE_OPENAI) or _MODE_OPENAI).lower()
                temperature = settings.get("llm.temperature", _DEFAULT_TEMPERATURE)
                max_tokens = settings.get("llm.max_tokens", _DEFAULT_MAX_TOKENS)
                timeout = settings.get("llm.timeout", _DEFAULT_TIMEOUT)
                max_retries = settings.get("llm.max_retries", _DEFAULT_MAX_RETRIES)
                extra_body = settings.get("llm.extra_body", {})
                if isinstance(extra_body, str):
                    try:
                        extra_body = json.loads(extra_body)
                    except json.JSONDecodeError:
                        extra_body = {}
            except Exception:
                logger.exception("failed to read settings for profile %s", profile)
                return ProviderConfig()

            if mode == _MODE_OPENCODE:
                return self._opencode_config(
                    settings,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    max_retries=max_retries,
                    extra_body=extra_body,
                )

            base_url = str(settings.get("llm.base_url", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL)
            api_key = str(settings.get("llm.api_key", "") or "")
            extra_headers = settings.get("llm.extra_headers", {})
            if isinstance(extra_headers, str):
                try:
                    extra_headers = json.loads(extra_headers)
                except json.JSONDecodeError:
                    extra_headers = {}
            return ProviderConfig(
                base_url=base_url,
                api_key=api_key,
                model="",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(timeout) if timeout else _DEFAULT_TIMEOUT,
                max_retries=(
                    int(max_retries)
                    if max_retries is not None
                    else _DEFAULT_MAX_RETRIES
                ),
                extra_headers=dict(extra_headers or {}),
                extra_body=dict(extra_body or {}),
            )

        service = LLMService(
            config_provider=_config_provider,
            on_event=_on_llm_event,
        )

        try:
            config = _config_provider("agent")
        except OpenCodeGoError as exc:
            logger.error("OpenCode credentials unavailable at startup: %s", exc)
            config = ProviderConfig()
        openai_provider = OpenAICompatibleProvider(config=config)
        service.register_provider("openai", openai_provider)
        service.register_provider("openai-compatible", openai_provider)

        ctx.register_service("llm", service)
        logger.info("LLM plugin setup complete; providers: %s", service.registry.names())

    # ------------------------------------------------------------------
    # OpenCode mode
    # ------------------------------------------------------------------
    def _opencode_config(
        self,
        settings: Any,
        *,
        temperature: Any,
        max_tokens: Any,
        timeout: Any,
        max_retries: Any,
        extra_body: Any,
    ) -> ProviderConfig:
        with self._refresh_lock:
            access = str(settings.get("llm.opencode_access_token", "") or "")
            refresh = str(settings.get("llm.opencode_refresh_token", "") or "")
            expires_ms = int(settings.get("llm.opencode_expires_at", 0) or 0)
            org_id = str(settings.get("llm.opencode_org_id", "") or "")
            if not access:
                raise OpenCodeGoError("请先在设置中关联 OpenCode 套餐。")
            if expires_ms and expires_ms <= int(time.time() * 1000) + _REFRESH_SKEW_MS:
                tokens = refresh_access_token(refresh)
                access = tokens.access_token
                refresh = tokens.refresh_token or refresh
                org_id = tokens.org_id or org_id
                self._store_opencode_tokens(settings, tokens, refresh=refresh)
        if not org_id:
            raise OpenCodeGoError("OpenCode 套餐凭据缺少 orgID，请重新关联。")
        return ProviderConfig(
            base_url=OPENCODE_GO_BASE_URL,
            api_key=access,
            model="",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(timeout) if timeout else _DEFAULT_TIMEOUT,
            max_retries=(
                int(max_retries)
                if max_retries is not None
                else _DEFAULT_MAX_RETRIES
            ),
            extra_headers={
                "x-opencode-org-id": org_id,
                "x-opencode-session": _OPENCODE_SESSION_ID,
            },
            extra_body=dict(extra_body or {}),
        )

    def _store_opencode_tokens(
        self,
        settings: Any,
        tokens: Any,
        *,
        refresh: str = "",
    ) -> None:
        refresh_value = tokens.refresh_token or refresh
        expires = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(tokens.expires_at_ms / 1000)
        )
        org = tokens.org_id or str(settings.get("llm.opencode_org_id", "") or "")
        try:
            settings.set("llm.opencode_access_token", tokens.access_token)
            if refresh_value:
                settings.set("llm.opencode_refresh_token", refresh_value)
            settings.set("llm.opencode_expires_at", int(tokens.expires_at_ms))
            if org:
                settings.set("llm.opencode_org_id", org)
            settings.set(
                "llm.opencode_status",
                f"已关联（{org or 'OpenCode'}，有效期至 {expires}）",
            )
        except Exception:  # noqa: BLE001 - settings storage failure is reported below
            logger.exception("failed to store OpenCode credentials")

    # ------------------------------------------------------------------
    # OpenCode actions
    # ------------------------------------------------------------------
    def _handle_opencode_login(self, _payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings
        if settings is None:
            return {"ok": False, "message": "设置服务未就绪", "refresh": True}
        with self._login_lock:
            if self._login_thread is not None and self._login_thread.is_alive():
                return {
                    "ok": True,
                    "message": "授权正在进行，请完成浏览器中的授权。",
                    "refresh": True,
                }
        try:
            flow = start_device_flow()
        except OpenCodeGoError as exc:
            return {"ok": False, "message": str(exc), "refresh": True}
        opened = False
        try:
            opened = bool(webbrowser.open(flow.verification_uri_complete))
        except Exception:  # noqa: BLE001 - browser is optional
            opened = False
        try:
            settings.set("llm.opencode_status", "等待授权…")
        except Exception:  # noqa: BLE001
            pass
        self._start_login_poller(settings, flow)
        hint = (
            "已打开 OpenCode 授权页面，请在浏览器完成授权。"
            if opened
            else f"请打开 {flow.verification_uri_complete} 完成授权。"
        )
        return {
            "ok": True,
            "message": f"{hint} 验证码：{flow.user_code}（完成后自动保存）",
            "refresh": True,
        }

    def _start_login_poller(self, settings: Any, flow: Any) -> None:
        with self._login_lock:
            if self._login_thread is not None and self._login_thread.is_alive():
                return
            self._login_thread = threading.Thread(
                target=self._poll_login,
                args=(settings, flow),
                name="opencode-device-login",
                daemon=True,
            )
            self._login_thread.start()

    def _poll_login(self, settings: Any, flow: Any) -> None:
        deadline = time.monotonic() + flow.expires_in
        while time.monotonic() < deadline:
            try:
                tokens = poll_device_token(flow)
            except OpenCodeGoError as exc:
                logger.warning("OpenCode device login failed: %s", exc)
                try:
                    settings.set("llm.opencode_status", f"授权失败：{exc}")
                except Exception:  # noqa: BLE001
                    pass
                return
            if tokens is not None:
                self._store_opencode_tokens(settings, tokens)
                logger.info("OpenCode Go subscription linked")
                return
            time.sleep(max(1, int(flow.interval)))
        try:
            settings.set("llm.opencode_status", "授权超时，请重新关联")
        except Exception:  # noqa: BLE001
            pass

    def _handle_opencode_logout(self, _payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings
        if settings is None:
            return {"ok": False, "message": "设置服务未就绪", "refresh": True}
        for key in (
            "llm.opencode_access_token",
            "llm.opencode_refresh_token",
            "llm.opencode_org_id",
        ):
            try:
                settings.reset(key)
            except Exception:  # noqa: BLE001
                pass
        try:
            settings.set("llm.opencode_expires_at", 0)
            settings.set("llm.opencode_status", "未关联")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "message": "已解除 OpenCode 套餐关联", "refresh": True}


def _on_llm_event(event_type: str, payload: dict[str, Any]) -> None:
    logger.debug("llm event %s: %s", event_type, payload)
