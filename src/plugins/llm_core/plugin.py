"""LLM configuration: providers, models and role assignments.

The page layout lives in :mod:`plugins.llm_core.settings_view`; this module owns
the interaction state (which provider/model/role is being edited), the device
login, and the actions behind the buttons.

Selection is kept server-side so the editor form always carries the values of
the item it writes.  All YAML writing still goes through the runtime's
structured methods; this module never parses ``models.yml``/``roles.yml``.
"""

from __future__ import annotations

import logging
import threading
import time
import webbrowser
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from agent_runtime.client import RuntimeClient
from core.agent_types import IpcMethod
from core.plugin import Plugin, PluginContext
from core.ui import BlockType
from integrations.opencode_go import (
    OpenCodeGoError,
    poll_device_token,
    start_device_flow,
)

from .settings_view import (
    INPUT_FORMATS,
    PAGE_ID,
    PRESETS,
    SettingsView,
    number,
    providers,
    role_titles,
    roles_using,
)

logger = logging.getLogger("lecture_agent.plugin.llm")

_ACTIONS = (
    "llm.configure",
    "llm.credentials",
    "llm.logout",
    "llm.opencode_login",
    "llm.opencode_cancel",
    "llm.provider.save",
    "llm.provider.remove",
    "llm.check",
    "llm.model.add",
    "llm.model.add_all",
    "llm.model.save",
    "llm.model.remove",
    "llm.role.save",
    "llm.select.provider",
    "llm.select.model",
    "llm.select.role",
)


class ProviderAuth(StrEnum):
    NONE = "none"
    API_KEY = "apiKey"
    OAUTH = "oauth"


class CredentialType(StrEnum):
    API_KEY = "api_key"
    OAUTH = "oauth"


class ProviderSource(StrEnum):
    CONFIG = "config"
    PROJECT = "project"


def _values(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("values")
    return nested if isinstance(nested, Mapping) else payload


@dataclass
class LoginFlow:
    """One OpenCode device authorization in flight."""

    provider: str
    flow: Any
    cancel: threading.Event
    expires_at: float
    thread: threading.Thread | None = None


class LLMPlugin(Plugin):
    id = "llm"
    requires = ("settings",)
    provides = ("llm",)

    def __init__(self) -> None:
        self._settings: Any = None
        self._runtime: RuntimeClient | None = None
        self._ctx: PluginContext | None = None
        self._login_lock = threading.RLock()
        self._login: LoginFlow | None = None
        self._login_status = ""
        # Which provider/model/role each page is editing.
        self._selected = {"provider": "", "model": "", "role": ""}
        # Last endpoint check per provider: {"models": [...], "error": str}.
        self._checks: dict[str, dict[str, Any]] = {}

    def setup(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._settings = ctx.require("settings")
        runtime = RuntimeClient(ctx.data_dir, settings=self._settings)
        self._runtime = runtime
        ctx.register_service("llm", runtime)
        handlers = (
            self._handle_configure,
            self._handle_credentials,
            self._handle_logout,
            self._handle_opencode_login,
            self._handle_opencode_cancel,
            self._handle_provider_save,
            self._handle_provider_remove,
            self._handle_check,
            self._handle_model_add,
            self._handle_model_add_all,
            self._handle_model_save,
            self._handle_model_remove,
            self._handle_role_save,
            self._handle_select_provider,
            self._handle_select_model,
            self._handle_select_role,
        )
        for action, handler in zip(_ACTIONS, handlers):
            ctx.ui.register_action(action, handler, plugin_id=self.id)
        self._settings.register_section("llm", "LLM", self._render_settings)

    def teardown(self) -> None:
        self._cancel_login()
        if self._settings is not None:
            self._settings.unregister_section("llm")
        if self._ctx is not None:
            for action in _ACTIONS:
                self._ctx.ui.unregister_action(action)
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None

    def _refresh(self) -> None:
        if self._ctx is not None:
            self._ctx.ui.refresh(PAGE_ID)

    def _call(self, method: IpcMethod, **params: Any) -> dict[str, Any]:
        if self._runtime is None:
            raise RuntimeError("The LLM runtime is unavailable")
        result = self._runtime.call(method, **params)
        if method == IpcMethod.INFO:
            result["roles"] = {
                role: resolved for role, resolved in result.get("roles", {}).items()
            }
            for provider in result.get("providers", []):
                provider["auth"] = ProviderAuth(provider["auth"])
                if provider.get("source"):
                    provider["source"] = ProviderSource(provider["source"])
        return result

    def _catalog(self) -> dict[str, Any]:
        return self._call(IpcMethod.CATALOG)

    # ------------------------------------------------------------------ view

    def _render_settings(self) -> list[dict[str, Any]]:
        try:
            catalog = self._catalog()
        except Exception as exc:
            logger.exception("failed to read the LLM catalog")
            return [
                {
                    "type": BlockType.MARKDOWN,
                    "text": f"### LLM\n\n运行时不可用：{exc}．修改配置前请先解决启动问题．",
                }
            ]
        view = SettingsView(
            catalog,
            self._selected,
            self._checks,
            login=self._login_state(),
            status=self._login_status,
        )
        return [{"type": BlockType.TABS, "id": PAGE_ID, "tabs": view.blocks()}]

    # --------------------------------------------------------------- helpers

    def _models_for(
        self, catalog: Mapping[str, Any], provider: str
    ) -> list[dict[str, Any]]:
        for item in providers(catalog):
            if item["id"] == provider:
                return [dict(model) for model in item.get("models", [])]
        return []

    @staticmethod
    def _model_payload(models: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Translate catalog models into the runtime's ``models_set`` rows."""

        payload: list[dict[str, Any]] = []
        for model in models:
            row: dict[str, Any] = {"id": str(model.get("id") or "")}
            for key in ("contextWindow", "maxTokens"):
                value = number(model.get(key))
                if value > 0:
                    row[key] = value
            formats = [
                fmt for fmt in model.get("input", []) if str(fmt) in INPUT_FORMATS
            ]
            row["input"] = formats or ["text"]
            row["reasoning"] = bool(model.get("reasoning"))
            payload.append(row)
        return payload

    def _write_models(
        self, provider: str, models: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return self._call(
            IpcMethod.MODELS_SET,
            provider=provider,
            models=self._model_payload(models),
        )

    # ------------------------------------------------------ selection actions

    def _handle_select_provider(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(_values(payload).get("provider") or "")
        if provider:
            self._selected["provider"] = provider
            self._selected["model"] = ""
            self._refresh()
        return {"ok": True, "message": "", "refresh": True}

    def _handle_select_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        provider = str(values.get("provider") or "")
        model = str(values.get("model") or "")
        if provider:
            self._selected["provider"] = provider
            self._selected["model"] = model
            self._refresh()
        return {"ok": True, "message": "", "refresh": True}

    def _handle_select_role(self, payload: dict[str, Any]) -> dict[str, Any]:
        role = str(_values(payload).get("role") or "")
        if role:
            self._selected["role"] = role
            self._refresh()
        return {"ok": True, "message": "", "refresh": True}

    # -------------------------------------------------------- provider actions

    def _handle_provider_save(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        preset = str(values.get("preset") or "")
        defaults = PRESETS.get(preset, {})
        provider = str(values.get("provider") or "").strip()
        if not provider and preset and preset != "custom":
            provider = preset
        if not provider:
            return {"ok": False, "message": "请填写提供方标识．", "refresh": False}
        base_url = str(values.get("base_url") or "").strip() or defaults.get(
            "base_url", ""
        )
        api = str(values.get("api") or "").strip() or defaults.get(
            "api", "openai-completions"
        )
        auth = str(values.get("auth") or "").strip() or defaults.get("auth", "apiKey")
        key = values.get("key")
        known = {item["id"] for item in providers(self._catalog())}
        try:
            if isinstance(key, str) and key.strip():
                self._call(IpcMethod.CREDENTIALS, provider=provider, key=key)
            self._call(
                IpcMethod.PROVIDER_SET,
                id=provider,
                base_url=base_url,
                api=api,
                auth=auth,
            )
        except Exception as exc:  # noqa: BLE001 - shown as a toast
            return {
                "ok": False,
                "message": f"保存提供方失败：{exc}．",
                "refresh": False,
            }
        self._checks.pop(provider, None)
        self._selected["provider"] = provider
        self._refresh()
        if provider in known:
            return {
                "ok": True,
                "message": f"已保存 {provider}；需要时在「模型」页检查端点．",
                "refresh": True,
            }
        return {
            "ok": True,
            "message": f"已接入 {provider}，模型取自端点．",
            "refresh": True,
        }

    def _handle_provider_remove(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(_values(payload).get("provider") or "")
        if not provider:
            return {"ok": False, "message": "请选择提供方．", "refresh": False}
        try:
            self._call(IpcMethod.PROVIDER_REMOVE, provider=provider)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"删除失败：{exc}．", "refresh": False}
        self._checks.pop(provider, None)
        if self._selected["provider"] == provider:
            self._selected["provider"] = ""
            self._selected["model"] = ""
        self._refresh()
        return {"ok": True, "message": f"已删除 {provider}．", "refresh": True}

    def _handle_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(_values(payload).get("provider") or "")
        if not provider:
            return {"ok": False, "message": "请选择提供方．", "refresh": False}
        try:
            found = self._call(IpcMethod.DISCOVER, provider=provider)
        except Exception as exc:  # noqa: BLE001 - shown inline and as a toast
            self._checks[provider] = {"models": [], "error": str(exc)}
            self._refresh()
            return {"ok": False, "message": f"检查端点失败：{exc}．", "refresh": True}
        models = list(found.get("models", []))
        self._checks[provider] = {"models": models, "error": ""}
        self._selected["provider"] = provider
        self._refresh()
        return {
            "ok": True,
            "message": f"端点返回 {len(models)} 个模型．",
            "refresh": True,
        }

    # ----------------------------------------------------------- model actions

    def _handle_model_add(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        provider = str(values.get("provider") or "")
        model_id = str(values.get("model") or "")
        found = {
            item["id"]: item
            for item in self._checks.get(provider, {}).get("models", [])
        }
        if model_id not in found:
            return {
                "ok": False,
                "message": "先检查端点，再从列表中挑选模型．",
                "refresh": False,
            }
        return self._append_models(provider, [found[model_id]])

    def _handle_model_add_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(_values(payload).get("provider") or "")
        known = {model["id"] for model in self._models_for(self._catalog(), provider)}
        found = [
            item
            for item in self._checks.get(provider, {}).get("models", [])
            if item["id"] not in known
        ]
        if not found:
            return {
                "ok": False,
                "message": "没有可添加的模型，请先检查端点．",
                "refresh": False,
            }
        return self._append_models(provider, found)

    def _append_models(
        self, provider: str, discovered: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        current = self._models_for(self._catalog(), provider)
        if not current:
            return {
                "ok": False,
                "message": "提供方不存在或尚无模型，请重新检查端点．",
                "refresh": False,
            }
        merged = current + [
            {
                "id": str(item["id"]),
                "contextWindow": item.get("contextWindow"),
                "maxTokens": item.get("maxTokens"),
                "input": list(item.get("input", [])),
                "reasoning": bool(item.get("reasoning")),
            }
            for item in discovered
        ]
        try:
            self._write_models(provider, merged)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "message": f"添加模型失败：{exc}．",
                "refresh": False,
            }
        self._refresh()
        return {
            "ok": True,
            "message": f"已添加 {len(discovered)} 个模型．",
            "refresh": True,
        }

    def _handle_model_save(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        provider = str(values.get("provider") or "")
        model_id = str(values.get("model") or "")
        models = self._models_for(self._catalog(), provider)
        if not any(model["id"] == model_id for model in models):
            return {"ok": False, "message": "模型不存在．", "refresh": True}
        formats = [
            fmt
            for fmt, key in (("text", "input_text"), ("image", "input_image"))
            if values.get(key)
        ]
        updated: list[dict[str, Any]] = []
        for model in models:
            if model["id"] != model_id:
                updated.append(model)
                continue
            updated.append(
                {
                    **model,
                    "contextWindow": values.get("context_window"),
                    "maxTokens": values.get("max_tokens"),
                    "input": formats or ["text"],
                    "reasoning": bool(values.get("reasoning")),
                }
            )
        try:
            self._write_models(provider, updated)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "message": f"保存模型失败：{exc}．",
                "refresh": True,
            }
        self._selected["provider"] = provider
        self._selected["model"] = model_id
        self._refresh()
        return {
            "ok": True,
            "message": f"已保存 {provider}/{model_id}．",
            "refresh": True,
        }

    def _handle_model_remove(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        provider = str(values.get("provider") or "")
        model_id = str(values.get("model") or "")
        catalog = self._catalog()
        models = self._models_for(catalog, provider)
        remaining = [model for model in models if model["id"] != model_id]
        if len(remaining) == len(models):
            return {"ok": False, "message": "模型不存在．", "refresh": True}
        roles = roles_using(catalog, provider, model_id)
        if roles:
            return {
                "ok": False,
                "message": (
                    f"`{provider}/{model_id}` 仍被角色 "
                    f"{role_titles(roles)} 使用．"
                    "请先在「角色」页切换．"
                ),
                "refresh": False,
            }
        if not remaining:
            return {
                "ok": False,
                "message": "提供方至少需要一个模型；如需停用请删除整个提供方．",
                "refresh": True,
            }
        try:
            self._write_models(provider, remaining)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "message": f"删除模型失败：{exc}．",
                "refresh": False,
            }
        if self._selected["model"] == model_id:
            self._selected["model"] = ""
        self._refresh()
        return {
            "ok": True,
            "message": f"已删除 {provider}/{model_id}．",
            "refresh": True,
        }

    # ------------------------------------------------------------ role action

    def _handle_role_save(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        role = str(values.get("role") or "")
        model = str(values.get("model") or "")
        level = str(values.get("level") or "off")
        if not role or not model:
            return {"ok": False, "message": "请选择角色与模型．", "refresh": False}
        catalog = self._catalog()
        if role == "board_recognition" and not model.startswith("@"):
            provider, _, model_id = model.partition("/")
            image = any(
                "image" in item.get("input", [])
                for provider_item in providers(catalog)
                if provider_item["id"] == provider
                for item in provider_item.get("models", [])
                if item["id"] == model_id
            )
            if not image:
                return {
                    "ok": False,
                    "message": f"`{model}` 没有启用 `image` 输入，板书识别无法使用．",
                    "refresh": False,
                }
        selector = model if model.startswith("@") else f"{model}:{level}"
        try:
            self._call(IpcMethod.ROLE_SET, role=role, selector=selector)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "message": f"保存角色失败：{exc}．",
                "refresh": False,
            }
        self._selected["role"] = role
        self._refresh()
        return {"ok": True, "message": f"已设置 {role}．", "refresh": True}

    # ------------------------------------------------------------ raw config

    def _handle_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        models = values.get("models_yaml")
        roles = values.get("roles_yaml")
        if not isinstance(models, str) or not isinstance(roles, str):
            return {
                "ok": False,
                "message": "请填写两份 YAML 配置后保存．",
                "refresh": False,
            }
        try:
            self._call(IpcMethod.CONFIGURE, models=models, roles=roles)
        except Exception as exc:  # noqa: BLE001 - the runtime owns validation
            return {
                "ok": False,
                "message": f"保存失败：{exc}．修改配置后重试．",
                "refresh": False,
            }
        self._checks.clear()
        self._refresh()
        return {
            "ok": True,
            "message": "已保存配置，新会话使用新配置．",
            "refresh": True,
        }

    # ------------------------------------------------------------ credentials

    def _handle_credentials(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        provider = str(values.get("provider") or "")
        key = values.get("key")
        if not provider or not isinstance(key, str) or not key.strip():
            return {
                "ok": False,
                "message": "请选择提供方并填写密钥．",
                "refresh": False,
            }
        try:
            self._call(IpcMethod.CREDENTIALS, provider=provider, key=key)
        except Exception:
            # Provider errors can contain request data. Never echo credential values.
            logger.error("failed to save provider credentials")
            return {
                "ok": False,
                "message": "密钥保存失败．检查提供方配置后重试．",
                "refresh": False,
            }
        self._refresh()
        return {"ok": True, "message": "已保存密钥．", "refresh": True}

    def _handle_logout(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(_values(payload).get("provider") or "")
        if not provider:
            return {"ok": False, "message": "请选择提供方．", "refresh": False}
        try:
            with self._login_lock:
                login = self._login
            if login is not None and login.provider == provider:
                self._cancel_login()
                self._login_status = ""
            self._call(IpcMethod.LOGOUT, provider=provider)
        except Exception:
            return {"ok": False, "message": "凭据删除失败．请重试．", "refresh": False}
        self._refresh()
        return {"ok": True, "message": "已删除凭据．", "refresh": True}

    # --------------------------------------------------------- opencode login

    def _login_state(self) -> dict[str, Any] | None:
        """What the settings page needs to finish an authorization in flight."""

        with self._login_lock:
            login = self._login
            if login is None:
                return None
            remaining = login.expires_at - time.monotonic()
            if remaining <= 0:
                return None
            return {
                "provider": login.provider,
                "uri": login.flow.verification_uri_complete,
                "code": login.flow.user_code,
                "valid_for": float(login.flow.expires_in),
            }

    def _cancel_login(self) -> None:
        with self._login_lock:
            login = self._login
            self._login = None
        if login is not None:
            login.cancel.set()

    def _handle_opencode_login(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = _values(payload)
        provider = str(values.get("provider") or "")
        force = bool(values.get("force"))
        try:
            allowed = {
                item["id"]
                for item in self._call(IpcMethod.INFO).get("providers", [])
                if item.get("auth") == ProviderAuth.OAUTH
            }
            if provider not in allowed:
                return {
                    "ok": False,
                    "message": "请选择 OpenCode 提供方．",
                    "refresh": False,
                }
            with self._login_lock:
                login = self._login
                active = (
                    login is not None
                    and not force
                    and login.expires_at > time.monotonic()
                )
                if active and login is not None and login.provider == provider:
                    # The link lives on the settings page, not in this toast.
                    return {
                        "ok": True,
                        "message": "授权仍在进行．授权链接与验证码见「提供方」页顶部．",
                        "refresh": True,
                    }
                if active and login is not None:
                    return {
                        "ok": True,
                        "message": (
                            f"`{login.provider}` 的授权仍在进行，见「提供方」页顶部．"
                            f"如需改为授权 `{provider}`，请先点「取消授权」．"
                        ),
                        "refresh": True,
                    }
                if login is not None:
                    login.cancel.set()
                    self._login = None
            flow = start_device_flow()
            with self._login_lock:
                self._login = LoginFlow(
                    provider=provider,
                    flow=flow,
                    cancel=threading.Event(),
                    expires_at=time.monotonic() + float(flow.expires_in),
                )
                self._login_status = "等待 OpenCode 授权．"
                login = self._login
                login.thread = threading.Thread(
                    target=self._poll_login,
                    args=(login,),
                    name="opencode-device-login",
                    daemon=True,
                )
                login.thread.start()
        except Exception as exc:  # noqa: BLE001 - shown as a toast
            logger.warning("failed to start the OpenCode device flow: %s", exc)
            self._cancel_login()
            return {
                "ok": False,
                "message": "无法启动 OpenCode 授权．检查网络后重试．",
                "refresh": True,
            }
        # Best effort only: the server usually has no browser, so the page link
        # is the real path to the authorization.
        try:
            webbrowser.open(flow.verification_uri_complete)
        except Exception:  # noqa: BLE001 - an unavailable browser is expected
            logger.debug("no browser available to open the OpenCode authorization")
        self._refresh()
        return {
            "ok": True,
            "message": "已生成授权链接．若浏览器没有打开新标签页，请用「提供方」页顶部的链接．",
            "refresh": True,
            # The browser tries a new tab; the page keeps the link as a fallback
            # because popup blockers and remote servers both happen.
            "open_url": flow.verification_uri_complete,
        }

    def _handle_opencode_cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._login_lock:
            login = self._login
        if login is None:
            return {"ok": True, "message": "当前没有进行中的授权．", "refresh": True}
        self._cancel_login()
        self._login_status = "已取消授权．"
        self._refresh()
        return {"ok": True, "message": "已取消授权．", "refresh": True}

    def _poll_login(self, login: LoginFlow) -> None:
        flow = login.flow
        try:
            while time.monotonic() < login.expires_at and not login.cancel.is_set():
                tokens = poll_device_token(flow)
                if tokens is not None:
                    with self._login_lock:
                        if login.cancel.is_set():
                            return
                        self._call(
                            IpcMethod.IMPORT_CREDENTIALS,
                            provider=login.provider,
                            credential={
                                "type": CredentialType.OAUTH,
                                "access": tokens.access_token,
                                "refresh": tokens.refresh_token,
                                "expires": tokens.expires_at_ms,
                                "org_id": tokens.org_id,
                            },
                        )
                        self._login_status = "已关联 OpenCode 套餐．"
                    return
                login.cancel.wait(max(1, int(flow.interval)))
            if not login.cancel.is_set():
                self._login_status = "OpenCode 授权超时．请重新关联．"
        except (OpenCodeGoError, RuntimeError):
            self._login_status = "OpenCode 授权失败．请重新关联．"
        except Exception:
            logger.exception("OpenCode device login failed")
            self._login_status = "OpenCode 凭据保存失败．请重新关联．"
        finally:
            with self._login_lock:
                if self._login is login:
                    self._login = None
            self._refresh()
