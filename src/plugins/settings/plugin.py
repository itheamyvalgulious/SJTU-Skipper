"""The built-in ``settings`` plugin: store, service, form UI."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Mapping

from core.plugin import Plugin, PluginContext
from .spec import SECRET_MASK, SettingsSpec
from .store import SettingsService, SettingsStore

logger = logging.getLogger("lecture_agent.plugins.settings")

TAB_ID = "settings"
PAGE_ID = "settings:main"
SAVE_ACTION = "settings.save"

_INPUT_KIND = {
    "string": "text",
    "text": "textarea",
    "int": "number",
    "float": "number",
    "bool": "checkbox",
    "select": "select",
    "secret": "password",
    "json": "textarea",
}


class SettingsPlugin(Plugin):
    """Registers the ``settings`` service and renders the settings UI."""

    id = "settings"
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ("settings",)

    def __init__(self) -> None:
        self.ctx: PluginContext | None = None
        self.store: SettingsStore | None = None
        self.service: SettingsService | None = None
        self._listener = None
        self._lock = threading.RLock()
        self._items: dict[str, str] = {}
        self._tab_added = False
        self._page_added = False
        self._action_added = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.store = SettingsStore(ctx.data_dir / "settings.json", logger_=ctx.logger)
        self.service = SettingsService(
            self.store, emit=ctx.events.emit, logger_=ctx.logger
        )
        ctx.register_service("settings", self.service)

        self._listener = self.service.add_listener(self._on_settings_changed)

        ui = ctx.ui
        ui.add_tab(
            "right",
            TAB_ID,
            "设置",
            icon="⚙",
            order=900,
            plugin_id=self.id,
        )
        self._tab_added = True
        ui.add_page(
            PAGE_ID,
            "设置",
            render=self._render_page,
            icon="⚙",
            order=900,
            plugin_id=self.id,
        )
        self._page_added = True
        ui.register_action(SAVE_ACTION, self._handle_save, plugin_id=self.id)
        self._action_added = True

        self._ensure_sidebar_item()
        ctx.logger.debug("settings plugin ready: %s", self.store.path)

    def teardown(self) -> None:
        if self.service is not None and self._listener is not None:
            self.service.remove_listener(self._listener)
            self._listener = None
        ctx = self.ctx
        if ctx is not None:
            if self._action_added:
                try:
                    ctx.ui.unregister_action(SAVE_ACTION)
                except Exception:  # noqa: BLE001 - UI may already be gone
                    pass
                self._action_added = False
            if self._page_added:
                try:
                    ctx.ui.remove_page(PAGE_ID)
                except Exception:  # noqa: BLE001
                    pass
                self._page_added = False
            try:
                ctx.ui.remove_toolbox_item("settings.open")
            except Exception:  # noqa: BLE001
                pass
            if self._tab_added:
                try:
                    ctx.ui.remove_tab(TAB_ID)
                except Exception:  # noqa: BLE001
                    pass
                self._tab_added = False
        with self._lock:
            self._items.clear()

    # ------------------------------------------------------------------
    # Toolbox / grouping
    # ------------------------------------------------------------------
    def _on_settings_changed(self, reason: str) -> None:
        ctx = self.ctx
        if ctx is not None:
            ctx.ui.refresh(f"settings:{reason}")

    def _spec_visible(self, spec: SettingsSpec) -> bool:
        """Resolve a spec's ``visible_when`` condition against live values."""

        if spec.visible_when is None:
            return True
        key, allowed = spec.visible_when
        service = self.service
        current = service.get(key) if service is not None else None
        return current in allowed

    def _grouped_specs(self) -> list[tuple[str, list[SettingsSpec]]]:
        service = self.service
        if service is None:
            return []
        grouped: dict[str, list[SettingsSpec]] = {}
        for spec in service.specs():
            if not self._spec_visible(spec):
                continue
            grouped.setdefault(spec.group, []).append(spec)
        return list(grouped.items())

    def _ensure_sidebar_item(self) -> None:
        """One compact settings entry; the page owns all group editing."""

        ctx = self.ctx
        if ctx is None:
            return
        with self._lock:
            if "open" in self._items:
                return
            self._items["open"] = "settings.open"
        try:
            ctx.ui.add_toolbox_item(
                TAB_ID,
                "settings.open",
                "设置",
                description="Canvas、LLM 连接、转写、板书与笔记",
                order=0,
                render=self._render_sidebar,
                plugin_id=self.id,
            )
        except Exception:  # noqa: BLE001 - duplicate/UI gone
            ctx.logger.exception("failed to add settings entry")

    def _render_sidebar(self) -> list[dict[str, Any]]:
        groups = self._grouped_specs()
        total = sum(len(specs) for _group, specs in groups)
        return [
            {
                "type": "markdown",
                "text": f"按分组编辑设置，保存后立即生效。当前 {len(groups)} 组 / {total} 项。",
            }
        ]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_page(self) -> list[dict[str, Any]]:
        groups = self._grouped_specs()
        if not groups:
            return [
                {
                    "type": "markdown",
                    "text": (
                        "### 设置\n\n"
                        "尚未注册任何设置项。加载 LLM、输入、STT、VLM 等插件后，"
                        "它们声明的设置会自动出现在这里。"
                    ),
                }
            ]
        tabs: list[dict[str, Any]] = []
        for group, specs in groups:
            blocks: list[dict[str, Any]] = []
            note = next((spec.group_note for spec in specs if spec.group_note), "")
            if note:
                blocks.append({"type": "markdown", "text": note})
            blocks.extend(self._form_blocks(specs, group))
            tabs.append({"title": group or "General", "blocks": blocks})
        return [{"type": "tabs", "id": "settings.groups", "tabs": tabs}]

    def _form_blocks(
        self, specs: list[SettingsSpec], group: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "form",
                "title": group or "General",
                "fields": [self._field(spec) for spec in specs],
                "submit": {"label": "保存", "action": SAVE_ACTION},
            }
        ]

    def _field(self, spec: SettingsSpec) -> dict[str, Any]:
        service = self.service
        raw = service.get(spec.key) if service is not None else spec.default
        value = self._form_value(spec, raw)
        options = [option for option, _label in spec.option_pairs()]
        return {
            "name": spec.key,
            "key": spec.key,
            "label": spec.label,
            "type": spec.type,
            "input": _INPUT_KIND.get(spec.type, "text"),
            "value": value,
            "options": options,
            "option_items": [
                {"value": option, "label": label}
                for option, label in spec.option_pairs()
            ],
            "placeholder": spec.placeholder,
            "description": spec.description,
            "secret": bool(spec.secret),
            "is_set": raw not in (None, ""),
            "order": spec.order,
        }

    def _form_value(self, spec: SettingsSpec, raw: Any) -> Any:
        """Serialize a value for the form (secrets masked, JSON as text)."""

        if spec.secret:
            return SECRET_MASK if raw not in (None, "") else ""
        if spec.type == "json":
            if raw is None:
                return ""
            try:
                return json.dumps(raw, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return str(raw)
        if spec.type == "bool":
            return bool(raw)
        return raw if raw is not None else ""

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _handle_save(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        service = self.service
        if service is None:
            return {"ok": False, "message": "设置服务尚未就绪", "refresh": True}
        if not isinstance(payload, Mapping):
            return {"ok": False, "message": "没有可保存的字段", "refresh": True}
        raw_values = payload.get("values")
        explicit_values = isinstance(raw_values, Mapping)
        if explicit_values:
            values: Mapping[str, Any] = raw_values
        else:
            # HTTP action payload is flat: {"llm.model": ..., "page_id": ...}.
            values = {
                key: value
                for key, value in payload.items()
                if key not in {"page_id", "item_id"}
            }
        if not values:
            if explicit_values:
                return {"ok": True, "message": "没有需要保存的修改", "refresh": True}
            return {"ok": False, "message": "没有可保存的字段", "refresh": True}
        saved: dict[str, Any] = {}
        errors: list[str] = []
        for key, value in values.items():
            if not isinstance(key, str):
                continue
            spec = service.get_spec(key)
            if spec is not None and spec.secret:
                if value == SECRET_MASK:
                    continue  # untouched masked secret -> keep current value
                if value is None or value == "":
                    # Clearing a secret removes the stored value instead of
                    # silently keeping it.
                    if service.get(key) not in (None, ""):
                        service.reset(key)
                        saved[key] = ""
                    continue
            try:
                saved[key] = service.set(key, value)
            except (ValueError, TypeError) as exc:
                label = spec.label if spec is not None else key
                errors.append(f"{label}：{exc}")
        if errors:
            message = "部分设置未保存：" + "；".join(errors)
            if saved:
                message += f"（已保存 {len(saved)} 项）"
            return {"ok": False, "message": message, "refresh": True}
        if not saved:
            return {"ok": True, "message": "没有需要保存的修改", "refresh": True}
        return {"ok": True, "message": f"已保存 {len(saved)} 项设置", "refresh": True}


__all__ = [
    "PAGE_ID",
    "SAVE_ACTION",
    "TAB_ID",
    "SettingsPlugin",
]
