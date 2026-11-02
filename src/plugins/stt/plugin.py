"""STT placeholder plugin.

The project does not currently implement speech recognition.  This plugin keeps
the audio-stream-facing surface alive (status + UI) so the rest of the pipeline
runs normally, but it deliberately emits no text and registers no STT settings.
"""

from __future__ import annotations

import logging
from typing import Any

from core.plugin import Plugin, PluginContext

LOGGER_NAME = "lecture_agent.plugin.stt"

TAB_ID = "classroom"
ITEM_ID = "stt.transcript"
PAGE_ID = "stt:transcript"


class STTService:
    """Public ``stt`` service; currently exposes status only."""

    def __init__(self, plugin: "SttPlugin") -> None:
        self._plugin = plugin

    def status(self) -> dict[str, Any]:
        return self._plugin.status()


class SttPlugin(Plugin):
    """Audio-stream placeholder: no backend, no settings, no text output."""

    id = "stt"
    requires: tuple[str, ...] = ()
    provides = ("stt",)

    def __init__(self) -> None:
        self._ctx: PluginContext | None = None
        self._logger = logging.getLogger(LOGGER_NAME)
        self._page_added = False
        self._item_added = False
        self._tab_added = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        ctx.register_service("stt", STTService(self))
        ctx.meta.register("stt", self._meta, plugin_id=self.id)
        self._setup_ui(ctx)
        ctx.logger.info("stt ready: no backend, no text output")

    def teardown(self) -> None:
        ctx = self._ctx
        if ctx is not None:
            if self._page_added:
                try:
                    ctx.ui.remove_page(PAGE_ID)
                except Exception:  # noqa: BLE001 - UI may already be gone
                    pass
                self._page_added = False
            if self._item_added:
                try:
                    ctx.ui.remove_toolbox_item(ITEM_ID)
                except Exception:  # noqa: BLE001
                    pass
                self._item_added = False
            if self._tab_added:
                try:
                    ctx.ui.remove_tab(TAB_ID)
                except Exception:  # noqa: BLE001
                    pass
                self._tab_added = False
        self._ctx = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        ctx = self._ctx
        session = ctx.streams.current if ctx is not None else None
        return {
            "state": "unavailable",
            "error": "未接入 STT 后端",
            "backend": "none",
            "session_id": session.session_id if session is not None else "",
            "audio_seconds": (
                round(session.audio.duration, 3) if session is not None else 0.0
            ),
            "segments": 0,
            "chars": 0,
        }

    def _meta(self) -> dict[str, Any]:
        status = self.status()
        return {
            "backend": "none",
            "state": status["state"],
            "session_id": status["session_id"],
            "audio_seconds": status["audio_seconds"],
        }

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self, ctx: PluginContext) -> None:
        ui = ctx.ui
        try:
            ui.add_tab(
                "left",
                TAB_ID,
                "课堂",
                plugin_id=self.id,
            )
            self._tab_added = True
        except ValueError:
            # Another classroom plugin already created the tab.
            pass
        try:
            ui.add_toolbox_item(
                TAB_ID,
                ITEM_ID,
                "转写",
                description="当前未接入 STT 后端",
                order=20,
                render=self._render_item,
                plugin_id=self.id,
            )
            self._item_added = True
        except ValueError:
            pass
        ui.add_page(
            PAGE_ID,
            "实时转写",
            render=self._render_page,
            icon="mic",
            order=20,
            plugin_id=self.id,
        )
        self._page_added = True

    def _render_item(self) -> list[dict[str, Any]]:
        return [
            {"type": "keyvalue", "items": self._status_rows()},
            {
                "type": "markdown",
                "text": (
                    "当前版本没有 STT 后端。音频流会正常播放，"
                    "但不会产生文字。"
                ),
            },
        ]

    def _render_page(self) -> list[dict[str, Any]]:
        return self._render_item()

    def _status_rows(self) -> list[list[str]]:
        status = self.status()
        return [
            ["状态", "未接入 STT 后端"],
            ["后端", "未接入"],
            ["会话", status["session_id"] or "—"],
            ["音频", f"{float(status['audio_seconds']):.1f}s"],
            ["已转写", "0 段 / 0 字"],
        ]


# Backwards-compatible alias for callers that used the old class name.
STTPlugin = SttPlugin


__all__ = [
    "ITEM_ID",
    "PAGE_ID",
    "STTPlugin",
    "STTService",
    "SttPlugin",
    "TAB_ID",
]
