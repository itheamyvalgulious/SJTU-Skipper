"""Offline end-to-end demo plugin."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from core.plugin import Plugin, PluginContext
from .script import demo_transcript

logger = logging.getLogger("lecture_agent.plugin.demo")


class DemoPlugin(Plugin):
    id = "demo"
    requires = ("settings", "llm", "llm-agent", "summarize")
    provides = ("demo",)

    def __init__(self) -> None:
        self.ctx: PluginContext | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._log: list[str] = []
        self._session_id: str | None = None
        self._lines = 0

    # ------------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._configure_demo_settings()
        ctx.ui.add_tab("left", "demo", "演示", icon="✦", order=90, plugin_id=self.id)
        ctx.ui.add_toolbox_item(
            "demo",
            "demo.controls",
            "离线演示",
            description="无需网络，合成一段课堂并跑通转写 -> 笔记链路",
            render=self._render_controls,
            plugin_id=self.id,
        )
        ctx.ui.add_page(
            "demo:main",
            "演示课堂",
            icon="✦",
            render=self._render_page,
            closable=True,
            plugin_id=self.id,
        )
        ctx.ui.register_action("demo.start", self._action_start, plugin_id=self.id)
        ctx.ui.register_action("demo.stop", self._action_stop, plugin_id=self.id)

    def teardown(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    # ------------------------------------------------------------------
    def _configure_demo_settings(self) -> None:
        assert self.ctx is not None
        try:
            settings = self.ctx.require("settings")
        except Exception:
            return
        overrides = {
            "stt.provider": "mock",
            "stt.enabled": True,
            "vlm.enabled": False,
            "summarize.trigger_chars": 260,
            "summarize.trigger_audio_seconds": 999999,
            "summarize.min_chars": 60,
        }
        for key, value in overrides.items():
            try:
                settings.set(key, value)
            except Exception:
                logger.debug("demo setting %s could not be applied", key, exc_info=True)

    def _action_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.ctx is not None
        if self._thread and self._thread.is_alive():
            return {"ok": False, "message": "演示已经在运行"}
        self._stop.clear()
        self._lines = 0
        session = self.ctx.streams.start_session(
            meta={
                "course": "演示课堂 · 复分析",
                "lecture": "留数定理",
                "demo": True,
                "started_at": time.time(),
            }
        )
        self._session_id = session.session_id
        self._log.append(f"开始演示会话 {session.session_id}")
        try:
            self.ctx.ui.open_page("demo:main")
        except Exception:
            pass
        self._thread = threading.Thread(
            target=self._run_demo, args=(session.session_id,), name="demo-script", daemon=True
        )
        self._thread.start()
        self.ctx.ui.refresh("demo-start")
        return {"ok": True, "message": "演示已开始", "open_page": "demo:main"}

    def _action_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.ctx is not None
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._session_id:
            self.ctx.streams.end_session("demo stopped")
            self._log.append("演示已停止")
        self.ctx.ui.refresh("demo-stop")
        return {"ok": True, "message": "演示已停止"}

    def _run_demo(self, session_id: str) -> None:
        assert self.ctx is not None
        lines = demo_transcript(1100)
        for index, line in enumerate(lines):
            if self._stop.is_set():
                return
            session = self.ctx.streams.get_session(session_id)
            if session is None:
                return
            session.transcript.append(
                line.text,
                start=float(index),
                end=float(index) + 0.8,
                source=line.source,
                meta={"category": line.category, "demo": True},
            )
            if index % 6 == 0:
                session.visual.append(
                    "【板书】板书：留数定理与围道积分，"
                    "$\\int_C f(z)\\,dz=2\\pi i\\sum_k \\operatorname{Res}(f,z_k)$。",
                    start=float(index),
                    end=float(index) + 0.8,
                    source="vlm",
                    meta={"demo": True},
                )
            self._lines = index + 1
            if index % 4 == 0:
                self._log.append(f"[{index:03d}] {line.text[:36]}…")
                self.ctx.ui.refresh("demo-script")
            time.sleep(0.06)
        self._log.append("演示脚本播放完毕，等待总结 Agent 收尾")
        self.ctx.streams.end_session("demo finished")
        self.ctx.ui.refresh("demo-script-done")

    # ------------------------------------------------------------------
    def _notes(self) -> dict[str, str]:
        if not self._session_id:
            return {}
        root = self.ctx.data_dir / "sessions" / self._session_id / "notes" if self.ctx else None
        if root is None:
            return {}
        out = {}
        for name in ("knowledge.md", "gossip.md", "notice.md"):
            path = root / name
            out[name] = path.read_text(encoding="utf-8") if path.exists() else ""
        return out

    def _status(self) -> dict[str, Any]:
        notes = self._notes()
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "session_id": self._session_id,
            "lines": self._lines,
            "knowledge_chars": len(notes.get("knowledge.md", "")),
            "gossip_chars": len(notes.get("gossip.md", "")),
            "notice_chars": len(notes.get("notice.md", "")),
        }

    def _render_controls(self) -> list[dict[str, Any]]:
        status = self._status()
        return [
            {
                "type": "keyvalue",
                "items": [
                    ["状态", "运行中" if status["running"] else "空闲"],
                    ["已播放", f"{status['lines']} 段"],
                ],
            },
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "开始演示",
                        "action": "demo.start",
                        "class": "btn btn-primary",
                    },
                    {"label": "停止", "action": "demo.stop", "class": "btn btn-danger"},
                ],
            },
        ]

    def _render_page(self) -> list[dict[str, Any]]:
        status = self._status()
        blocks: list[dict[str, Any]] = [
            {
                "type": "keyvalue",
                "items": [
                    ["会话", status["session_id"] or "—"],
                    ["运行", "是" if status["running"] else "否"],
                    ["已播放片段", status["lines"]],
                    ["knowledge 字数", status["knowledge_chars"]],
                    ["gossip 字数", status["gossip_chars"]],
                    ["notice 字数", status["notice_chars"]],
                ],
            },
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "开始演示",
                        "action": "demo.start",
                        "class": "btn btn-primary",
                    },
                    {"label": "停止", "action": "demo.stop", "class": "btn btn-danger"},
                ],
            },
        ]
        notes = self._notes()
        if any(notes.values()):
            blocks.append(
                {
                    "type": "tabs",
                    "tabs": [
                        {"title": "知识", "blocks": [{"type": "markdown", "text": notes["knowledge.md"]}]},
                        {"title": "闲聊", "blocks": [{"type": "markdown", "text": notes["gossip.md"]}]},
                        {"title": "通知", "blocks": [{"type": "markdown", "text": notes["notice.md"]}]},
                    ],
                }
            )
        blocks.append({"type": "log", "lines": self._log[-80:]})
        return blocks
