"""Input plugin: Canvas/SJTU video portal → ffmpeg → audio/video streams.

Registers settings, service ``"input"``, UI tabs/pages, and actions.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from core.plugin import Plugin, PluginContext
from integrations.sjtu_canvas import InputError
from plugins.settings.spec import SettingsSpec
from .catalog import Catalog
from .sources import SourceManager, SourceSpec

_log = logging.getLogger("lecture_agent.plugins.input")

_LOCAL_TZ = datetime.now().astimezone().tzinfo


def _local_time(iso_string: str) -> str:
    """Convert an ISO-ish timestamp to local time display."""
    if not iso_string:
        return ""
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(iso_string.strip(), fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            # SJTU/SJTU video API timestamps are already campus-local.
            return dt.strftime("%m-%d %H:%M")
        local = dt.astimezone(_LOCAL_TZ)
        return local.strftime("%m-%d %H:%M")
    return iso_string


def _build_course_page(
    ctx: PluginContext,
    course_id: str,
    catalog: Catalog,
    course_name: str = "",
) -> list[dict[str, Any]]:
    """Render a single course's recordings + live table."""
    blocks: list[dict[str, Any]] = []

    title = course_name or f"课程 {course_id}"
    blocks.append({"type": "markdown", "text": f"## {title}\n\n"})

    # ---- Recordings ----
    blocks.append({"type": "markdown", "text": "### 📼 录播回放\n"})
    try:
        recs = catalog.list_recordings(course_id)
    except InputError as exc:
        blocks.append({"type": "text", "text": f"⚠️ {exc}"})
        recs = []

    if not recs:
        blocks.append({"type": "text", "text": "暂无录播记录。"})
    else:
        rows: list[str] = []
        rows.append("| 名称 | 日期 | 教室 | 操作 |")
        rows.append("|------|------|------|------|")
        for r in recs:
            name = r.get("name", "?")
            date = _local_time(r.get("begin", ""))
            classroom = r.get("classroom", "")
            vid = r.get("video_id", "")
            btn = (
                f'<button class="input-action" '
                f'data-action="input.play_recording" '
                f'data-course="{course_id}" '
                f'data-video="{vid}">▶ 播放</button>'
            )
            rows.append(f"| {name} | {date} | {classroom} | {btn} |")
        blocks.append({"type": "markdown", "text": "\n".join(rows)})

    # ---- Live section ----
    blocks.append({"type": "markdown", "text": "\n### 🔴 直播\n"})
    now = datetime.now()
    start_str = now.strftime("%Y-%m-%d 00:00:00")
    end_str = now.strftime("%Y-%m-%d 23:59:59")
    try:
        lives = catalog.list_lives(start_str, end_str)
    except InputError as exc:
        blocks.append({"type": "text", "text": f"⚠️ {exc}"})
        lives = []

    if not lives:
        blocks.append({"type": "text", "text": "今日暂无直播。"})
    else:
        rows2: list[str] = []
        rows2.append("| 名称 | 时间 | 教室 | 操作 |")
        rows2.append("|------|------|------|------|")
        for lv in lives:
            name = lv.get("name", "?")
            begin = _local_time(lv.get("begin", ""))
            classroom = lv.get("classroom", "")
            lid = lv.get("id", "")
            btn = (
                f'<button class="input-action" '
                f'data-action="input.watch_live" '
                f'data-live="{lid}">🔴 观看</button>'
            )
            rows2.append(f"| {name} | {begin} | {classroom} | {btn} |")
        blocks.append({"type": "markdown", "text": "\n".join(rows2)})

    return blocks


class InputPlugin(Plugin):
    id = "input"
    requires = ("settings",)
    provides = ("input",)

    def setup(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        settings = ctx.require("settings")

        # ---- Register settings ----
        for spec in _INPUT_SETTINGS:
            settings.register(spec)

        # ---- Build catalog ----
        self._catalog = Catalog(settings=settings)

        # ---- Build source manager ----
        self._sources = SourceManager(
            hub=ctx.streams,
            emit=ctx.events.emit,
            sample_rate=int(settings.get("input.audio_rate", 16000)),
            video_fps=float(settings.get("input.video_fps", 4.0)),
            video_width=int(settings.get("input.video_width", 0)),
            playback_speed=float(settings.get("input.playback_speed", 1.0)),
            reconnect=bool(settings.get("input.reconnect", True)),
        )

        # ---- Register input service ----
        ctx.register_service("input", self._create_service())

        def _input_meta() -> dict[str, Any]:
            session = ctx.streams.current
            return {
                "course": session.meta.get("course", ""),
                "lecture": session.meta.get("lecture", ""),
                "session_id": session.session_id,
                "source": session.meta.get("title", ""),
                "kind": session.meta.get("kind", ""),
                "audio_seconds": session.audio.duration,
                "video_seconds": session.video.duration,
            }

        ctx.meta.register("input", _input_meta, plugin_id=self.id)

        # ---- Register UI ----
        self._setup_ui(ctx)

        ctx.logger.info("Input plugin ready")

    def teardown(self) -> None:
        self._sources.stop()
        self._catalog.close()

    # ------------------------------------------------------------------
    # Service object
    # ------------------------------------------------------------------

    def _create_service(self) -> Any:
        mgr = self._sources
        cat = self._catalog

        class InputService:
            """Service facade for other plugins (mostly demo/agent)."""

            @property
            def status(self):
                return mgr.status()

            def start(self, spec: SourceSpec) -> None:
                return mgr.start(spec)

            def stop(self) -> None:
                return mgr.stop()

            def pause(self) -> None:
                return mgr.pause()

            def resume(self) -> None:
                return mgr.resume()

            def list_courses(self) -> list[dict[str, Any]]:
                return cat.list_courses()

            def list_recordings(
                self, course_id: str, limit: int = 50
            ) -> list[dict[str, Any]]:
                return cat.list_recordings(course_id, limit=limit)

            def get_recording_url(self, course_id: str, video_id: str) -> str:
                return cat.get_recording_url(course_id, video_id)

            def list_lives(self, start: str, end: str) -> list[dict[str, Any]]:
                return cat.list_lives(start, end)

            def get_live_devices(self, live_id: str) -> list[dict[str, Any]]:
                return cat.get_live_devices(live_id)

        return InputService()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self, ctx: PluginContext) -> None:
        ui = ctx.ui
        pid = self.id

        # Tab
        ui.add_tab("left", "courses", "课程", icon="📚", plugin_id=pid)

        # Toolbox items
        ui.add_toolbox_item(
            "courses",
            "input.courses",
            "我的课程",
            description="浏览 Canvas 课程与录播/直播",
            order=0,
            render=self._render_courses_list,
            plugin_id=pid,
        )
        ui.add_toolbox_item(
            "courses",
            "input.manual",
            "手动 URL",
            description="手动输入直播/录播地址",
            order=1,
            render=self._render_manual_form,
            plugin_id=pid,
        )
        ui.add_toolbox_item(
            "courses",
            "input.session",
            "当前会话",
            description="查看/停止当前播放",
            order=2,
            render=self._render_session,
            plugin_id=pid,
        )

        # Pages
        ui.add_page(
            "input:session",
            "播放状态",
            render=lambda: self._render_session(),
            plugin_id=pid,
        )

        # ---- Actions ----
        ui.register_action("input.refresh_courses", self._act_refresh_courses, plugin_id=pid)
        ui.register_action("input.open_course", self._act_open_course, plugin_id=pid)
        ui.register_action("input.play_recording", self._act_play_recording, plugin_id=pid)
        ui.register_action("input.watch_live", self._act_watch_live, plugin_id=pid)
        ui.register_action("input.play_url", self._act_play_url, plugin_id=pid)
        ui.register_action("input.stop", self._act_stop, plugin_id=pid)
        ui.register_action("input.pause", self._act_pause, plugin_id=pid)
        ui.register_action("input.resume", self._act_resume, plugin_id=pid)

    # ------------------------------------------------------------------
    # Toolbox renders
    # ------------------------------------------------------------------

    def _render_courses_list(self) -> list[dict[str, Any]]:
        catalog = self._catalog
        if not catalog.has_token():
            return [
                {
                    "type": "markdown",
                    "text": (
                        "⚠️ **Canvas Token 未配置**\n\n"
                        "请在设置 → **Canvas** 分组中填写 `input.canvas_token`，"
                        "或将 `CANVAS_TOKEN` 写入 `.env`。"
                    ),
                }
            ]

        blocks: list[dict[str, Any]] = []
        blocks.append(
            {
                "type": "markdown",
                "text": "### 📚 我的课程\n",
            }
        )
        blocks.append(
            {
                "type": "html",
                "html": (
                    '<button class="input-action" '
                    'data-action="input.refresh_courses">🔄 刷新</button>'
                ),
            }
        )

        try:
            courses = catalog.list_courses()
        except InputError as exc:
            blocks.append({"type": "text", "text": f"⚠️ {exc}"})
            return blocks

        if not courses:
            blocks.append({"type": "text", "text": "没有找到课程。请确认 Canvas token 是否正确。"})
            return blocks

        cards: list[dict[str, Any]] = []
        for c in courses:
            name = c.get("name", "?")
            term = c.get("term", "")
            teacher = c.get("teacher", "")
            cid = c.get("id", "")
            subtitle = " · ".join(part for part in (term, teacher) if part)
            cards.append(
                {
                    "title": name,
                    "subtitle": subtitle,
                    "actions": [
                        {
                            "label": "📂 打开",
                            "action": "input.open_course",
                            "value": {"course": cid, "name": name},
                            "class": "btn btn-sm",
                        }
                    ],
                }
            )
        blocks.append({"type": "cards", "items": cards})
        return blocks

    def _render_manual_form(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "markdown",
                "text": "### 🔗 手动输入 URL\n",
            },
            {
                "type": "form",
                "fields": [
                    {
                        "key": "url",
                        "label": "视频地址",
                        "type": "string",
                        "placeholder": "https://... 或 rtsp://...",
                    },
                    {
                        "key": "title",
                        "label": "标题",
                        "type": "string",
                        "placeholder": "可选",
                    },
                    {
                        "key": "kind",
                        "label": "类型",
                        "type": "select",
                        "options": [
                            {"value": "url", "label": "自动检测"},
                            {"value": "live", "label": "直播 (低延迟)"},
                            {"value": "recording", "label": "录播 (-re 节奏)"},
                        ],
                        "default": "url",
                    },
                ],
                "action": "input.play_url",
            },
        ]

    def _render_session(self) -> list[dict[str, Any]]:
        status = self._sources.status()
        state = status.get("state", "idle")
        source = status.get("source") or {}
        blocks: list[dict[str, Any]] = []

        mapping = {
            "idle": "⏹️ 空闲",
            "playing": "▶️ 播放中",
            "paused": "⏸️ 已暂停",
            "stopped": "⏹️ 已停止",
        }
        blocks.append(
            {
                "type": "markdown",
                "text": f"### {mapping.get(str(state), state)}\n",
            }
        )

        if source:
            blocks.append(
                {
                    "type": "markdown",
                    "text": (
                        f"- 标题: **{source.get('title', '-')}**\n"
                        f"- 类型: {source.get('kind', '-')}\n"
                        f"- URL: `{source.get('url', '-')[:80]}`\n"
                    ),
                }
            )

        if state in ("playing", "paused"):
            blocks.append(
                {
                    "type": "html",
                    "html": (
                        '<button class="input-action" '
                        'data-action="input.pause">⏸️ 暂停</button> '
                        '<button class="input-action" '
                        'data-action="input.resume">▶️ 继续</button> '
                        '<button class="input-action" '
                        'data-action="input.stop">⏹️ 停止</button>'
                    ),
                }
            )
        elif state == "stopped":
            blocks.append(
                {
                    "type": "html",
                    "html": (
                        '<button class="input-action" '
                        'data-action="input.stop">🗑️ 清除</button>'
                    ),
                }
            )
        return blocks

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _act_refresh_courses(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._catalog.reset()
        return {"ok": True}

    def _act_open_course(self, payload: dict[str, Any]) -> dict[str, Any]:
        course_id = str(payload.get("course", "") or payload.get("course_id", ""))
        if not course_id:
            return {"ok": False, "message": "缺少课程 ID"}

        course_name = str(payload.get("name", ""))
        page_id = f"input:course:{course_id}"
        ctx = self._ctx

        # Create page if not exists
        try:
            ctx.ui.add_page(
                page_id,
                f"课程: {course_name or course_id}",
                render=lambda cid=course_id: _build_course_page(
                    ctx, cid, self._catalog, course_name
                ),
                plugin_id=self.id,
            )
        except ValueError:
            # Already exists, update renderer
            ctx.ui.set_page_render(
                page_id,
                lambda cid=course_id: _build_course_page(
                    ctx, cid, self._catalog, course_name
                ),
            )
        return {"ok": True, "open_page": page_id}

    def _act_play_recording(self, payload: dict[str, Any]) -> dict[str, Any]:
        course_id = str(payload.get("course", ""))
        video_id = str(payload.get("video", ""))
        if not course_id or not video_id:
            return {"ok": False, "message": "缺少课程或视频 ID"}

        def _bg() -> None:
            try:
                url = self._catalog.get_recording_url(course_id, video_id)
            except InputError as exc:
                self._ctx.events.emit(
                    "input.error",
                    message=str(exc),
                    source=f"course {course_id} video {video_id}",
                )
                return
            spec = SourceSpec(
                kind="recording",
                url=url,
                title=f"录播 {video_id}",
                course={"id": course_id, "video_id": video_id},
            )
            self._sources.start(spec)

        threading.Thread(target=_bg, daemon=True).start()
        return {"ok": True, "message": "正在获取播放地址…", "open_page": "input:session"}

    def _act_watch_live(self, payload: dict[str, Any]) -> dict[str, Any]:
        live_id = str(payload.get("live", ""))
        if not live_id:
            return {"ok": False, "message": "缺少直播 ID"}

        device_pref = self._ctx.require("settings").get("input.live_device", "0")

        def _bg() -> None:
            try:
                devices = self._catalog.get_live_devices(live_id)
            except InputError as exc:
                self._ctx.events.emit(
                    "input.error",
                    message=str(exc),
                    source=f"live {live_id}",
                )
                return

            url = ""
            name = ""
            if devices:
                idx = 0
                try:
                    idx = max(0, int(device_pref))
                except (ValueError, TypeError):
                    pass
                if idx < len(devices):
                    url = devices[idx].get("url", "")
                    name = devices[idx].get("name", "")
                else:
                    url = devices[0].get("url", "")
                    name = devices[0].get("name", "")

            if not url:
                self._ctx.events.emit(
                    "input.error",
                    message=f"未找到直播 {live_id} 的可用设备流。",
                    source=f"live {live_id}",
                )
                return

            spec = SourceSpec(
                kind="live",
                url=url,
                title=name or f"直播 {live_id}",
                course={"live_id": live_id},
            )
            self._sources.start(spec)

        threading.Thread(target=_bg, daemon=True).start()
        return {"ok": True, "message": "正在获取直播地址…", "open_page": "input:session"}

    def _act_play_url(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = str(payload.get("url", "")).strip()
        if not url:
            return {"ok": False, "message": "请输入 URL"}
        title = str(payload.get("title", "")).strip() or url[:60]
        kind_raw = str(payload.get("kind", "url"))
        # Map "url" -> auto-detect; we just pass it through
        kind: Any = kind_raw if kind_raw in ("live", "recording") else "url"

        spec = SourceSpec(
            kind=kind,
            url=url,
            title=title,
        )
        self._sources.start(spec)
        return {"ok": True, "message": "开始播放", "open_page": "input:session"}

    def _act_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._sources.stop()
        return {"ok": True, "message": "已停止"}

    def _act_pause(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._sources.pause()
        return {"ok": True, "message": "已暂停"}

    def _act_resume(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._sources.resume()
        return {"ok": True, "message": "已继续"}


# ---------------------------------------------------------------------------
# Settings specs
# ---------------------------------------------------------------------------

_INPUT_SETTINGS: list[SettingsSpec] = [
    SettingsSpec(
        key="input.canvas_token",
        label="Canvas Token",
        type="secret",
        default="",
        group="Canvas",
        description="SJTU Canvas 访问令牌（Bearer Token）",
        order=0,
        secret=True,
        placeholder="粘贴 Canvas 访问令牌…",
    ),
    SettingsSpec(
        key="input.canvas_base_url",
        label="Canvas 地址",
        type="string",
        default="https://oc.sjtu.edu.cn",
        group="Canvas",
        description="Canvas 实例地址",
        order=1,
    ),
    SettingsSpec(
        key="input.video_fps",
        label="视频帧率",
        type="float",
        default=4.0,
        group="播放",
        description="每秒提取的视频帧数（原管线为 4fps）",
        order=10,
    ),
    SettingsSpec(
        key="input.video_width",
        label="视频宽度",
        type="int",
        default=0,
        group="播放",
        description="视频输出宽度；0 表示保持原分辨率，不做缩放",
        order=11,
    ),
    SettingsSpec(
        key="input.audio_rate",
        label="音频采样率",
        type="int",
        default=16000,
        group="播放",
        description="音频采样率（Hz）",
        order=12,
    ),
    SettingsSpec(
        key="input.playback_speed",
        label="播放速度",
        type="float",
        default=1.0,
        group="播放",
        description="播放速度倍率（1.0 = 正常）",
        order=13,
    ),
    SettingsSpec(
        key="input.live_device",
        label="直播设备",
        type="select",
        default="0",
        group="直播",
        description="默认使用的直播设备索引",
        options=("0", "1", "2", "3"),
        order=20,
    ),
    SettingsSpec(
        key="input.reconnect",
        label="自动重连",
        type="bool",
        default=True,
        group="直播",
        description="断线时自动重连",
        order=21,
    ),
    SettingsSpec(
        key="input.enabled",
        label="启用输入插件",
        type="bool",
        default=True,
        group="General",
        description="是否在启动时加载输入插件",
        order=0,
    ),
]