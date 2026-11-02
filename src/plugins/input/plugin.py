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


def _live_state(begin: str, end: str) -> str:
    """Return 直播中 / 未开始 / 已结束 for a local-time range."""

    try:
        begin_dt = datetime.strptime(begin, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""
    now = datetime.now()
    if begin_dt <= now <= end_dt:
        return "直播中"
    if now < begin_dt:
        return "未开始"
    return "已结束"


def _build_course_page(
    ctx: PluginContext,
    course_id: str,
    catalog: Catalog,
    course_name: str = "",
) -> list[dict[str, Any]]:
    """Render one course's recordings only; live has its own page."""

    title = course_name or f"课程 {course_id}"
    blocks: list[dict[str, Any]] = [
        {"type": "markdown", "text": f"## {title}"},
        {
            "type": "keyvalue",
            "items": [
                ["课程 ID", course_id],
                ["内容", "录播回放"],
            ],
        },
        {"type": "markdown", "text": "### 📼 录播回放"},
    ]

    try:
        recs = catalog.list_recordings(course_id)
    except InputError as exc:
        blocks.append({"type": "text", "text": f"⚠️ {exc}"})
        recs = []

    if not recs:
        blocks.append({"type": "text", "text": "暂无录播记录。"})
    else:
        rows: list[dict[str, Any]] = []
        for record in recs:
            rows.append(
                {
                    "name": str(record.get("name", "?")),
                    "date": _local_time(str(record.get("begin", ""))),
                    "classroom": str(record.get("classroom", "")),
                    "course": course_id,
                    "video": str(record.get("video_id", "")),
                }
            )
        blocks.append(
            {
                "type": "table",
                "columns": [
                    {"key": "name", "label": "节次"},
                    {"key": "date", "label": "时间"},
                    {"key": "classroom", "label": "教室"},
                ],
                "rows": rows,
                "row_actions": [
                    {
                        "label": "▶ 播放录播",
                        "action": "input.play_recording",
                        "keys": ["course", "video"],
                        "class": "btn btn-sm btn-primary",
                    }
                ],
            }
        )
    blocks.append(
        {
            "type": "buttons",
            "buttons": [
                {
                    "label": "🔴 去直播",
                    "action": "ui.open_page",
                    "value": {"page_id": "input:live"},
                    "class": "btn btn-sm",
                }
            ],
        }
    )
    return blocks


def _build_live_page(ctx: PluginContext, catalog: Catalog) -> list[dict[str, Any]]:
    """Render live streams and upcoming streams as separate sections."""

    now = datetime.now()
    start_str = now.strftime("%Y-%m-%d 00:00:00")
    end_str = now.strftime("%Y-%m-%d 23:59:59")
    blocks: list[dict[str, Any]] = [{"type": "markdown", "text": "### 🔴 今日直播"}]
    try:
        lives = catalog.list_lives(start_str, end_str)
    except InputError as exc:
        blocks.append({"type": "text", "text": f"⚠️ {exc}"})
        lives = []

    live_rows: list[dict[str, Any]] = []
    upcoming_rows: list[dict[str, Any]] = []
    for live in lives:
        state = _live_state(str(live.get("begin", "")), str(live.get("end", "")))
        row = {
            "name": str(live.get("name", "?")),
            "time": _local_time(str(live.get("begin", ""))),
            "classroom": str(live.get("classroom", "")),
            "state": state,
            "live": str(live.get("id", "")),
        }
        if state == "直播中":
            live_rows.append(row)
        elif state == "未开始":
            upcoming_rows.append(row)

    columns = [
        {"key": "name", "label": "课程"},
        {"key": "time", "label": "时间"},
        {"key": "classroom", "label": "教室"},
        {"key": "state", "label": "状态"},
    ]
    if live_rows:
        blocks.append({"type": "markdown", "text": "#### 正在直播"})
        blocks.append(
            {
                "type": "table",
                "columns": columns,
                "rows": live_rows,
                "row_actions": [
                    {
                        "label": "🔴 观看直播",
                        "action": "input.watch_live",
                        "keys": ["live", "name"],
                        "class": "btn btn-sm btn-primary",
                    }
                ],
            }
        )
    if upcoming_rows:
        blocks.append({"type": "markdown", "text": "#### 即将开始"})
        blocks.append({"type": "table", "columns": columns, "rows": upcoming_rows})
    if not live_rows and not upcoming_rows:
        blocks.append({"type": "text", "text": "今日暂无正在或即将开始的直播。"})
    blocks.append(
        {
            "type": "buttons",
            "buttons": [
                {
                    "label": "🔄 刷新直播",
                    "action": "input.refresh_live",
                    "class": "btn btn-sm",
                },
                {
                    "label": "📚 去课程录播",
                    "action": "ui.open_page",
                    "value": {"page_id": "input:courses"},
                    "class": "btn btn-sm",
                },
            ],
        }
    )
    return blocks


class InputPlugin(Plugin):
    id = "input"
    requires = ("settings",)
    provides = ("input",)

    def setup(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._pending_message = ""
        self._last_error = ""
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
        ctx.events.on("input.status", self._on_input_event, plugin_id=self.id)
        ctx.events.on("input.error", self._on_input_event, plugin_id=self.id)

        ctx.logger.info("Input plugin ready")

    def teardown(self) -> None:
        self._sources.stop()
        self._catalog.close()

    def _on_input_event(self, event: Any) -> None:
        if event.type == "input.error":
            self._pending_message = ""
            self._last_error = str(event.get("message", "") or "输入源错误")
        elif event.type == "input.status":
            status = str(event.get("status", ""))
            if status in {"playing", "stopped"}:
                self._pending_message = ""
                if status == "playing":
                    self._last_error = ""
        self._ctx.ui.refresh(f"input:{event.type}")

    def _clear_pending(self) -> None:
        self._pending_message = ""

    # ------------------------------------------------------------------
    # Service object
    # ------------------------------------------------------------------

    def _create_service(self) -> Any:
        mgr = self._sources
        cat = self._catalog

        class InputService:
            """Service facade for other plugins."""

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

        # Left: recordings and live are separate, first-class tabs.
        ui.add_tab("left", "courses", "课程", icon="📚", order=0, plugin_id=pid)
        ui.add_tab("left", "live", "直播", icon="🔴", order=5, plugin_id=pid)

        ui.add_toolbox_item(
            "courses",
            "input.courses",
            "课程录播",
            description="选择课程，查看历史录播",
            order=0,
            render=self._render_courses_list,
            plugin_id=pid,
        )
        ui.add_toolbox_item(
            "live",
            "input.live",
            "今日直播",
            description="正在或即将开始的课堂直播",
            order=0,
            render=self._render_live,
            plugin_id=pid,
        )
        ui.add_toolbox_item(
            "live",
            "input.manual",
            "手动播放 URL",
            description="直接播放 HLS/FLV/mp4/本地文件",
            order=1,
            render=self._render_manual_form,
            plugin_id=pid,
        )

        # Right: a persistent playback control tab.
        ui.add_tab("right", "session", "播放控制", icon="▶", order=10, plugin_id=pid)
        ui.add_toolbox_item(
            "session",
            "input.session",
            "当前播放",
            description="暂停、继续或停止当前音视频源",
            order=0,
            render=self._render_session_item,
            plugin_id=pid,
        )

        ui.add_page(
            "input:session",
            "播放控制",
            render=lambda: self._render_session(),
            plugin_id=pid,
        )
        ui.add_page(
            "input:live",
            "直播",
            render=lambda: _build_live_page(ctx, self._catalog),
            plugin_id=pid,
        )
        ui.add_page(
            "input:courses",
            "课程录播",
            render=self._render_courses_list,
            plugin_id=pid,
        )
        ui.add_page(
            "input:manual",
            "手动播放",
            render=self._render_manual_form,
            plugin_id=pid,
        )

        # ---- Actions ----
        for action_id, handler in (
            ("input.refresh_courses", self._act_refresh_courses),
            ("input.refresh_live", self._act_refresh_live),
            ("input.open_course", self._act_open_course),
            ("input.play_recording", self._act_play_recording),
            ("input.watch_live", self._act_watch_live),
            ("input.play_url", self._act_play_url),
            ("input.stop", self._act_stop),
            ("input.pause", self._act_pause),
            ("input.resume", self._act_resume),
        ):
            ui.register_action(action_id, handler, plugin_id=pid)

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
                        "或将 `CANVAS_TOKEN` 写入根 `.env`。"
                    ),
                },
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "去设置 Canvas",
                            "action": "ui.open_page",
                            "value": {"page_id": "settings:main"},
                            "class": "btn btn-sm btn-primary",
                        }
                    ],
                },
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
                    '<button class="btn btn-sm" '
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
                            "label": "📂 查看录播",
                            "action": "input.open_course",
                            "value": {"course": cid, "name": name},
                            "class": "btn btn-sm",
                        }
                    ],
                }
            )
        blocks.append({"type": "cards", "items": cards})
        return blocks

    def _render_live(self) -> list[dict[str, Any]]:
        if not self._catalog.has_token():
            return [
                {
                    "type": "markdown",
                    "text": "⚠️ **Canvas Token 未配置**\n\n请先在设置中填写 `input.canvas_token`。",
                },
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "去设置 Canvas",
                            "action": "ui.open_page",
                            "value": {"page_id": "settings:main"},
                            "class": "btn btn-sm btn-primary",
                        }
                    ],
                },
            ]
        now = datetime.now()
        start_str = now.strftime("%Y-%m-%d 00:00:00")
        end_str = now.strftime("%Y-%m-%d 23:59:59")
        try:
            lives = self._catalog.list_lives(start_str, end_str)
        except InputError as exc:
            return [
                {"type": "text", "text": f"⚠️ 直播列表获取失败：{exc}"},
                {"type": "buttons", "buttons": [
                    {"label": "🔄 重试", "action": "input.refresh_live", "class": "btn btn-sm"}
                ]},
            ]
        active = [
            live for live in lives
            if _live_state(str(live.get("begin", "")), str(live.get("end", ""))) != "已结束"
        ]
        blocks: list[dict[str, Any]] = [
            {"type": "markdown", "text": "### 🔴 今日直播"},
            {"type": "text", "text": f"正在/即将开始：{len(active)} 场"},
            {
                "type": "buttons",
                "buttons": [
                    {"label": "打开直播页", "action": "ui.open_page", "value": {"page_id": "input:live"}, "class": "btn btn-sm btn-primary"},
                    {"label": "🔄 刷新", "action": "input.refresh_live", "class": "btn btn-sm"},
                ],
            },
        ]
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

    def _render_session_item(self) -> list[dict[str, Any]]:
        status = self._sources.status()
        state = status.get("state", "idle")
        source = status.get("source") or {}
        streams = self._ctx.streams.current
        state_label = {
            "idle": "空闲",
            "playing": "播放中",
            "paused": "已暂停",
            "stopped": "已停止",
        }.get(str(state), str(state))
        if self._pending_message:
            state_label = self._pending_message
        items = [
            ["状态", state_label],
            ["标题", source.get("title", "—")],
            ["音频/视频", f"{streams.audio.duration:.1f}s / {streams.video.duration:.1f}s"],
        ]
        if self._last_error:
            items.append(["错误", self._last_error])
        return [
            {
                "type": "keyvalue",
                "items": items,
            },
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "打开播放控制",
                        "action": "ui.open_page",
                        "value": {"page_id": "input:session"},
                        "class": "btn btn-sm btn-primary",
                    }
                ],
            },
        ]

    def _render_session(self) -> list[dict[str, Any]]:
        status = self._sources.status()
        state = status.get("state", "idle")
        source = status.get("source") or {}
        streams = self._ctx.streams.current
        state_label = {
            "idle": "空闲",
            "playing": "播放中",
            "paused": "已暂停",
            "stopped": "已停止",
        }.get(str(state), str(state))
        if self._pending_message:
            state_label = self._pending_message
        items = [
            ["状态", state_label],
            ["标题", source.get("title", "—")],
            ["类型", source.get("kind", "—")],
            ["音频", f"{streams.audio.duration:.1f}s"],
            ["视频", f"{streams.video.duration:.1f}s"],
        ]
        if self._last_error:
            items.append(["错误", self._last_error])
        blocks: list[dict[str, Any]] = [{"type": "keyvalue", "items": items}]
        if state in ("playing", "paused"):
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {"label": "暂停", "action": "input.pause", "class": "btn btn-sm"},
                        {"label": "继续", "action": "input.resume", "class": "btn btn-sm btn-primary"},
                        {"label": "停止", "action": "input.stop", "class": "btn btn-sm btn-danger"},
                    ],
                }
            )
        else:
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {"label": "去直播", "action": "ui.open_page", "value": {"page_id": "input:live"}, "class": "btn btn-sm"},
                        {"label": "去课程录播", "action": "ui.open_page", "value": {"page_id": "input:courses"}, "class": "btn btn-sm"},
                    ],
                }
            )
        return blocks

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _act_refresh_courses(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._catalog.reset()
        return {"ok": True}

    def _act_refresh_live(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._catalog.reset()
        self._ctx.ui.refresh("input.live.refresh")
        return {"ok": True, "message": "直播列表已刷新"}

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
        self._pending_message = "正在获取录播地址…"
        self._last_error = ""

        def _bg() -> None:
            try:
                url = self._catalog.get_recording_url(course_id, video_id)
            except InputError as exc:
                self._pending_message = ""
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
            self._pending_message = ""
            try:
                self._sources.start(spec)
            except Exception as exc:  # noqa: BLE001 - surface startup failures
                self._pending_message = ""
                self._ctx.events.emit(
                    "input.error",
                    message=f"启动播放失败：{exc}",
                    source=spec.title,
                )

        threading.Thread(target=_bg, daemon=True).start()
        self._ctx.ui.refresh("input:recording:pending")
        return {"ok": True, "message": "正在获取播放地址…", "open_page": "input:session"}

    def _act_watch_live(self, payload: dict[str, Any]) -> dict[str, Any]:
        live_id = str(payload.get("live", ""))
        if not live_id:
            return {"ok": False, "message": "缺少直播 ID"}
        self._pending_message = "正在获取直播地址…"
        self._last_error = ""

        device_pref = self._ctx.require("settings").get("input.live_device", "0")

        def _bg() -> None:
            try:
                devices = self._catalog.get_live_devices(live_id)
            except InputError as exc:
                self._pending_message = ""
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
            self._pending_message = ""
            try:
                self._sources.start(spec)
            except Exception as exc:  # noqa: BLE001 - surface startup failures
                self._pending_message = ""
                self._ctx.events.emit(
                    "input.error",
                    message=f"启动播放失败：{exc}",
                    source=spec.title,
                )

        threading.Thread(target=_bg, daemon=True).start()
        self._ctx.ui.refresh("input:live:pending")
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
        try:
            self._sources.start(spec)
        except Exception as exc:  # noqa: BLE001 - surface startup failures
            self._ctx.events.emit(
                "input.error",
                message=f"启动播放失败：{exc}",
                source=title,
            )
            return {"ok": False, "message": f"启动播放失败：{exc}"}
        return {"ok": True, "message": "开始播放", "open_page": "input:session"}

    def _act_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._pending_message = ""
        self._last_error = ""
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