"""Input plugin: Canvas/SJTU video portal → ffmpeg → audio/video streams.

Registers settings, the ``"input"`` service, the left-side course tree, the
per-class playback toolbox item and the related actions.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from typing import Any

from core.plugin import Plugin, PluginContext
from integrations.sjtu_canvas import InputError
from plugins.settings.spec import SettingsSpec
from .catalog import Catalog
from .ffmpeg import probe_duration
from .sources import SourceManager, SourceSpec
from .tree import build_tree, live_state, local_time

_log = logging.getLogger("lecture_agent.plugins.input")



def _local_time(iso_string: str) -> str:
    """Convert an ISO-ish timestamp to local time display."""

    return local_time(iso_string)


def _live_state(begin: str, end: str) -> str:
    """Return 直播中 / 未开始 / 已结束 for a local-time range."""

    return live_state(begin, end)


def _fmt_size(value: int | float) -> str:
    size = float(value or 0)
    if size >= 1 << 30:
        return f"{size / (1 << 30):.1f}GB"
    if size >= 1 << 20:
        return f"{size / (1 << 20):.1f}MB"
    if size >= 1 << 10:
        return f"{size / (1 << 10):.0f}KB"
    return f"{int(size)}B"


def _fmt_clock(seconds: float) -> str:
    total = max(0, int(seconds or 0))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _map_view_streams(views: dict[str, str]) -> dict[str, str]:
    """Map recording views to ``main`` (classroom) and ``screen`` URLs."""

    main = views.get("1") or next(iter(views.values()), "")
    screen = views.get("2", "")
    out: dict[str, str] = {}
    if main:
        out["main"] = main
    if screen and screen != main:
        out["screen"] = screen
    return out


def _map_live_streams(devices: list[dict[str, Any]]) -> dict[str, str]:
    """Map live devices to ``main`` (classroom) and ``screen`` URLs."""

    by_view: dict[int, str] = {}
    ordered: list[str] = []
    for device in devices:
        url = str(device.get("url", "") or "")
        if not url:
            continue
        ordered.append(url)
        try:
            view = int(device.get("view_num") or 0)
        except (TypeError, ValueError):
            continue
        by_view.setdefault(view, url)

    main = by_view.get(1) or (ordered[0] if ordered else "")
    screen = by_view.get(2) or (ordered[1] if len(ordered) > 1 else "")
    out: dict[str, str] = {}
    if main:
        out["main"] = main
    if screen and screen != main:
        out["screen"] = screen
    return out


class InputPlugin(Plugin):
    id = "input"
    requires = ("settings",)
    provides = ("input",)

    def setup(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._pending_message = ""
        self._last_error = ""
        settings = ctx.require("settings")

        # Course tree cache / sidebar state.
        self._tree_lock = threading.RLock()
        self._tree: dict[str, Any] = {
            "nodes": [],
            "error": "",
            "at": 0.0,
            "started": 0.0,
            "loading": False,
        }
        self._tree_ttl = 30.0
        self._expanded: set[str] = set()
        self._current_class_id = ""

        # Current class / player state.
        self._class_lock = threading.RLock()
        self._current_class: dict[str, Any] | None = None
        self._class_index = 0
        self._class_streams: dict[str, str] = {}
        self._streams_loading = False
        self._streams_error = ""
        self._processing = False
        self._want_processing = False
        self._session_started = False
        self._class_finished = False
        self._download: dict[str, Any] = {}
        self._total_seconds = 0.0
        self._processing_started = 0.0
        self._class_sessions: dict[str, dict[str, Any]] = {}
        self._class_session_meta: dict[str, Any] = {}
        self._class_session_id: str = ""

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
            reconnect=bool(settings.get("input.reconnect", True)),
            media_root=ctx.data_dir / "sessions",
        )

        # ---- Register input service ----
        ctx.register_service("input", self._create_service())

        def _input_meta() -> dict[str, Any]:
            session = ctx.streams.current
            return {
                "course": session.meta.get("course_name", "")
                or (
                    session.meta.get("course", "")
                    if isinstance(session.meta.get("course", ""), str)
                    else ""
                ),
                "course_id": session.meta.get("course_id", ""),
                "lecture": session.meta.get("class_label", "")
                or session.meta.get("lecture", ""),
                "session_id": session.session_id,
                "source": session.meta.get("title", ""),
                "kind": session.meta.get("kind", ""),
                "audio_seconds": session.audio.duration,
                "video_seconds": session.video.duration,
            }

        ctx.meta.register("input", _input_meta, plugin_id=self.id)

        # ---- Register UI ----
        self._setup_ui(ctx)
        ctx.ui.register_task_provider(self.running_tasks)
        ctx.events.on("session.started", self._on_session_event, plugin_id=self.id)
        ctx.events.on("session.stopped", self._on_session_event, plugin_id=self.id)
        self._load_session_index()
        ctx.events.on("input.status", self._on_input_event, plugin_id=self.id)
        ctx.events.on("input.download", self._on_input_event, plugin_id=self.id)
        ctx.events.on("input.error", self._on_input_event, plugin_id=self.id)
        ctx.events.on("input.ended", self._on_input_event, plugin_id=self.id)
        ctx.events.on("settings.changed", self._on_settings_changed, plugin_id=self.id)

        ctx.logger.info("Input plugin ready")

    def teardown(self) -> None:
        self._sources.stop()
        self._catalog.close()

    def _on_input_event(self, event: Any) -> None:
        if event.type == "input.download":
            status = str(event.get("status", ""))
            with self._class_lock:
                self._download = {
                    "status": status,
                    "received": int(event.get("received", 0) or 0),
                    "total": int(event.get("total", 0) or 0),
                    "error": str(event.get("error", "") or ""),
                }
            self._ctx.ui.refresh("input.download")
            return
        if event.type == "input.error":
            self._pending_message = ""
            self._last_error = str(event.get("message", "") or "输入源错误")
        elif event.type == "input.ended":
            _log.info("input.ended received; draining consumers")
            threading.Thread(target=self._on_class_ended, daemon=True).start()
        elif event.type == "input.status":
            status = str(event.get("status", ""))
            if status in {"playing", "stopped"}:
                self._pending_message = ""
                if status == "playing":
                    self._last_error = ""
        self._ctx.ui.refresh(f"input:{event.type}")

    def running_tasks(self) -> list[dict[str, Any]]:
        with self._class_lock:
            if not self._processing:
                return []
            cls = self._current_class or {}
            label = str(cls.get("name") or cls.get("label") or "")
            started = self._processing_started
        session = self._ctx.streams.current
        ingested = float(session.video.duration or session.audio.duration)
        return [
            {
                "id": "input",
                "label": f"处理课程 · {label or '—'}",
                "detail": f"已摄录 {_fmt_clock(ingested)}",
                "started": started,
            }
        ]

    # ------------------------------------------------------------------
    # Session index (survives restarts)
    # ------------------------------------------------------------------
    @staticmethod
    def _class_key(course: Any, video: Any) -> str:
        course_text = str(course or "").strip()
        video_text = str(video or "").strip()
        if course_text and video_text:
            return f"{course_text}:{video_text}"
        return ""

    def _session_meta_path(self, session_id: str) -> Path:
        return self._ctx.data_dir / "sessions" / session_id / "meta.json"

    def _load_session_index(self) -> None:
        """Rebuild class → last session from ``meta.json`` files on disk."""

        root = self._ctx.data_dir / "sessions"
        entries: list[dict[str, Any]] = []
        try:
            paths = list(root.glob("*/meta.json"))
        except OSError:
            return
        for path in paths:
            try:
                data = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            meta = data.get("meta") or {}
            if not isinstance(meta, dict):
                continue
            sid = str(data.get("session_id") or path.parent.name)
            key = self._class_key(meta.get("course_id"), meta.get("video_id"))
            entry = {
                "session_id": sid,
                "ended_at": data.get("ended_at") or 0,
                "meta": meta,
                "at": float(data.get("started_at") or 0),
            }
            entries.append(entry)
            if key:
                current = self._class_sessions.get(key)
                if current is None or entry["at"] >= current["at"]:
                    self._class_sessions[key] = entry
        if entries:
            latest = max(entries, key=lambda item: item["at"])
            self._ctx.streams.set_notes_session(
                latest["session_id"], latest.get("meta") or {}
            )

    def _on_session_event(self, event: Any) -> None:
        sid = str(event.get("session_id") or "")
        if not sid:
            return
        path = self._session_meta_path(sid)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if event.type == "session.started":
                meta = dict(event.get("meta") or {})
                payload = {
                    "session_id": sid,
                    "started_at": float(event.get("started_at") or 0),
                    "ended_at": 0,
                    "meta": meta,
                }
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), "utf-8"
                )
                key = self._class_key(meta.get("course_id"), meta.get("video_id"))
                if key:
                    self._class_sessions[key] = {
                        "session_id": sid,
                        "ended_at": 0,
                        "meta": meta,
                        "at": payload["started_at"],
                    }
            else:
                data: dict[str, Any] = {}
                if path.exists():
                    try:
                        data = json.loads(path.read_text("utf-8"))
                    except (OSError, ValueError):
                        data = {}
                data["ended_at"] = float(event.get("ended_at") or time.time())
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
                )
                for entry in self._class_sessions.values():
                    if entry["session_id"] == sid:
                        entry["ended_at"] = data["ended_at"]
        except OSError:
            return

    def _find_finished_class(self, cls: dict[str, Any]) -> dict[str, Any] | None:
        videos = cls.get("videos") or []
        key = self._class_key(cls.get("course"), videos[0] if videos else "")
        entry = self._class_sessions.get(key)
        if entry and entry.get("ended_at"):
            return entry
        return None

    def _on_settings_changed(self, event: Any) -> None:
        key = str(event.get("key", ""))
        if key in {"input.canvas_token", "input.canvas_base_url"}:
            self._catalog.reset()
            self._invalidate_tree()
            self._ensure_tree(force=True)
        if key.startswith("input."):
            self._apply_source_settings()
            self._ctx.ui.refresh("input.settings")

    def _apply_source_settings(self) -> None:
        settings = self._ctx.require("settings")
        self._sources.configure(
            sample_rate=int(settings.get("input.audio_rate", 16000)),
            video_fps=float(settings.get("input.video_fps", 4.0)),
            video_width=int(settings.get("input.video_width", 0)),
            reconnect=bool(settings.get("input.reconnect", True)),
        )

    def _start_source(self, spec: SourceSpec) -> None:
        """Apply current playback settings, then start one source."""

        self._apply_source_settings()
        self._sources.start(spec)

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

        # Left sidebar: one collapsible course tree, no tab strip.
        ui.add_tab("left", "courses", "课程", icon="📚", order=0, plugin_id=pid)
        ui.add_toolbox_item(
            "courses",
            "input.courses",
            "",
            order=0,
            render=self._render_courses_list,
            plugin_id=pid,
        )

        # Right sidebar: functional areas for the selected class.
        ui.add_tab("right", "session", "本节课", icon="▶", order=10, plugin_id=pid)
        ui.add_toolbox_item(
            "session",
            "input.session",
            "处理控制",
            order=0,
            render=self._render_session_item,
            plugin_id=pid,
            page_id="input:session",
        )

        ui.add_page(
            "input:session",
            "课堂",
            render=lambda: self._render_session(),
            plugin_id=pid,
        )
        ui.add_page(
            "input:manual",
            "手动播放",
            render=self._render_manual_form,
            plugin_id=pid,
        )

        for action_id, handler in (
            ("input.refresh_courses", self._act_refresh_courses),
            ("input.toggle_course", self._act_toggle_course),
            ("input.open_class", self._act_open_class),
            ("input.start_processing", self._act_start_processing),
            ("input.reprocess", self._act_reprocess),
            ("input.pause_processing", self._act_pause_processing),
            ("input.play_url", self._act_play_url),
            ("input.stop", self._act_stop),
            ("input.pause", self._act_pause),
            ("input.resume", self._act_resume),
        ):
            ui.register_action(action_id, handler, plugin_id=pid)

        ui.register_command(
            "input.refresh",
            "刷新课程列表",
            category="课程",
            icon="🔄",
            order=0,
            action="input.refresh_courses",
            plugin_id=pid,
        )
        ui.register_command(
            "input.manual",
            "手动播放 URL",
            subtitle="HLS / FLV / mp4 / 本地文件",
            category="调试",
            icon="🔗",
            order=10,
            page_id="input:manual",
            plugin_id=pid,
        )

    # ------------------------------------------------------------------
    # Toolbox renders
    # ------------------------------------------------------------------

    def _render_courses_list(self) -> list[dict[str, Any]]:
        if not self._catalog.has_token():
            return [
                {"type": "markdown", "text": "**未配置 Canvas Token**"},
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "去设置",
                            "action": "ui.open_page",
                            "value": {"page_id": "settings:main"},
                            "class": "btn btn-sm btn-primary",
                        }
                    ],
                },
            ]

        self._ensure_tree()
        with self._tree_lock:
            nodes = list(self._tree["nodes"])
            error = str(self._tree["error"])
            loading = bool(self._tree["loading"])
            slow = loading and time.monotonic() - float(self._tree["started"]) > 20.0

        blocks: list[dict[str, Any]] = []
        if nodes:
            blocks.append(
                {
                    "type": "tree",
                    "id": "courses",
                    "nodes": nodes,
                    "empty_text": "没有课程",
                }
            )
        elif loading and not slow:
            blocks.append({"type": "text", "text": "加载课程…"})
        elif loading:
            blocks.append({"type": "text", "text": "课程列表加载超时，点「刷新」重试。"})
        elif error:
            blocks.append({"type": "text", "text": f"⚠️ {error}"})
        else:
            blocks.append({"type": "text", "text": "没有找到课程。"})

        blocks.append(
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "刷新",
                        "action": "input.refresh_courses",
                        "class": "btn btn-sm btn-ghost",
                    }
                ],
            }
        )
        return blocks

    # ------------------------------------------------------------------
    # Course tree cache
    # ------------------------------------------------------------------
    def _ensure_tree(self, *, force: bool = False) -> None:
        with self._tree_lock:
            if self._tree["loading"]:
                stale = time.monotonic() - float(self._tree["started"]) > 30.0
                if not (force and stale):
                    return
            fresh = (
                self._tree["nodes"]
                and time.monotonic() - float(self._tree["at"]) < self._tree_ttl
            )
            if fresh and not force:
                return
            self._tree["loading"] = True
            self._tree["started"] = time.monotonic()
        threading.Thread(
            target=self._load_tree, name="course-tree", daemon=True
        ).start()

    def _invalidate_tree(self) -> None:
        with self._tree_lock:
            self._tree["at"] = 0.0

    def _apply_tree_state(self) -> None:
        """Re-apply expand/active flags to the cached tree (no refetch)."""

        with self._tree_lock:
            expanded = set(self._expanded)
            current = self._current_class_id
            nodes = self._tree["nodes"]

        def walk(items: list[dict[str, Any]]) -> None:
            for node in items:
                node_id = str(node.get("id", ""))
                if node_id.startswith("course:"):
                    cid = node_id.split(":", 1)[1]
                    node["expanded"] = (cid in expanded) or bool(node.get("live"))
                if node.get("action"):
                    node["active"] = node_id == current
                walk(node.get("children", []) or [])

        walk(nodes)

    def _load_tree(self) -> None:
        nodes: list[dict[str, Any]] = []
        error = ""
        try:
            courses = self._catalog.list_courses()
            recordings: dict[str, list[dict[str, Any]]] = {}
            for course in courses:
                cid = str(course.get("id", ""))
                try:
                    recordings[cid] = self._catalog.list_recordings(cid)
                except InputError:
                    recordings[cid] = []
            now = datetime.now()
            lives = self._catalog.list_lives(
                now.strftime("%Y-%m-%d 00:00:00"),
                now.strftime("%Y-%m-%d 23:59:59"),
            )
            with self._tree_lock:
                expanded = set(self._expanded)
                current = self._current_class_id
            nodes = build_tree(
                courses,
                recordings,
                lives,
                now=now,
                expanded=expanded,
                current_class_id=current,
            )
        except InputError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - never break the sidebar
            _log.exception("course tree build failed")
            error = f"课程列表加载失败：{exc}"
        with self._tree_lock:
            self._tree.update(
                {
                    "nodes": nodes,
                    "error": error,
                    "at": time.monotonic(),
                    "loading": False,
                }
            )
        self._ctx.ui.refresh("input.tree")

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

    # ------------------------------------------------------------------
    # Player / processing control
    # ------------------------------------------------------------------
    def _render_session_item(self) -> list[dict[str, Any]]:
        with self._class_lock:
            cls = dict(self._current_class) if self._current_class else None
            index = self._class_index
            processing = self._processing
            finished = self._class_finished
            loading = self._streams_loading
            error = self._streams_error

        if cls is None:
            return [{"type": "text", "text": "未选择课程"}]

        parts = len(cls.get("videos") or []) or 1
        if finished:
            state_label = "已结束"
        elif processing:
            state_label = "处理中"
        elif loading:
            state_label = "准备中"
        else:
            state_label = "已暂停"

        items = [
            ["课程", str(cls.get("name", "") or "—")],
            ["节次", str(cls.get("label", "") or "—")],
            ["进度", f"第 {index + 1}/{parts} 节" if parts > 1 else "单节"],
            ["状态", state_label],
        ]
        if error:
            items.append(["错误", error])

        blocks: list[dict[str, Any]] = [{"type": "keyvalue", "items": items}]
        if finished:
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "重新处理",
                            "action": "input.reprocess",
                            "class": "btn btn-sm",
                        }
                    ],
                }
            )
        elif processing:
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "暂停处理",
                            "action": "input.pause_processing",
                            "class": "btn btn-sm",
                        }
                    ],
                }
            )
        else:
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "继续处理",
                            "action": "input.start_processing",
                            "class": "btn btn-sm btn-primary",
                        }
                    ],
                }
            )
        return blocks

    def _render_session(self) -> list[dict[str, Any]]:
        with self._class_lock:
            cls = dict(self._current_class) if self._current_class else None
            index = self._class_index
            processing = self._processing
            finished = self._class_finished
            streams = dict(self._class_streams)
            loading = self._streams_loading
            error = self._streams_error
            download = dict(self._download)

        if cls is None:
            return [
                {
                    "type": "markdown",
                    "text": "### 课堂\n\n从左侧课程树选择一节课。",
                }
            ]

        parts = len(cls.get("videos") or []) or 1
        live = cls.get("kind") == "live"
        title = str(cls.get("label", "") or cls.get("name", "") or "课堂")
        name = str(cls.get("name", "") or "")
        heading = f"### {name} · {title}" if name else f"### {title}"
        if parts > 1:
            heading += f" · 第 {index + 1}/{parts} 节"
        heading += "  `直播`" if live else "  `录播`"
        blocks: list[dict[str, Any]] = [{"type": "markdown", "text": heading}]

        if loading:
            blocks.append({"type": "text", "text": "正在获取视频地址…"})
        if error:
            blocks.append({"type": "text", "text": f"⚠️ {error}"})

        if live:
            blocks.extend(
                self._render_live_player(
                    cls, index, parts, streams, processing, finished, loading, error
                )
            )
        else:
            with self._class_lock:
                session_id = self._class_session_id
                session_meta = dict(self._class_session_meta)
            blocks.extend(
                self._render_recording_page(
                    processing, finished, download, session_id, session_meta
                )
            )
        return blocks

    def _render_live_player(
        self,
        cls: dict[str, Any],
        index: int,
        parts: int,
        streams: dict[str, str],
        processing: bool,
        finished: bool,
        loading: bool,
        error: str,
    ) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        if streams.get("main"):
            sources.append({"key": "main", "label": "课堂", "src": streams["main"]})
        if streams.get("screen"):
            sources.append(
                {
                    "key": "screen",
                    "label": "屏幕",
                    "src": streams["screen"],
                    "muted": True,
                }
            )
        if not sources:
            if loading or error:
                return []
            return [{"type": "text", "text": "没有可用的视频流。"}]
        return [
            {
                "type": "player",
                "id": f"input:session:{cls.get('id', '')}:{index}",
                "sources": sources,
                "mode": "live",
                "live": True,
                "processing": processing,
                "finished": finished,
                "part": index + 1,
                "parts": parts,
                "screen_missing": bool(streams.get("main"))
                and not streams.get("screen"),
                "start_action": "input.start_processing",
                "pause_action": "input.pause_processing",
            }
        ]

    def _render_recording_page(
        self,
        processing: bool,
        finished: bool,
        download: dict[str, Any],
        session_id: str = "",
        session_meta: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recordings are processed at full speed: progress instead of a player."""

        blocks: list[dict[str, Any]] = []
        session = self._ctx.streams.current
        meta = dict(session_meta or {})
        if not meta:
            meta = dict(session.meta)
        if not session_id:
            session_id = session.session_id
        ingested = float(session.video.duration or session.audio.duration)
        total = float(self._total_seconds or 0.0)

        status = str(download.get("status", ""))
        if status in {"started", "progress"}:
            received = int(download.get("received", 0) or 0)
            size = int(download.get("total", 0) or 0)
            pct = f"{received * 100 / size:.1f}%" if size else f"{received / 1e6:.0f}MB"
            blocks.append(
                {
                    "type": "progress",
                    "label": f"下载录像 {_fmt_size(received)} / {_fmt_size(size) if size else '…'}",
                    "value": (received / size) if size else 0.0,
                }
            )
            blocks.append({"type": "text", "text": f"正在下载：{pct}"})
            return blocks
        if status == "failed":
            blocks.append(
                {
                    "type": "text",
                    "text": f"⚠️ 本地保存失败，改为直接读取远端流：{download.get('error', '')}",
                }
            )

        if processing or (finished and ingested > 0):
            consumed = self._consumed_seconds()
            denom = total if total > 0 else max(ingested, consumed, 1.0)
            value = min(1.0, max(0.0, consumed / denom)) if denom else 0.0
            processed_label = (
                f"{_fmt_clock(consumed)} / {_fmt_clock(total) if total else '…'}"
            )
            label = "处理完成" if finished else "正在处理"
            if finished:
                value = 1.0
            blocks.append(
                {"type": "progress", "label": f"{label} {processed_label}", "value": value}
            )
            if not finished:
                detail = (
                    f"已摄录 {_fmt_clock(ingested)}"
                    f"，语音 {_fmt_clock(self._consumer_cursor('stt'))}"
                    f"，板书 {_fmt_clock(self._consumer_cursor('vlm'))}"
                )
                blocks.append({"type": "text", "text": detail})

        media_name = str(meta.get("media_name", "") or "")
        if media_name and (processing or finished):
            blocks.append(
                {
                    "type": "markdown",
                    "text": (
                        f"[打开录像](/api/artifacts/{session_id}/media/{media_name})"
                        "（本地文件，可在新窗口直接播放或下载）"
                    ),
                }
            )

        if finished:
            blocks.append({"type": "text", "text": "处理完成，讲义已生成。"})
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "重新处理",
                            "action": "input.reprocess",
                            "class": "btn btn-sm",
                        }
                    ],
                }
            )
        elif processing:
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "暂停处理",
                            "action": "input.pause_processing",
                            "class": "btn btn-sm",
                        }
                    ],
                }
            )
        else:
            blocks.append(
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "开始处理",
                            "action": "input.start_processing",
                            "class": "btn btn-sm btn-primary",
                        }
                    ],
                }
            )
        return blocks

    def _consumer_cursor(self, service_id: str) -> float:
        service = self._ctx.service(service_id)
        if service is None:
            return 0.0
        try:
            return float(service.status().get("cursor", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 - status is best effort
            return 0.0

    def _consumed_seconds(self) -> float:
        cursors = [
            self._consumer_cursor(name) for name in ("stt", "vlm")
        ]
        return max(cursors) if cursors else 0.0

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _act_refresh_courses(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._catalog.reset()
        self._invalidate_tree()
        self._ensure_tree(force=True)
        return {"ok": True, "message": "正在刷新课程"}

    def _act_toggle_course(self, payload: dict[str, Any]) -> dict[str, Any]:
        course_id = str(payload.get("course", ""))
        if course_id:
            with self._tree_lock:
                if course_id in self._expanded:
                    self._expanded.discard(course_id)
                else:
                    self._expanded.add(course_id)
            self._apply_tree_state()
        return {"ok": True, "refresh": True}

    def _act_open_class(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Select a class: load its streams but never auto-start playback."""

        class_id = str(payload.get("id", ""))
        kind = str(payload.get("kind", "recording") or "recording")
        videos = payload.get("videos") or []
        if isinstance(videos, str):
            videos = [part for part in videos.split(",") if part]
        videos = [str(video) for video in videos if str(video)]
        live_id = str(payload.get("live_id", ""))

        if kind == "recording" and not videos:
            return {"ok": False, "message": "缺少录播 ID"}
        if kind == "live" and not live_id:
            return {"ok": False, "message": "缺少直播 ID"}

        with self._class_lock:
            if (
                self._current_class is not None
                and self._current_class.get("id") == class_id
                and not self._class_finished
            ):
                # Re-selecting the same class must not interrupt playback.
                return {"ok": True, "open_page": "input:session"}

        # Switching classes ends the previous class session, but must not
        # start anything for the newly selected one.
        self._sources.stop()
        self._end_open_session("class switched")

        record = self._find_finished_class(
            {"course": payload.get("course", ""), "videos": videos}
        )
        with self._class_lock:
            self._current_class = {
                "id": class_id,
                "kind": kind,
                "course": str(payload.get("course", "")),
                "name": str(payload.get("name", "")),
                "label": str(payload.get("label", "")),
                "videos": videos,
                "live_id": live_id,
            }
            self._class_index = (len(videos) - 1) if record else 0
            self._class_streams = {}
            self._streams_loading = record is None
            self._streams_error = ""
            self._processing = False
            self._want_processing = False
            self._session_started = False
            self._class_finished = record is not None
            self._download = {}
            self._total_seconds = 0.0
            self._class_session_meta = dict(record["meta"]) if record else {}
            self._class_session_id = record["session_id"] if record else ""
        if record:
            # Let the notes pages show this class's handout.
            self._ctx.streams.set_notes_session(
                record["session_id"], record.get("meta") or {}
            )

        with self._tree_lock:
            self._current_class_id = class_id
        self._apply_tree_state()

        threading.Thread(
            target=self._resolve_class_streams, daemon=True
        ).start()
        self._ctx.ui.refresh("input.class.opened")
        return {"ok": True, "open_page": "input:session"}

    def _end_open_session(self, reason: str) -> None:
        hub = self._ctx.streams
        current = hub.current
        if current.session_id != "boot" and current.ended_at is None:
            hub.end_session(reason)

    def _resolve_class_streams(self) -> None:
        with self._class_lock:
            cls = dict(self._current_class) if self._current_class else None
            index = self._class_index
        if cls is None:
            return

        streams: dict[str, str] = {}
        error = ""
        try:
            if cls.get("kind") == "live":
                devices = self._catalog.get_live_devices(
                    str(cls.get("live_id", ""))
                )
                streams = _map_live_streams(devices)
            else:
                videos = cls.get("videos") or []
                if index >= len(videos):
                    raise InputError("没有更多小节")
                views = self._catalog.get_recording_streams(
                    str(cls.get("course", "")), str(videos[index])
                )
                streams = _map_view_streams(views)
            if not streams.get("main"):
                raise InputError("没有找到可播放的视频流。")
        except InputError as exc:
            error = str(exc)

        with self._class_lock:
            if (
                self._current_class is None
                or self._current_class.get("id") != cls.get("id")
            ):
                return
            self._class_streams = streams
            self._streams_error = error
            self._streams_loading = False
            want = self._want_processing

        self._ctx.ui.refresh("input.class.streams")
        if want and streams.get("main"):
            self._start_processing_now()

    def _act_reprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._class_lock:
            self._class_finished = False
            self._class_index = 0
            self._class_session_meta = {}
            self._class_session_id = ""
        return self._act_start_processing(payload)

    def _act_start_processing(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._class_lock:
            if self._current_class is None:
                return {"ok": False, "message": "未选择课程"}
            if self._processing:
                return {"ok": True, "message": "正在处理"}
            streams = dict(self._class_streams)
            loading = self._streams_loading
            error = self._streams_error
            self._want_processing = True

        if streams.get("main"):
            self._start_processing_now()
            return {"ok": True, "message": "开始处理"}

        if loading:
            self._ctx.ui.refresh("input.class.waiting")
            return {"ok": True, "message": "正在准备视频…"}
        if error:
            with self._class_lock:
                self._want_processing = False
            return {"ok": False, "message": error}

        with self._class_lock:
            self._streams_loading = True
            self._streams_error = ""
        threading.Thread(
            target=self._resolve_class_streams, daemon=True
        ).start()
        return {"ok": True, "message": "正在准备视频…"}

    def _start_processing_now(self) -> None:
        with self._class_lock:
            cls = dict(self._current_class) if self._current_class else None
            index = self._class_index
            streams = dict(self._class_streams)
            started = self._session_started
            self._want_processing = False
            self._download = {}
        if cls is None:
            return
        url = streams.get("main", "")
        if not url:
            return

        live = cls.get("kind") == "live"
        label = str(cls.get("label", ""))
        videos = cls.get("videos") or []
        spec = SourceSpec(
            kind="live" if live else "recording",
            url=url,
            title=label or str(cls.get("name", "")),
            continue_session=started,
            # Recordings are always processed at full speed; live is 1:1.
            realtime=live,
            download=not live and not url.startswith(("file://", "/")),
            audio_gate_limit=(
                self._gate_limit("stt", "stt.chunk_seconds", 50.0) if not live else 0.0
            ),
            video_gate_limit=(
                self._gate_limit("vlm", "vlm.window_seconds", 20.0)
                if not live
                else 0.0
            ),
            course={
                "course_id": str(cls.get("course", "")),
                "course_name": str(cls.get("name", "")),
                "class_label": label,
                "video_id": str(videos[index]) if index < len(videos) else "",
                "live_id": str(cls.get("live_id", "")),
            },
        )
        self._pending_message = ""
        with self._class_lock:
            if (
                self._current_class is not None
                and self._current_class.get("id") == cls.get("id")
            ):
                self._processing = True
                self._processing_started = time.time()
                self._session_started = True
                self._class_finished = False
        self._ctx.ui.refresh("input.class.processing")

        def _run() -> None:
            try:
                self._start_source(spec)
                self._total_seconds = self._probe_total_seconds(spec, live)
            except Exception as exc:  # noqa: BLE001 - surface startup failures
                self._ctx.events.emit(
                    "input.error",
                    message=f"启动处理失败：{exc}",
                    source=spec.title,
                )
                with self._class_lock:
                    self._processing = False
            self._ctx.ui.refresh("input.class.started")

        # Downloading + starting ffmpeg can take minutes; never block the
        # HTTP action that triggered it.
        threading.Thread(target=_run, name="input-start", daemon=True).start()

    def _gate_limit(self, service_id: str, window_key: str, default_window: float) -> float:
        """Backlog limit so one session cannot buffer the whole lecture."""

        service = self._ctx.service(service_id)
        try:
            enabled = bool(service.enabled()) if service is not None else False
        except Exception:  # noqa: BLE001 - status is best effort
            enabled = False
        if not enabled:
            return 0.0
        settings = self._ctx.require("settings")
        try:
            concurrency = int(settings.get(f"{service_id}.concurrency", 4))
        except Exception:  # noqa: BLE001
            concurrency = 4
        try:
            window = float(settings.get(window_key, default_window))
        except Exception:  # noqa: BLE001
            window = default_window
        window = max(5.0, window)
        return window * (max(1, concurrency) + 4)

    def _probe_total_seconds(self, spec: SourceSpec, live: bool) -> float:
        if live:
            return 0.0
        session = self._ctx.streams.current
        media_path = str(session.meta.get("media_path", "") or "")
        if not media_path:
            return 0.0
        try:
            return float(probe_duration(media_path) or 0.0)
        except Exception:  # noqa: BLE001 - progress is best effort
            return 0.0

    def _act_pause_processing(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._class_lock:
            if not self._processing:
                self._processing = False
                self._want_processing = False
                return {"ok": True, "message": "已暂停"}
            kind = (self._current_class or {}).get("kind")
            self._processing = False
            self._want_processing = False

        if kind == "live":
            # A live stream cannot be paused: stop our ingestion and resume at
            # the live edge later.
            self._sources.stop()
        else:
            self._sources.pause()
        self._ctx.ui.refresh("input.class.paused")
        return {"ok": True, "message": "已暂停处理"}

    def _on_class_ended(self) -> None:
        """Natural end of the current recording period."""

        with self._class_lock:
            cls = dict(self._current_class) if self._current_class else None
            index = self._class_index
            processing = self._processing
        if not processing:
            return
        if cls is None:
            # Manual URL playback: nothing to advance to, just finish cleanly.
            self._wait_for_consumers()
            with self._class_lock:
                self._processing = False
                self._want_processing = False
                self._class_finished = True
            self._sources.stop()
            self._end_open_session("playback finished")
            self._prune_media()
            self._ctx.ui.refresh("input.finished")
            return

        if cls.get("kind") != "live":
            # full-speed ingest: wait until STT/VLM committed every window
            # before closing the session (summarize runs on session.stopped).
            self._wait_for_consumers()

        videos = cls.get("videos") or []
        if index + 1 < len(videos):
            with self._class_lock:
                self._class_index = index + 1
                self._class_streams = {}
                self._streams_loading = True
                self._streams_error = ""
                self._want_processing = True
            self._ctx.ui.refresh("input.class.next")
            threading.Thread(
                target=self._resolve_class_streams, daemon=True
            ).start()
            return
        self._finish_class()

    def _wait_for_consumers(self, timeout: float = 3600.0) -> bool:
        """Block until the recording's STT/VLM queues are drained."""

        services = []
        for name in ("stt", "vlm"):
            service = self._ctx.service(name)
            if service is not None and hasattr(service, "busy"):
                services.append(service)
        if not services:
            return True
        deadline = time.monotonic() + max(1.0, float(timeout))
        _log.info("draining STT/VLM before closing the session")
        while time.monotonic() < deadline:
            with self._class_lock:
                if not self._processing:
                    return True
            busy = [type(s).__name__ for s in services if s.busy()]
            if not busy:
                _log.info("consumers drained")
                return True
            time.sleep(0.5)
        self._ctx.events.emit(
            "input.error",
            message="板书/语音处理超时，将按已完成的内容生成讲义。",
            source="",
        )
        return False

    def _finish_class(self) -> None:
        with self._class_lock:
            self._processing = False
            self._want_processing = False
            self._class_finished = True
            self._session_started = False
        self._sources.stop()
        self._end_open_session("playback finished")
        self._prune_media()
        self._ctx.ui.refresh("input.class.finished")

    def _prune_media(self) -> None:
        """Keep only the most recent N recording folders."""

        try:
            keep = int(self._ctx.require("settings").get("input.keep_recordings", 4))
        except Exception:  # noqa: BLE001
            keep = 4
        keep = max(0, keep)
        root = self._ctx.data_dir / "sessions"
        try:
            folders = [
                path
                for path in root.glob("*/media")
                if path.is_dir()
            ]
        except OSError:
            return
        folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        current = self._ctx.streams.current.session_id
        for old in folders[keep:]:
            if old.parent.name == current:
                continue
            try:
                shutil.rmtree(old, ignore_errors=True)
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass

    def _act_play_url(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = str(payload.get("url", "")).strip()
        if not url:
            return {"ok": False, "message": "请输入 URL"}
        title = str(payload.get("title", "")).strip() or url[:60]
        kind_raw = str(payload.get("kind", "url"))
        # Map "url" -> auto-detect; we just pass it through
        kind: Any = kind_raw if kind_raw in ("live", "recording") else "url"

        live = kind == "live"
        spec = SourceSpec(
            kind=kind,
            url=url,
            title=title,
            realtime=live,
            download=not live and url.startswith(("http://", "https://")),
            audio_gate_limit=(
                self._gate_limit("stt", "stt.chunk_seconds", 50.0) if not live else 0.0
            ),
            video_gate_limit=(
                self._gate_limit("vlm", "vlm.window_seconds", 20.0)
                if not live
                else 0.0
            ),
        )
        with self._class_lock:
            self._processing = True
            self._class_finished = False

        def _run() -> None:
            try:
                self._start_source(spec)
            except Exception as exc:  # noqa: BLE001 - surface startup failures
                self._ctx.events.emit(
                    "input.error",
                    message=f"启动播放失败：{exc}",
                    source=title,
                )
                self._ctx.ui.refresh("input.error")

        # Downloading + starting ffmpeg must never block the action request.
        threading.Thread(target=_run, name="input-start-url", daemon=True).start()
        return {"ok": True, "message": "开始播放", "open_page": "input:session"}

    def _act_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._pending_message = ""
        self._last_error = ""
        with self._class_lock:
            self._processing = False
            self._want_processing = False
            self._session_started = False
            self._class_finished = self._current_class is not None
        self._sources.stop()
        self._end_open_session("stopped")
        self._ctx.ui.refresh("input.stopped")
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
        description="SJTU Canvas 访问令牌（Bearer Token）；留空表示未配置，修改后保存即可生效。",
        order=0,
        secret=True,
        placeholder="粘贴 Canvas 访问令牌…",
        env=("CANVAS_TOKEN",),
        group_note="Token 可来自根目录 `.env` 的 `CANVAS_TOKEN`；首次启动会把环境值写入设置（已配置时显示掩码），之后以设置页为准。",
    ),
    SettingsSpec(
        key="input.canvas_base_url",
        label="Canvas 地址",
        type="string",
        default="https://oc.sjtu.edu.cn",
        group="Canvas",
        description="Canvas 实例地址",
        order=1,
        env=("CANVAS_BASE_URL",),
    ),
    SettingsSpec(
        key="input.video_fps",
        label="视频帧率",
        type="float",
        default=4.0,
        group="播放",
        description="每秒提取的视频帧数（原管线为 4fps）；下次播放生效。",
        order=10,
    ),
    SettingsSpec(
        key="input.video_width",
        label="视频宽度",
        type="int",
        default=0,
        group="播放",
        description="视频输出宽度；0 表示保持原分辨率，不做缩放；下次播放生效。",
        order=11,
    ),
    SettingsSpec(
        key="input.audio_rate",
        label="音频采样率",
        type="int",
        default=16000,
        group="播放",
        description="音频采样率（Hz）；下次播放生效。",
        order=12,
    ),
    SettingsSpec(
        key="input.keep_recordings",
        label="本地保留录像",
        type="int",
        default=4,
        group="录播",
        description=(
            "录播先下载到会话目录再全速处理，便于回看；"
            "只保留最近 N 节，超出按时间从旧到新删除（0 = 全部删除）。"
        ),
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
        description="断线时自动重连；下次播放生效。",
        order=21,
    ),
]
