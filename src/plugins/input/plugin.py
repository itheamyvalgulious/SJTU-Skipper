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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from html import escape
from pathlib import Path
from typing import Any

from core.coverage import CoverageStore
from core.lecture import source_id
from core.media import MediaKind, SourceKind, ProcessingMode
from core.navigation import current_navigation
from core.streams import StreamKind
from core.ui import BlockType, FormInputType, Side, StatusLabel, TaskState
from core.plugin import Plugin, PluginContext
from core.settings_spec import SettingType, SettingsSpec
from integrations.sjtu_canvas import InputError
from .catalog import Catalog
from .sources import DownloadState, SourceManager, SourceSpec, SourceState
from .tree import (
    LiveState,
    TreeBadge,
    build_tree,
    live_state,
    local_time,
    next_boundary,
    parse_time,
)
from .ffmpeg import probe_duration

_log = logging.getLogger("lecture_agent.plugins.input")

# UI contributions registered by ``_setup_ui`` (removed again on teardown).
_PAGE_IDS = ("input:session", "input:manual")
_TAB_IDS = ("courses", "session")
_ITEM_IDS = ("input.courses", "input.session", "input.queue")
_ACTION_IDS = (
    "input.refresh_courses",
    "input.open_class",
    "input.start_processing",
    "input.delete",
    "input.pause_processing",
    "input.queue_pause",
    "input.queue_resume",
    "input.queue_cancel",
    "input.queue_retry",
    "input.queue_clear",
    "input.play_url",
    "input.stop",
    "input.pause",
    "input.resume",
)
_COMMAND_IDS = ("input.refresh", "input.manual")

# The course tree needs one video-portal round trip per course; fetching them
# serially took ~45s for 14 courses and tripped the sidebar's timeout label.
_TREE_FETCH_WORKERS = 8


# Job states.  "preparing" resolves the media URL, "running" is ingesting,
# "paused" keeps the active slot (the queue waits), the rest are terminal.
class JobState(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


_JOB_TERMINAL = {JobState.DONE, JobState.FAILED}
_JOB_LABELS = {
    JobState.QUEUED: StatusLabel.QUEUED,
    JobState.PREPARING: StatusLabel.PREPARING,
    JobState.RUNNING: StatusLabel.RUNNING,
    JobState.PAUSED: StatusLabel.PAUSED,
    JobState.DONE: StatusLabel.FINISHED,
    JobState.FAILED: StatusLabel.FAILED,
}


class StreamView(StrEnum):
    MAIN = "main"
    SCREEN = "screen"


class TaskIndicator(StrEnum):
    NONE = ""
    BUSY = "busy"
    WARNING = "warn"
    OK = "ok"
    ERROR = "err"
    IDLE = "idle"


@dataclass
class _Course:
    """Course media data, independent of browser navigation and engine state."""

    cls: dict[str, Any]
    streams: dict[StreamView, str] = field(default_factory=dict)
    streams_loading: bool = False
    error: str = ""
    recording_streams: dict[str, dict[StreamView, str]] = field(default_factory=dict)
    session_id: str = ""
    session_meta: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    download: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Job:
    """One class waiting for, or holding, the processing engine."""

    id: str
    cls: dict[str, Any]
    input_part_index: int = 0
    state: JobState = JobState.QUEUED
    streams: dict[StreamView, str] = field(default_factory=dict)
    streams_loading: bool = False
    error: str = ""
    session_id: str = ""
    session_meta: dict[str, Any] = field(default_factory=dict)
    total_seconds: float = 0.0
    part_durations: dict[str, float] = field(default_factory=dict)
    recording_streams: dict[str, dict[StreamView, str]] = field(default_factory=dict)
    downloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    durations_probing: bool = False
    started: float = 0.0
    finished_at: float = 0.0
    message: str = ""
    enqueued_at: float = field(default_factory=time.time)

    @property
    def course(self) -> str:
        return str(self.cls.get("name", "") or "")

    @property
    def label(self) -> str:
        return str(self.cls.get("label", "") or "")

    @property
    def kind(self) -> SourceKind:
        return SourceKind(self.cls.get("kind") or SourceKind.RECORDING)

    @property
    def parts(self) -> int:
        return max(1, len(self.cls.get("videos") or []))

    def title(self) -> str:
        name = self.course or "课程"
        label = self.label
        return f"{name} {label}" if label else name


def _local_time(iso_string: str) -> str:
    """Convert an ISO-ish timestamp to local time display."""

    return local_time(iso_string)


def _live_state(begin: str, end: str) -> LiveState:
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


def _epoch_seconds(value: Any) -> float:
    """Campus-local timestamp → epoch seconds (0.0 when unparseable)."""

    dt = parse_time(value)
    return dt.timestamp() if dt is not None else 0.0


def _shift_intervals(
    intervals: list[tuple[float, float]], shift: float
) -> list[list[float]]:
    """Move live-session ranges onto the recording's media timeline."""

    out: list[list[float]] = []
    for start, end in intervals:
        moved_start = max(0.0, float(start) + float(shift))
        moved_end = float(end) + float(shift)
        if moved_end - moved_start > 0.5:
            out.append([round(moved_start, 3), round(moved_end, 3)])
    return out


def _map_view_streams(views: dict[str, str]) -> dict[StreamView, str]:
    """Map recording views to ``main`` (classroom) and ``screen`` URLs."""

    main = views.get("1") or next(iter(views.values()), "")
    screen = views.get("2", "")
    out: dict[StreamView, str] = {}
    if main:
        out[StreamView.MAIN] = main
    if screen and screen != main:
        out[StreamView.SCREEN] = screen
    return out


def _map_live_streams(devices: list[dict[str, Any]]) -> dict[StreamView, str]:
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
    out: dict[StreamView, str] = {}
    if main:
        out[StreamView.MAIN] = main
    if screen and screen != main:
        out[StreamView.SCREEN] = screen
    return out


class InputPlugin(Plugin):
    id = "input"
    requires = ("settings",)
    optional_requires = ("stt", "vlm", "notes")
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
            "next_boundary": 0.0,
        }
        self._tree_ttl = 30.0

        # Course media records and the engine queue do not store browser selection.
        self._class_lock = threading.RLock()
        self._courses: dict[str, _Course] = {}
        # Processing queue: one engine slot, FIFO, live classes jump ahead.
        self._jobs: dict[str, _Job] = {}
        self._queue: list[str] = []
        self._active_job: _Job | None = None
        # Manual playback and gap filling use the engine without a queue job.
        self._processing = False
        self._playback_class_id = ""
        self._processing_started = 0.0
        # Gap-fill pass state (see _fill_gaps): remaining recording parts.
        self._fill_queue: list[dict[str, Any]] = []
        self._fill_target = ""
        self._fill_live_started = 0.0
        self._fill_media_begin = 0.0
        self._fill_course: dict[str, Any] = {}

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
            audio_enabled=self._audio_enabled,
        )

        # ---- Register input service ----
        ctx.register_service("input", self._create_service())
        self._gap_fill_notes = ctx.service("notes")
        if self._gap_fill_notes is not None:
            self._gap_fill_notes.set_gap_filler(self._fill_gaps)

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
        ctx.ui.set_context_provider(self._ui_context, plugin_id=self.id)
        ctx.ui.register_task_provider(self.running_tasks)
        ctx.events.on("session.started", self._on_session_event, plugin_id=self.id)
        ctx.events.on("session.stopped", self._on_session_event, plugin_id=self.id)
        ctx.events.on("clock.tick", self._on_clock_tick, plugin_id=self.id)
        ctx.events.on("input.status", self._on_input_event, plugin_id=self.id)
        ctx.events.on("input.download", self._on_input_event, plugin_id=self.id)
        ctx.events.on("input.error", self._on_input_event, plugin_id=self.id)
        ctx.events.on("input.prepared", self._on_input_event, plugin_id=self.id)
        ctx.events.on("input.ended", self._on_input_event, plugin_id=self.id)
        ctx.events.on("settings.changed", self._on_settings_changed, plugin_id=self.id)

        ctx.logger.info("Input plugin ready")

    def teardown(self) -> None:
        with self._class_lock:
            self._queue.clear()
            self._active_job = None
            self._processing = False
        try:
            self._sources.stop()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            _log.exception("failed to stop input sources")
        try:
            self._catalog.close()
        except Exception:  # noqa: BLE001
            _log.exception("failed to close the input catalog")
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            return
        if self._gap_fill_notes is not None:
            self._gap_fill_notes.set_gap_filler(None)
        ui = ctx.ui
        # Drop every contribution this plugin made: pages, toolbox items,
        # tabs, actions and commands (idempotent; the manager does the same on
        # unload, this keeps a bare teardown() self-sufficient).
        ui.set_context_provider(None, plugin_id=self.id)
        for page_id in _PAGE_IDS:
            try:
                ui.remove_page(page_id)
            except Exception:  # noqa: BLE001 - UI may already be gone
                pass
        for item_id in _ITEM_IDS:
            try:
                ui.remove_toolbox_item(item_id)
            except Exception:  # noqa: BLE001
                pass
        for tab_id in _TAB_IDS:
            try:
                ui.remove_tab(tab_id)
            except Exception:  # noqa: BLE001
                pass
        for action_id in _ACTION_IDS:
            try:
                ui.unregister_action(action_id)
            except Exception:  # noqa: BLE001
                pass
        for command_id in _COMMAND_IDS:
            try:
                ui.unregister_command(command_id)
            except Exception:  # noqa: BLE001
                pass

    def _on_input_event(self, event: Any) -> None:
        session_id = str(event.get("session_id", "") or "")
        if session_id and session_id != self._ctx.streams.current.session_id:
            return
        if session_id:
            with self._class_lock:
                job = self._active_job
                if job is not None:
                    job.session_id = session_id
                    job.session_meta = dict(self._ctx.streams.current.meta)
                else:
                    class_id = self._fill_target or self._playback_class_id
                    course = self._courses.get(class_id)
                    if course is not None:
                        course.session_id = session_id
                        course.session_meta = dict(self._ctx.streams.current.meta)
        if event.type == "input.prepared":
            self._record_part_duration(
                str(event.get("video_id", "") or ""),
                float(event.get("duration", 0.0) or 0.0),
            )
            self._ctx.ui.refresh("input.prepared")
            return
        if event.type == "input.download":
            status = DownloadState(event.get("status"))
            with self._class_lock:
                video_id = str(event.get("video_id", "") or "")
                job = self._active_job
                if job is not None and video_id not in {
                    str(value) for value in job.cls.get("videos", [])
                }:
                    return
                progress = {
                    "status": status,
                    "video_id": video_id,
                    "received": int(event.get("received", 0) or 0),
                    "total": int(event.get("total", 0) or 0),
                    "error": str(event.get("error", "") or ""),
                }
                if job is not None:
                    job.downloads[video_id] = progress
                    session = self._ctx.streams.get_session(job.session_id)
                    if session is not None:
                        session.meta["download_progress"] = self._download_progress(job)
                else:
                    course = self._courses.get(
                        self._fill_target or self._playback_class_id
                    )
                    if course is not None:
                        course.download = progress
            self._ctx.ui.refresh("input.download")
            return
        if event.type == "input.error":
            self._pending_message = ""
            self._last_error = str(event.get("message", "") or "输入源错误")
            with self._class_lock:
                job = self._active_job
                if job is None:
                    course = self._courses.get(
                        self._fill_target or self._playback_class_id
                    )
                    if course is not None:
                        course.error = self._last_error
            if job is not None:
                threading.Thread(
                    target=self._fail_job, args=(job, self._last_error), daemon=True
                ).start()
        elif event.type == "input.ended":
            self._record_part_duration(
                str(event.get("video_id", "") or ""),
                float(event.get("duration", 0.0) or 0.0),
            )
            _log.info("input.ended received")
            threading.Thread(target=self._on_class_ended, daemon=True).start()
        elif event.type == "input.status":
            status = SourceState(event.get("status"))
            if status in {SourceState.PLAYING, SourceState.STOPPED}:
                self._pending_message = ""
                if status == SourceState.PLAYING:
                    self._last_error = ""
        self._ctx.ui.refresh(f"input:{event.type}")

    def running_tasks(self) -> list[dict[str, Any]]:
        """Status-bar entries: the active job plus a compact queue summary."""

        with self._class_lock:
            active = self._active_job
            active_state = active.state if active is not None else ""
            active_title = active.title() if active is not None else ""
            active_started = active.started if active is not None else 0.0
            legacy_busy = self._processing and self._active_job is None
            legacy_label = ""
            if legacy_busy:
                legacy_label = str(
                    self._fill_course.get("course_name")
                    or self._ctx.streams.current.meta.get("title")
                    or "手动播放"
                )
            queued = [
                self._jobs[job_id]
                for job_id in self._queue
                if job_id in self._jobs and self._jobs[job_id].state == JobState.QUEUED
            ]

        session = self._ctx.streams.current
        read_progress = (
            f"音频已读取 {_fmt_clock(session.audio.duration)}"
            f"，视频已读取 {_fmt_clock(session.video.duration)}"
        )
        tasks: list[dict[str, Any]] = []
        if legacy_busy:
            tasks.append(
                {
                    "id": "input",
                    "label": f"处理：{legacy_label}",
                    "detail": read_progress,
                    "started": self._processing_started,
                }
            )
        if active is not None and active_state in {
            JobState.PREPARING,
            JobState.RUNNING,
            JobState.PAUSED,
        }:
            if active_state == JobState.PREPARING:
                detail = StatusLabel.PREPARING
            elif active_state == JobState.PAUSED:
                detail = StatusLabel.PAUSED
            else:
                detail = read_progress
            tasks.append(
                {
                    "id": "input",
                    "label": f"处理：{active_title}",
                    "detail": detail,
                    "started": active_started,
                }
            )
        for position, job in enumerate(queued[:3], start=1):
            detail = StatusLabel.QUEUED
            tasks.append(
                {
                    "id": f"queue:{job.id}",
                    "state": TaskState.PENDING,
                    "label": f"排队 {position}：{job.title()}",
                    "detail": detail,
                    "started": job.enqueued_at,
                }
            )
        if len(queued) > 3:
            tasks.append(
                {
                    "id": "queue:more",
                    "state": TaskState.PENDING,
                    "label": "处理队列",
                    "detail": f"另有 {len(queued) - 3} 节等待",
                    "started": 0.0,
                }
            )
        return tasks

    # ------------------------------------------------------------------
    # Panel context (which course / lesson the right sidebar is about)
    # ------------------------------------------------------------------
    def _ui_context(self) -> dict[str, Any] | None:
        if getattr(self, "_ctx", None) is None:
            return None
        with self._class_lock:
            cls = dict(self._current_class) if self._current_class else None
            job = self._job_for_locked(self._current_class_id) if cls else None
            job_state = job.state if job is not None else None
            if cls and self._current_class_id in {
                self._fill_target,
                self._playback_class_id,
            }:
                job_state = JobState.RUNNING if self._processing else JobState.PAUSED

        if cls is None:
            return None

        title = str(cls.get("name") or "课程")
        kind = "直播" if cls.get("kind") == SourceKind.LIVE else "录播"
        bits = [str(cls.get("label") or ""), kind]
        subtitle = " ".join(part for part in bits if part)
        dot = {
            JobState.RUNNING: TaskIndicator.BUSY,
            JobState.PREPARING: TaskIndicator.BUSY,
            JobState.PAUSED: TaskIndicator.WARNING,
            JobState.DONE: TaskIndicator.OK,
            JobState.FAILED: TaskIndicator.ERROR,
        }.get(job_state, TaskIndicator.NONE)
        return {
            "title": title,
            "subtitle": subtitle,
            "selected": True,
            "state": dot,
        }

    # ------------------------------------------------------------------
    # Stored class records
    # ------------------------------------------------------------------

    @property
    def _current_class(self) -> dict[str, Any] | None:
        return current_navigation().class_meta or None

    @property
    def _current_class_id(self) -> str:
        return current_navigation().class_id

    def _session_meta_path(self, session_id: str) -> Path:
        return self._ctx.data_dir / "sessions" / session_id / "meta.json"

    def _load_session_meta(self, session_id: str) -> dict[str, Any]:
        """Read one session's ``meta.json`` (empty dict when unavailable).

        Exposed through the ``input`` service so upper layers never have to
        parse the on-disk file themselves.
        """

        if not session_id:
            return {}
        try:
            data = json.loads(self._session_meta_path(session_id).read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        meta = data.get("meta")
        return dict(meta) if isinstance(meta, dict) else {}

    def _on_session_event(self, event: Any) -> None:
        sid = str(event.get("session_id") or "")
        if not sid or sid == "boot":
            return
        path = self._session_meta_path(sid)
        meta = dict(event.get("meta") or {})
        class_id = str(meta.get("class_id") or sid)
        with self._class_lock:
            course = self._courses.get(class_id)
            if course is not None:
                course.session_id = sid
                course.session_meta = meta
                course.finished = event.type == "session.stopped" and not meta.get(
                    "processing_cancelled"
                )
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
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), "utf-8"
                )
                temporary.replace(path)
            else:
                data: dict[str, Any] = {}
                if path.exists():
                    try:
                        data = json.loads(path.read_text("utf-8"))
                    except (OSError, ValueError):
                        data = {}
                if (
                    data.get("started_at")
                    and event.get("started_at")
                    and data["started_at"] != event.get("started_at")
                ):
                    return
                data["ended_at"] = (
                    0
                    if meta.get("processing_cancelled")
                    else float(event.get("ended_at") or time.time())
                )
                data["meta"] = dict(event.get("meta") or data.get("meta") or {})
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
                )
                temporary.replace(path)
        except OSError:
            return

    def _find_finished_class(self, class_id: str) -> dict[str, Any] | None:
        try:
            data = json.loads(self._session_meta_path(class_id).read_text("utf-8"))
        except (OSError, ValueError):
            return None
        meta = data.get("meta") or {}
        if (
            data.get("ended_at")
            and not meta.get("processing_cancelled")
            and not meta.get("input_failed")
        ):
            return data
        return None

    def _resolve_class(self, class_id: str) -> dict[str, Any]:
        if (
            not class_id
            or Path(class_id).name != class_id
            or class_id in {".", "..", "boot"}
        ):
            raise KeyError("Class not found")
        with self._class_lock:
            job = self._jobs.get(class_id)
            course = self._courses.get(class_id)
            cls = dict(job.cls) if job else dict(course.cls) if course else None
        if cls is None:
            meta = self._load_session_meta(class_id)
            if meta:
                cls = {
                    **meta,
                    "id": class_id,
                    "course": str(meta.get("course_id") or ""),
                    "name": str(meta.get("course_name") or ""),
                    "label": str(meta.get("class_label") or ""),
                    "videos": [
                        str(part["video_id"])
                        for part in meta.get("media_parts") or []
                        if part.get("video_id")
                    ]
                    or ([str(meta["video_id"])] if meta.get("video_id") else []),
                    "live_id": str(meta.get("live_id") or ""),
                }
            else:

                def find(nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
                    for node in nodes:
                        if node.get("id") == class_id:
                            return dict(node.get("action", {}).get("value") or {})
                        found = find(node.get("children", []))
                        if found:
                            return found
                    return None

                with self._tree_lock:
                    cls = find(self._tree["nodes"])
                if cls is None:
                    self._load_tree()
                    with self._tree_lock:
                        cls = find(self._tree["nodes"])
        if not cls:
            raise KeyError("Class not found")
        self._remember_class(cls)
        return dict(cls)

    def _remember_class(self, cls: dict[str, Any]) -> None:
        class_id = str(cls["id"])
        record = self._find_finished_class(class_id)
        with self._class_lock:
            course = self._courses.get(class_id)
            if course is None:
                course = _Course(cls=dict(cls))
                self._courses[class_id] = course
            else:
                course.cls = dict(cls)
            if record:
                course.finished = True
                course.session_id = str(record.get("session_id") or class_id)
                course.session_meta = dict(record.get("meta") or {})
            job = self._jobs.get(class_id)
            resolve = (
                not course.finished
                and not course.streams_loading
                and not course.streams
                and not course.error
                and (job is None or job.state in _JOB_TERMINAL)
            )
            if resolve:
                course.streams_loading = True
        if resolve:
            threading.Thread(
                target=self._resolve_class_streams,
                args=(class_id,),
                name="class-streams",
                daemon=True,
            ).start()

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
    # Gap filling
    # ------------------------------------------------------------------
    def _fill_gaps(self, live_session_id: str) -> bool:
        """Process the class recording(s), skipping what the live pass covered.

        The notes plugin calls this after a live class ends. Recording parts
        share one session and one timeline. Return False if no recording exists.
        """

        ctx = self._ctx
        if ctx is None:
            return False
        try:
            infos = self._ctx.streams.list_sessions()
        except Exception:  # noqa: BLE001 - best effort
            return False
        info = next((s for s in infos if s.session_id == live_session_id), None)
        if info is None:
            return False
        meta = dict(info.meta)
        if str(meta.get("kind") or "") != SourceKind.LIVE and not meta.get("fill_for"):
            return False
        with self._class_lock:
            busy = self._processing or self._active_job is not None
        if busy or self._sources.status().get("state") not in {SourceState.IDLE, None}:
            _log.info("gap fill skipped: a source is already running")
            return False

        course_id = str(meta.get("course_id") or "")
        if not course_id:
            return False
        started = float(info.started_at or 0.0)
        ended = float(info.ended_at or 0.0) or started
        try:
            recordings = self._catalog.list_recordings(course_id, limit=50)
        except InputError as exc:
            _log.info("gap fill: no recordings for %s (%s)", course_id, exc)
            return False
        parts = self._pick_recordings(recordings, started, ended)
        if not parts:
            _log.info("gap fill: no recording overlaps the live window")
            return False

        self._fill_target = live_session_id
        self._fill_media_begin = _epoch_seconds(parts[0].get("begin"))
        origin = float(meta.get("media_begin") or started)
        shift = origin - self._fill_media_begin if self._fill_media_begin else 0.0
        if ctx.streams.consumers_busy():
            return False
        directory = ctx.data_dir / "sessions" / live_session_id
        directory.mkdir(parents=True, exist_ok=True)
        if shift:
            for name in ("transcript.jsonl", "visual.jsonl"):
                path = directory / name
                if not path.exists():
                    continue
                records = [
                    json.loads(line)
                    for line in path.read_text("utf-8").splitlines()
                    if line.strip()
                ]
                for record in records:
                    for key in ("start", "end"):
                        if record.get(key) is not None:
                            record[key] = float(record[key]) + shift
                path.write_text(
                    "".join(
                        json.dumps(record, ensure_ascii=False) + "\n"
                        for record in records
                    ),
                    "utf-8",
                )
            coverage = CoverageStore(ctx.data_dir, live_session_id)
            (directory / "coverage.json").write_text(
                json.dumps(
                    {
                        kind.value: _shift_intervals(
                            coverage.coverage(kind).intervals, shift
                        )
                        for kind in MediaKind
                    }
                ),
                "utf-8",
            )
        self._fill_live_started = self._fill_media_begin
        self._fill_queue = parts[1:]
        self._fill_course = dict(meta)
        with self._class_lock:
            self._playback_class_id = ""
            course = self._courses.get(live_session_id)
            if course is not None:
                course.finished = False
                course.download = {}
            self._processing = True
            self._processing_started = time.time()
        self._jobs_changed("input.class.processing")
        if self._start_fill_part(parts[0], continue_session=False):
            return True
        # First part failed: try the rest before giving up.
        while self._fill_queue:
            if self._start_fill_part(self._fill_queue.pop(0), continue_session=False):
                return True
        with self._class_lock:
            self._processing = False
            self._fill_target = ""
            self._fill_queue = []
        return False

    def _start_fill_part(
        self, recording: dict[str, Any], *, continue_session: bool
    ) -> bool:
        """Start one recording part of a gap-fill pass."""

        meta = dict(self._fill_course)
        course_id = str(meta.get("course_id") or "")
        video_id = str(recording.get("video_id") or "")
        try:
            views = self._catalog.get_recording_streams(course_id, video_id)
        except InputError as exc:
            _log.info("gap fill: cannot resolve recording %s (%s)", video_id, exc)
            return False
        url = _map_view_streams(views).get(StreamView.MAIN, "")
        if not url:
            return False

        begin = _epoch_seconds(recording.get("begin"))
        session = self._ctx.streams.current
        offset = float(session.audio.duration) if continue_session else 0.0
        shift = offset + (self._fill_live_started - begin if begin else 0.0)
        store = CoverageStore(self._ctx.data_dir, self._fill_target)
        skip = {
            "audio": _shift_intervals(store.coverage(MediaKind.AUDIO).intervals, shift),
            "video": _shift_intervals(store.coverage(MediaKind.VIDEO).intervals, shift),
        }
        _log.info(
            "gap fill part %s for %s (offset %.1fs, skip audio=%s video=%s)",
            video_id,
            self._fill_target,
            offset,
            skip["audio"],
            skip["video"],
        )
        spec = SourceSpec(
            kind=SourceKind.RECORDING,
            url=url,
            title=str(meta.get("title") or meta.get("class_label") or ""),
            continue_session=continue_session,
            realtime=False,
            download=not url.startswith(("file://", "/")),
            audio_buffer_limit=self._buffer_limit(MediaKind.AUDIO),
            video_buffer_limit=self._buffer_limit(MediaKind.VIDEO),
            final_part=not self._fill_queue,
            meta={
                "fill_for": self._fill_target,
                # The first part's wall-clock start maps the whole fill session
                # (parts are back-to-back) onto the lecture timeline.
                "media_begin": self._fill_media_begin,
                "skip_ranges": skip,
            },
            course={
                "class_id": self._fill_target,
                "course_id": course_id,
                "course_name": str(meta.get("course_name") or ""),
                "class_label": str(meta.get("class_label") or ""),
                "video_id": video_id,
                "live_id": "",
            },
        )
        try:
            self._start_source(spec)
        except Exception:  # noqa: BLE001 - try the next part
            _log.exception("gap fill source failed to start")
            return False
        return True

    @staticmethod
    def _pick_recordings(
        recordings: list[dict[str, Any]], started: float, ended: float
    ) -> list[dict[str, Any]]:
        """Every recording that overlaps the live class, in time order."""

        picked: list[dict[str, Any]] = []
        for item in recordings:
            begin = _epoch_seconds(item.get("begin"))
            end = _epoch_seconds(item.get("end")) or begin
            if not begin:
                continue
            if min(end, ended + 600.0) - max(begin, started - 600.0) > 0:
                picked.append(item)
        picked.sort(key=lambda item: _epoch_seconds(item.get("begin")))
        return picked

    # ------------------------------------------------------------------
    # Service object
    # ------------------------------------------------------------------

    def _create_service(self) -> Any:
        mgr = self._sources
        cat = self._catalog
        plugin = self

        class InputService:
            """Input sources, course metadata, and stored sessions."""

            @property
            def status(self) -> dict[str, Any]:
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

            def session_meta(self, session_id: str) -> dict[str, Any]:
                """Session metadata from ``sessions/<id>/meta.json`` (no parsing
                of internal files by callers)."""

                return plugin._load_session_meta(session_id)

            def resolve_class(self, class_id: str) -> dict[str, Any]:
                return plugin._resolve_class(class_id)

            def open_class(self, class_id: str) -> None:
                result = plugin._act_open_class(plugin._resolve_class(class_id))
                if not result.get("ok"):
                    raise ValueError(result.get("message", "Cannot open class"))

        return InputService()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self, ctx: PluginContext) -> None:
        ui = ctx.ui
        pid = self.id

        # Left sidebar: one collapsible course tree, no tab strip.
        ui.add_tab(Side.LEFT, "courses", "课程", order=0, plugin_id=pid)
        ui.add_toolbox_item(
            "courses",
            "input.courses",
            "",
            order=0,
            render=self._render_courses_list,
            plugin_id=pid,
        )

        # Right sidebar: functional areas for the selected class.
        ui.add_tab(Side.RIGHT, "session", "课次", order=10, plugin_id=pid)
        ui.add_toolbox_item(
            "session",
            "input.session",
            "",
            order=0,
            render=self._render_session_item,
            plugin_id=pid,
        )
        ui.add_toolbox_item(
            "session",
            "input.queue",
            "",
            order=1,
            render=self._render_queue_item,
            plugin_id=pid,
        )

        ui.add_page(
            "input:session",
            "课次",
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
            ("input.open_class", self._act_open_class),
            ("input.start_processing", self._act_start_processing),
            ("input.delete", self._act_delete),
            ("input.pause_processing", self._act_pause_processing),
            ("input.queue_pause", self._act_queue_pause),
            ("input.queue_resume", self._act_queue_resume),
            ("input.queue_cancel", self._act_queue_cancel),
            ("input.queue_retry", self._act_queue_retry),
            ("input.queue_clear", self._act_queue_clear),
            ("input.play_url", self._act_play_url),
            ("input.stop", self._act_stop),
            ("input.pause", self._act_pause),
            ("input.resume", self._act_resume),
        ):
            ui.register_action(action_id, handler, plugin_id=pid)

        ui.register_command(
            "input.refresh",
            "刷新课程列表",
            icon="arrow-clockwise",
            description="重新加载课程列表",
            category="课程",
            order=0,
            action="input.refresh_courses",
            plugin_id=pid,
        )
        ui.register_command(
            "input.manual",
            "手动播放 URL",
            icon="play-circle",
            description="打开手动播放页面，输入视频或直播地址",
            category="调试",
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
                {"type": BlockType.MARKDOWN, "text": "**未配置 Canvas Token**"},
                {
                    "type": BlockType.BUTTONS,
                    "buttons": [
                        {
                            "label": "去设置",
                            "action": "ui.open_page",
                            "value": {"page_id": "settings:main"},
                            "class": "btn-outline theme-primary btn-sm",
                        }
                    ],
                },
            ]

        self._ensure_tree()
        with self._tree_lock:
            nodes = self._apply_tree_state(self._tree["nodes"])
            error = str(self._tree["error"])
            loading = bool(self._tree["loading"])
            slow = loading and time.monotonic() - float(self._tree["started"]) > 20.0

        blocks: list[dict[str, Any]] = []
        if nodes:
            blocks.append(
                {
                    "type": BlockType.TREE,
                    "id": "courses",
                    "nodes": nodes,
                    "empty_text": "没有课程",
                }
            )
        elif loading and not slow:
            blocks.append({"type": BlockType.TEXT, "text": "加载课程…"})
        elif loading:
            blocks.append(
                {"type": BlockType.TEXT, "text": "课程列表加载超时，点「刷新」重试。"}
            )
        elif error:
            blocks.append({"type": BlockType.TEXT, "text": error})
        else:
            blocks.append({"type": BlockType.TEXT, "text": "没有找到课程。"})

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

    def _on_clock_tick(self, event: Any) -> None:
        """Reload the tree when the timetable says a class state changed.

        ``_load_tree`` records the nearest upcoming begin/end across today's
        live streams.  When that moment passes, the 直播中/未开始/已结束 labels
        are stale even though nothing else happened, so refresh without waiting
        for a manual 刷新 click.
        """

        with self._tree_lock:
            boundary = float(self._tree.get("next_boundary", 0.0) or 0.0)
        if not boundary:
            return
        try:
            now = float(event.get("ts", time.time()))
        except (TypeError, ValueError):
            now = time.time()
        if now < boundary:
            return
        with self._tree_lock:
            if float(self._tree.get("next_boundary", 0.0) or 0.0) != boundary:
                return  # another path already handled this boundary
            self._tree["next_boundary"] = 0.0
        _log.info(
            "timetable boundary %s reached; refreshing course tree",
            datetime.fromtimestamp(boundary).strftime("%H:%M:%S"),
        )
        self._invalidate_tree()
        self._ensure_tree(force=True)
        self._ctx.ui.refresh("input.tree")

    def _apply_tree_state(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Decorate a render copy without changing cached course metadata."""

        with self._class_lock:
            job_states = {
                job_id: job.state
                for job_id, job in self._jobs.items()
                if job.state
                in {
                    JobState.QUEUED,
                    JobState.PREPARING,
                    JobState.RUNNING,
                    JobState.PAUSED,
                }
            }
        current = current_navigation().class_id

        def decorate(node: dict[str, Any]) -> dict[str, Any]:
            node = dict(node)
            node_id = str(node.get("id", ""))
            if node.get("action"):
                node["active"] = node_id == current
                value = node["action"].get("value") or {}
                job_state = job_states.get(str(value.get("id", "")))
                if job_state:
                    node["badge"] = _JOB_LABELS[job_state]
                    node["badge_class"] = (
                        TreeBadge.PAUSED
                        if job_state == JobState.PAUSED
                        else TreeBadge.LIVE
                        if job_state in {JobState.PREPARING, JobState.RUNNING}
                        else TreeBadge.UPCOMING
                    )
            if "children" in node:
                node["children"] = [decorate(child) for child in node["children"]]
            return node

        return [decorate(node) for node in nodes]

    def _jobs_changed(self, reason: str = "input.queue.changed") -> None:
        """Repaint the queue-related UI, including the course-tree badges."""

        self._ctx.ui.refresh(reason)

    def _course_recordings(self, course_id: str) -> list[dict[str, Any]]:
        """Fetch one course's recordings; a broken course yields an empty list."""

        try:
            return self._catalog.list_recordings(course_id)
        except InputError:
            return []

    def _load_tree(self) -> None:
        nodes: list[dict[str, Any]] = []
        error = ""
        now = datetime.now()
        lives: list[dict[str, Any]] = []
        try:
            courses = self._catalog.list_courses()
            course_ids = [str(course.get("id", "")) for course in courses]
            recordings: dict[str, list[dict[str, Any]]] = {}

            def publish() -> None:
                with self._tree_lock:
                    previous = {
                        node["id"]: node for node in self._tree.get("nodes", [])
                    }
                    self._tree["nodes"] = build_tree(
                        courses,
                        recordings,
                        lives,
                        now=now,
                    )
                    for node in self._tree["nodes"]:
                        cid = node["id"].removeprefix("course:")
                        if cid not in recordings and node["id"] in previous:
                            node["children"] = previous[node["id"]].get("children", [])
                            node["live"] = previous[node["id"]].get("live", False)
                self._ctx.ui.refresh("input.tree")

            publish()
            workers = max(1, min(_TREE_FETCH_WORKERS, len(course_ids) + 1))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="course-tree"
            ) as pool:
                recording_jobs = {
                    pool.submit(self._course_recordings, cid): cid for cid in course_ids
                }
                live_job = None
                for job in as_completed(recording_jobs):
                    recordings[recording_jobs[job]] = job.result()
                    publish()
                    if live_job is None and recordings[recording_jobs[job]]:
                        live_job = pool.submit(
                            self._catalog.list_lives,
                            now.strftime("%Y-%m-%d 00:00:00"),
                            now.strftime("%Y-%m-%d 23:59:59"),
                        )
                if live_job is None:
                    live_job = pool.submit(
                        self._catalog.list_lives,
                        now.strftime("%Y-%m-%d 00:00:00"),
                        now.strftime("%Y-%m-%d 23:59:59"),
                    )
                lives = live_job.result()
            publish()
            with self._tree_lock:
                nodes = self._tree["nodes"]
        except InputError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - never break the sidebar
            _log.exception("course tree build failed")
            error = f"课程列表加载失败：{exc}"
        with self._tree_lock:
            if error:
                nodes = self._tree.get("nodes", nodes)
            self._tree.update(
                {
                    "nodes": nodes,
                    "error": error,
                    "at": time.monotonic(),
                    "loading": False,
                    "next_boundary": next_boundary(lives, now=now),
                }
            )
        self._ctx.ui.refresh("input.tree")

    def _render_manual_form(self) -> list[dict[str, Any]]:
        return [
            {
                "type": BlockType.FORM,
                "fields": [
                    {
                        "key": "url",
                        "label": "视频地址",
                        "type": FormInputType.STRING,
                        "placeholder": "https://... 或 rtsp://...",
                    },
                    {
                        "key": "title",
                        "label": "标题",
                        "type": FormInputType.STRING,
                        "placeholder": "可选",
                    },
                    {
                        "key": "kind",
                        "label": "类型",
                        "type": FormInputType.SELECT,
                        "options": [
                            {"value": SourceKind.URL, "label": "自动检测"},
                            {"value": SourceKind.LIVE, "label": "直播"},
                            {"value": SourceKind.RECORDING, "label": "录播"},
                        ],
                        "default": SourceKind.URL,
                    },
                ],
                "action": "input.play_url",
            },
        ]

    # ------------------------------------------------------------------
    # Player / processing control
    # ------------------------------------------------------------------
    def _selected_job_locked(self) -> _Job | None:
        return self._job_for_locked(current_navigation().class_id)

    def _job_for_locked(self, class_id: str) -> _Job | None:
        return self._jobs.get(str(class_id)) if class_id else None

    def _render_session_item(self) -> list[dict[str, Any]]:
        with self._class_lock:
            cls = dict(self._current_class) if self._current_class else None
            job = self._selected_job_locked()
            job_state = job.state if job is not None else None
            job_error = job.error if job is not None else ""
            course = self._courses.get(self._current_class_id)
            finished = (job is not None and job.state == JobState.DONE) or (
                job is None and course is not None and course.finished
            )
            loading = (job is not None and job.streams_loading) or (
                job is None and course is not None and course.streams_loading
            )
            error = job_error or (course.error if course is not None else "")
            standalone = self._current_class_id in {
                self._fill_target,
                self._playback_class_id,
            }
            if standalone:
                finished = False
                job_state = JobState.RUNNING if self._processing else JobState.PAUSED

        if cls is None:
            return []

        if job_state:
            state_label = _JOB_LABELS[job_state]
        elif finished:
            state_label = StatusLabel.FINISHED
        elif loading:
            state_label = StatusLabel.PREPARING
        else:
            state_label = StatusLabel.NOT_STARTED
        state_theme = {
            JobState.QUEUED: "secondary",
            JobState.PREPARING: "primary",
            JobState.RUNNING: "primary",
            JobState.PAUSED: "warning",
            JobState.DONE: "success",
            JobState.FAILED: "danger",
        }.get(
            job_state, "success" if finished else "primary" if loading else "secondary"
        )

        details = []
        if error:
            details.append(error)

        buttons: list[dict[str, Any]] = []
        if standalone:
            buttons.extend(
                [
                    {
                        "label": "暂停处理" if self._processing else "继续处理",
                        "action": "input.pause_processing"
                        if self._processing
                        else "input.resume",
                        "class": "btn-solid theme-secondary",
                    },
                    {
                        "label": "停止处理",
                        "action": "input.stop",
                        "class": "btn-solid theme-secondary",
                    },
                ]
            )
        elif job is not None and job_state == JobState.RUNNING:
            buttons.append(
                {
                    "label": "暂停处理",
                    "action": "input.pause_processing",
                    "class": "btn-solid theme-secondary",
                }
            )
            buttons.append(
                {
                    "label": "停止处理",
                    "action": "input.queue_cancel",
                    "class": "btn-solid theme-secondary",
                }
            )
        elif job is not None and job_state in {JobState.QUEUED, JobState.PREPARING}:
            buttons.append(
                {
                    "label": "提前处理",
                    "action": "input.start_processing",
                    "class": "btn-solid theme-primary",
                }
            )
            buttons.append(
                {
                    "label": "取消排队",
                    "action": "input.pause_processing",
                    "class": "btn-solid theme-secondary",
                }
            )
        elif job is not None and job_state == JobState.PAUSED:
            buttons.append(
                {
                    "label": "继续处理",
                    "action": "input.start_processing",
                    "class": "btn-solid theme-primary",
                }
            )
            buttons.append(
                {
                    "label": "放弃",
                    "action": "input.stop",
                    "class": "btn-solid theme-secondary",
                }
            )
        elif finished:
            buttons.append(
                {
                    "label": "删除板书",
                    "action": "input.delete",
                    "class": "btn-solid theme-danger",
                }
            )
            if self._ctx.ui.has_action("notes.fill_again"):
                buttons.append(
                    {
                        "label": "补录缺失内容",
                        "action": "notes.fill_again",
                        "class": "btn-solid theme-secondary",
                    }
                )
        elif job is not None and job_state == JobState.FAILED:
            if job.kind != SourceKind.LIVE:
                buttons.append(
                    {
                        "label": "继续处理",
                        "action": "input.start_processing",
                        "class": "btn-solid theme-primary",
                    }
                )
            buttons.append(
                {
                    "label": "删除板书",
                    "action": "input.delete",
                    "class": "btn-solid theme-danger",
                }
            )
        else:
            buttons.append(
                {
                    "label": "开始处理",
                    "action": "input.start_processing",
                    "class": "btn-solid theme-primary",
                }
            )
        controls = "".join(
            f'<button type="button" class="{button["class"]}" '
            f'data-action="{button["action"]}" data-id="{escape(str(cls.get("id", "")), quote=True)}">{escape(button["label"])}</button>'
            for button in buttons
        )
        detail_html = "".join(
            f'<p class="fg-2 mt-2 mb-0">{escape(detail)}</p>' for detail in details
        )
        return [
            {
                "type": BlockType.HTML,
                "html": f"""<section class="sidebar-section" aria-label="课次处理">
          <div class="d-flex align-items-center justify-content-between gap-2">
            <h3 class="fs-md fw-semibold mb-0">课次处理</h3>
            <span class="badge theme-{state_theme}" role="status">{escape(state_label)}</span>
          </div>
          {detail_html}
          <div class="d-flex flex-wrap align-items-center gap-2 mt-3">
            <button type="button" class="btn-outline theme-primary" data-action="ui.open_page" data-value='{{"page_id":"input:session"}}'>打开课次</button>
            {controls}
          </div>
        </section>""",
            }
        ]

    def _render_queue_item(self) -> list[dict[str, Any]]:
        """Right sidebar: the whole processing queue, independent of selection."""

        with self._class_lock:
            active = self._active_job
            ordered: list[_Job] = []
            if active is not None:
                ordered.append(active)
            for job_id in self._queue:
                job = self._jobs.get(job_id)
                if (
                    job is not None
                    and job is not active
                    and job.state == JobState.QUEUED
                ):
                    ordered.append(job)
        if not ordered:
            return []
        rows: list[dict[str, Any]] = []
        for job in ordered:
            progress = ""
            state = _JOB_LABELS[job.state]
            action, label = "", ""
            if job is active:
                if job.state == JobState.RUNNING:
                    consumed = self._consumed_seconds()
                    total = job.total_seconds
                    progress = f"{_fmt_clock(consumed)} / {_fmt_clock(total) if total else '未知时长'}"
                    if total > 0:
                        progress += f" ({min(100, int(consumed * 100 / total))}%)"
                    action, label = "input.queue_pause", "暂停"
                elif job.state == JobState.PAUSED:
                    action, label = "input.queue_resume", "继续"
            kind = "直播" if job.kind == SourceKind.LIVE else "录播"
            actions = []
            if action:
                actions.append(
                    {
                        "action": action,
                        "label": label,
                        "icon": "pause" if job.state == JobState.RUNNING else "play",
                        "value": {"id": job.id},
                    }
                )
            actions.append(
                {
                    "action": "input.queue_cancel",
                    "label": "取消",
                    "icon": "x",
                    "theme": "danger",
                    "value": {"id": job.id},
                }
            )
            rows.append(
                {
                    "id": job.id,
                    "title": job.course or "课程",
                    "detail": f"{job.label} {kind}".strip(),
                    "subtitle": state,
                    "progress": progress,
                    "theme": "warning"
                    if job.state == JobState.PAUSED
                    else "secondary"
                    if job.state == JobState.QUEUED
                    else "info"
                    if job.state == JobState.PREPARING
                    else "primary",
                    "actions": actions,
                }
            )
        running = sum(
            job.state in {JobState.RUNNING, JobState.PREPARING} for job in ordered
        )
        pending = sum(job.state == JobState.QUEUED for job in ordered)
        return [
            {
                "type": BlockType.QUEUE,
                "title": "处理队列",
                "summary": f"{StatusLabel.RUNNING} {running} 个，{StatusLabel.QUEUED} {pending} 个",
                "items": rows,
            }
        ]

    def _render_session(self) -> list[dict[str, Any]]:
        with self._class_lock:
            cls = dict(self._current_class) if self._current_class else None
            job = self._selected_job_locked()
            active = job is not None and self._active_job is job
            course = self._courses.get(self._current_class_id) or _Course(cls or {})
            index = job.input_part_index if job is not None else 0
            streams = dict(job.streams if job is not None else course.streams)
            loading = job.streams_loading if job is not None else course.streams_loading
            error = job.error if job is not None else course.error
            job_state = job.state if job is not None else None
            session_id = job.session_id if job is not None else course.session_id
            session_meta = dict(
                job.session_meta if job is not None else course.session_meta
            )
            standalone = self._current_class_id in {
                self._fill_target,
                self._playback_class_id,
            }
            if standalone:
                active = True
                session = self._ctx.streams.current
                session_id = session.session_id
                session_meta = dict(session.meta)
                job = None
                job_state = None
                error = course.error
                loading = False
            active = active or standalone
            processing = (standalone and self._processing) or (
                active and job_state in {JobState.RUNNING, JobState.PREPARING}
            )
            paused = (standalone and not self._processing) or (
                active and job_state == JobState.PAUSED
            )
            finished = (
                False
                if standalone
                else job.state == JobState.DONE
                if job is not None
                else course.finished
            )
            download = (
                self._download_progress(job)
                if job is not None
                else dict(course.download)
                if standalone
                else dict(session_meta.get("download_progress", {}))
            )
            total = (
                float(job.total_seconds)
                if job is not None
                else float(session_meta.get("total_media_seconds") or 0.0)
            )

        if cls is None:
            return [
                {
                    "type": BlockType.TEXT,
                    "text": "未选择课次",
                }
            ]

        parts = len(cls.get("videos") or []) or 1
        live = cls.get("kind") == SourceKind.LIVE
        title = str(cls.get("label", "") or cls.get("name", "") or "课次")
        name = str(cls.get("name", "") or "")
        heading = f"### {name} {title}" if name else f"### {title}"
        heading += "  直播" if live else "  录播"
        blocks: list[dict[str, Any]] = [{"type": BlockType.MARKDOWN, "text": heading}]

        if job is not None and job_state == JobState.QUEUED and not active:
            with self._class_lock:
                position = self._queue.index(job.id) + 1 if job.id in self._queue else 0
                busy = self._active_job.title() if self._active_job is not None else ""
            text = f"排队中（第 {position} 位）。"
            if busy:
                text += f"当前正在处理：{busy}。"
            blocks.append({"type": BlockType.TEXT, "text": text})
            if not live:
                return blocks
        if loading:
            blocks.append({"type": BlockType.TEXT, "text": "正在获取视频地址…"})
        if error:
            blocks.append({"type": BlockType.TEXT, "text": error})

        if live:
            blocks.extend(
                self._render_live_player(
                    cls,
                    index,
                    parts,
                    streams,
                    processing,
                    finished,
                    loading,
                    error,
                    queued=bool(
                        job is not None
                        and job_state in {JobState.QUEUED, JobState.PREPARING}
                        and not active
                    ),
                )
            )
            blocks.extend(
                self._render_recording_page(
                    processing,
                    finished,
                    download,
                    session_id,
                    session_meta,
                    job=job,
                    active=active,
                    total=total,
                    progress_only=True,
                )
            )
        else:
            blocks.extend(
                self._render_recording_page(
                    processing,
                    finished,
                    download,
                    session_id,
                    session_meta,
                    job=job,
                    active=active,
                    total=total,
                )
            )
        return blocks

    def _render_live_player(
        self,
        cls: dict[str, Any],
        index: int,
        parts: int,
        streams: dict[StreamView, str],
        processing: bool,
        finished: bool,
        loading: bool,
        error: str,
        *,
        queued: bool = False,
    ) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        if streams.get(StreamView.MAIN):
            sources.append(
                {
                    "key": StreamView.MAIN,
                    "label": "课堂画面",
                    "src": streams[StreamView.MAIN],
                }
            )
        if streams.get(StreamView.SCREEN):
            sources.append(
                {
                    "key": StreamView.SCREEN,
                    "label": "屏幕画面",
                    "src": streams[StreamView.SCREEN],
                    "muted": True,
                }
            )
        if not sources:
            if loading or error:
                return []
            return [{"type": BlockType.TEXT, "text": "没有可用的视频流。"}]
        return [
            {
                "type": BlockType.PLAYER,
                "id": f"input:session:{cls.get('id', '')}:{index}",
                "sources": sources,
                "mode": ProcessingMode.LIVE,
                "live": True,
                "processing": processing,
                "finished": finished,
                "queued": queued,
                "screen_missing": bool(streams.get(StreamView.MAIN))
                and not streams.get(StreamView.SCREEN),
            }
        ]

    def _download_progress(self, job: _Job) -> dict[str, Any]:
        videos = [str(value) for value in job.cls.get("videos", [])]
        progress = [job.downloads.get(video, {}) for video in videos]
        if not any(progress):
            return {}
        complete = all(item.get("status") == DownloadState.DONE for item in progress)
        known = all(int(item.get("total", 0)) > 0 for item in progress)
        return {
            "status": DownloadState.DONE if complete else DownloadState.PROGRESS,
            "received": sum(int(item.get("received", 0)) for item in progress),
            "total": sum(int(item.get("total", 0)) for item in progress)
            if known
            else 0,
        }

    def _render_recording_page(
        self,
        processing: bool,
        finished: bool,
        download: dict[str, Any],
        session_id: str = "",
        session_meta: dict[str, Any] | None = None,
        *,
        job: _Job | None = None,
        active: bool = False,
        total: float | None = None,
        progress_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Recordings are processed at full speed: progress instead of a player."""

        blocks: list[dict[str, Any]] = []
        hub = self._ctx.streams
        session = hub.get_session(session_id) if session_id else None
        if session is None and active:
            session = hub.current
            session_id = session.session_id
        meta = dict(session.meta if session is not None else session_meta or {})
        total_seconds = float(total or meta.get("total_media_seconds") or 0.0)
        live = progress_only and total_seconds <= 0
        estimated = False
        if total_seconds <= 0 and job is not None:
            durations = []
            for part in job.cls.get("recording_parts", []):
                begin, end = parse_time(part.get("begin")), parse_time(part.get("end"))
                if begin is not None and end is not None:
                    durations.append(max(0.0, (end - begin).total_seconds()))
            if len(durations) == len(job.cls.get("videos", [])) and durations:
                total_seconds = sum(durations)
                estimated = True
        coverage = CoverageStore(self._ctx.data_dir, session_id) if session_id else None
        cursors = {}
        for kind, stream in (
            (MediaKind.AUDIO, StreamKind.TRANSCRIPT),
            (MediaKind.VIDEO, StreamKind.VISUAL),
        ):
            progress = (
                hub.text_progress(session_id, stream) if session is not None else None
            )
            cursor = progress.before if progress is not None else 0.0
            if active:
                cursor = max(cursor, self._consumer_cursor(kind))
            elif coverage is not None:
                ranges = coverage.coverage(kind).intervals
                cursor = max(cursor, max((end for _, end in ranges), default=0.0))
            cursors[kind] = cursor
        if live and session is not None:
            stt = self._ctx.service("stt")
            status = stt.status() if stt is not None else {}
            partial = status.get("partial")
            if status.get("session_id") == session_id and partial is not None:
                cursors[MediaKind.AUDIO] = max(
                    cursors[MediaKind.AUDIO],
                    min(session.audio.duration, float(partial["end_ms"]) / 1000),
                )
        pipeline = self._ctx.service("notes")
        notes = (
            pipeline.status(session_id) if pipeline is not None and session_id else None
        )
        note_enabled = bool(notes and notes["chunk_enabled"])
        enabled_by_kind = {}
        for kind, service_id in ((MediaKind.AUDIO, "stt"), (MediaKind.VIDEO, "vlm")):
            service = self._ctx.service(service_id)
            enabled_by_kind[kind] = bool(service is not None and service.enabled())
        enabled = [kind for kind in MediaKind if enabled_by_kind[kind]]
        completed = notes["completed_chunks"] if notes else 0
        expected = notes.get("expected_chunks") if notes else None
        if live:
            end_time = (
                (session.input_done_at or session.ended_at or time.time())
                if session is not None
                else time.time()
            )
            total_seconds = (
                max(0.0, end_time - session.started_at) if session is not None else 0.0
            )
        consumed = min((cursors[kind] for kind in enabled), default=0.0)
        denominator = (
            f"{'约 ' if estimated else ''}{_fmt_clock(total_seconds)}"
            if total_seconds or live
            else "待获取时长"
        )
        ratios = [
            min(1.0, cursors[kind] / total_seconds) if total_seconds else 0.0
            for kind in enabled
        ]
        note_ratio = (
            min(1.0, float(notes.get("published_end", 0)) / total_seconds)
            if live and note_enabled and total_seconds
            else min(1.0, completed / expected)
            if expected and note_enabled
            else min(1.0, float(notes.get("published_end", 0)) / total_seconds)
            if total_seconds and note_enabled
            else 0.0
        )
        if note_enabled and not live:
            ratios.append(note_ratio)
        blocks.append(
            {
                "type": BlockType.PROGRESS,
                "label": f"总进度 落后 {_fmt_clock(max(0.0, total_seconds - consumed))}"
                if live
                else f"总进度 {_fmt_clock(consumed)} / {denominator}",
                "value": min(ratios) if ratios else (1.0 if finished else 0.0),
            }
        )
        if not live:
            received, size = (
                int(download.get("received", 0)),
                int(download.get("total", 0)),
            )
            download_done = download.get("status") == DownloadState.DONE or (
                finished and not download
            )
            blocks.append(
                {
                    "type": BlockType.PROGRESS,
                    "label": "下载进度 完成"
                    if download_done
                    else f"下载进度 {_fmt_size(received)} / {_fmt_size(size) if size else '待获取大小'}",
                    "value": 1.0
                    if download_done
                    else min(1.0, received / size)
                    if size
                    else 0.0,
                }
            )
        for kind, name in ((MediaKind.AUDIO, "STT"), (MediaKind.VIDEO, "VLM")):
            blocks.append(
                {
                    "type": BlockType.PROGRESS,
                    "label": (
                        f"{name} 进度 落后 {_fmt_clock(max(0.0, total_seconds - cursors[kind]))}"
                        if live
                        else f"{name} 进度 {_fmt_clock(cursors[kind])} / {denominator}"
                    )
                    if enabled_by_kind[kind]
                    else f"{name} 进度 未启用",
                    "value": min(1.0, cursors[kind] / total_seconds)
                    if total_seconds and enabled_by_kind[kind]
                    else 0.0,
                }
            )
        if note_enabled:
            chunk_label = (
                f"{completed}/{expected}"
                if expected is not None
                else f"已完成 {completed} 段"
            )
            if live:
                chunk_label += f" / 落后 {_fmt_clock(max(0.0, total_seconds - float(notes.get('published_end', 0))))}"
            blocks.append(
                {
                    "type": BlockType.PROGRESS,
                    "label": f"分段笔记进度 {chunk_label}",
                    "value": note_ratio,
                }
            )
        if notes:
            messages = []
            if notes.get("error"):
                messages.append(
                    f"笔记生成失败：{notes['error']}．删除笔记后可重新生成．"
                )
            for error in notes.get("errors", []):
                message = str(error.get("error") or error.get("message") or "")
                if not message or message == notes.get("error"):
                    continue
                location = str(error.get("source") or "来源读取")
                if error.get("chunk") is not None:
                    location = f"分段笔记 第 {error['chunk'] + 1} 段"
                if error.get("start") is not None and error.get("end") is not None:
                    location += f" {error['start']:.1f}-{error['end']:.1f}s"
                messages.append(f"{location}：{message}")
            if messages:
                text = "\n".join(dict.fromkeys(messages))
                blocks.append(
                    {
                        "type": BlockType.HTML,
                        "html": f'<div class="alert theme-danger" role="alert"><pre class="session-errors overflow-auto mb-0" aria-label="处理错误">{escape(text)}</pre></div>',
                    }
                )

        if progress_only:
            return blocks
        media_name = str(meta.get("media_name", "") or "")
        if media_name and (processing or finished):
            blocks.append(
                {
                    "type": BlockType.MARKDOWN,
                    "text": f"[打开录像](/api/artifacts/{session_id}/media/{media_name})",
                }
            )

        return blocks

    def _consumer_cursor(self, kind: MediaKind) -> float:
        """Return the committed position of the registered media consumers."""

        ctx = self._ctx
        hub = ctx.streams if ctx is not None else None
        if hub is None:
            return 0.0
        try:
            return float(hub.consumer_cursor(kind) or 0.0)
        except Exception:  # noqa: BLE001 - progress is best effort
            return 0.0

    def _consumed_seconds(self) -> float:
        kinds = {consumer["kind"] for consumer in self._ctx.streams.consumers()}
        cursors = [self._consumer_cursor(kind) for kind in kinds]
        return min(cursors) if cursors else 0.0

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _act_refresh_courses(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._catalog.reset()
        self._invalidate_tree()
        self._ensure_tree(force=True)
        return {"ok": True, "message": "正在刷新课程"}

    def _act_open_class(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Change request navigation without changing the processing engine."""

        class_id = str(payload.get("id", ""))
        if (
            not class_id
            or Path(class_id).name != class_id
            or class_id in {".", "..", "boot"}
        ):
            return {"ok": False, "message": "无效的课次 ID"}
        if "kind" not in payload:
            try:
                cls = self._resolve_class(class_id)
            except KeyError:
                return {"ok": False, "message": "未找到课次"}
        else:
            kind = SourceKind(payload.get("kind") or SourceKind.RECORDING)
            videos = payload.get("videos") or []
            if isinstance(videos, str):
                videos = [part for part in videos.split(",") if part]
            videos = [str(video) for video in videos if str(video)]
            live_id = str(payload.get("live_id", ""))
            if kind == SourceKind.RECORDING and not videos:
                return {"ok": False, "message": "缺少录播 ID"}
            if kind == SourceKind.LIVE and not live_id:
                return {"ok": False, "message": "缺少直播 ID"}
            cls = {
                "id": class_id,
                "kind": kind,
                "course": str(payload.get("course", "")),
                "name": str(payload.get("name", "")),
                "label": str(payload.get("label", "")),
                "videos": videos,
                "recording_parts": list(payload.get("recording_parts") or []),
                "live_id": live_id,
            }
            self._remember_class(cls)
        navigation = current_navigation()
        navigation.class_id = class_id
        navigation.class_meta = cls
        self._ctx.ui.open_page("input:session")
        return {"ok": True, "open_page": "input:session"}

    def _end_open_session(self, reason: str) -> None:
        hub = self._ctx.streams
        current = hub.current
        if current.session_id != "boot" and current.ended_at is None:
            hub.end_session(reason)

    def _fetch_streams(
        self, cls: dict[str, Any], index: int
    ) -> tuple[dict[StreamView, str], str, dict[str, dict[StreamView, str]]]:
        """Resolve the class URLs and select one part for playback."""

        streams: dict[StreamView, str] = {}
        error = ""
        resolved: dict[str, dict[StreamView, str]] = {}
        try:
            if cls.get("kind") == SourceKind.LIVE:
                devices = self._catalog.get_live_devices(str(cls.get("live_id", "")))
                streams = _map_live_streams(devices)
            else:
                videos = cls.get("videos") or []
                with ThreadPoolExecutor(max_workers=max(1, len(videos))) as pool:
                    jobs = {
                        str(video_id): pool.submit(
                            self._catalog.get_recording_streams,
                            str(cls.get("course", "")),
                            str(video_id),
                        )
                        for video_id in videos
                    }
                    for video_id, job in jobs.items():
                        resolved[video_id] = _map_view_streams(job.result())
                if index >= len(videos):
                    raise InputError("没有更多小节")
                streams = resolved[str(videos[index])]
            if not streams.get(StreamView.MAIN):
                raise InputError("没有找到可播放的视频流。")
        except InputError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - never leave a job stuck preparing
            _log.warning("failed to resolve streams for %s: %s", cls.get("id", ""), exc)
            error = f"获取视频地址失败：{exc}"
        return streams, error, resolved

    def _resolve_class_streams(self, class_id: str) -> None:
        """Resolve media for an explicit course, not a browser selection."""

        with self._class_lock:
            course = self._courses.get(class_id)
            if course is None:
                return
            cls = dict(course.cls)
        streams, error, resolved = self._fetch_streams(cls, 0)
        with self._class_lock:
            course.streams = streams
            course.recording_streams = resolved
            course.error = error
            course.streams_loading = False
        self._ctx.ui.refresh("input.class.streams")

    # ------------------------------------------------------------------
    # Processing queue
    # ------------------------------------------------------------------
    def _enqueue_job_locked(self, job: _Job) -> int:
        """Add *job* to the queue; live classes jump ahead. Returns position."""

        self._jobs[job.id] = job
        if job.kind == SourceKind.LIVE:
            self._queue.insert(0, job.id)
        else:
            self._queue.append(job.id)
        return self._queue.index(job.id) + 1

    def _dispatch_jobs(self) -> None:
        """Start the next queued class when the engine is free."""

        with self._class_lock:
            if (
                self._active_job is not None
                or self._processing
                or self._playback_class_id
                or self._fill_target
            ):
                return
            job: _Job | None = None
            while self._queue:
                job_id = self._queue.pop(0)
                candidate = self._jobs.get(job_id)
                if candidate is None or candidate.state != JobState.QUEUED:
                    continue
                job = candidate
                break
            if job is None:
                return
            job.state = (
                JobState.RUNNING
                if job.streams.get(StreamView.MAIN)
                else JobState.PREPARING
            )
            job.streams_loading = not bool(job.streams.get(StreamView.MAIN))
            job.started = job.started or time.time()
            self._active_job = job
            self._processing = True
            self._processing_started = job.started
            has_streams = bool(job.streams.get(StreamView.MAIN))

        self._jobs_changed("input.queue.dispatch")
        if has_streams:
            self._begin_job(job)
        else:
            threading.Thread(
                target=self._resolve_job_streams,
                args=(job,),
                name=f"job-streams-{job.id[-12:]}",
                daemon=True,
            ).start()

    def _resolve_job_streams(self, job: _Job) -> None:
        cls = dict(job.cls)
        streams, error, resolved = self._fetch_streams(cls, job.input_part_index)
        with self._class_lock:
            if self._active_job is not job:
                return
            job.streams = streams
            job.recording_streams = resolved
            job.error = error
            job.streams_loading = False
            if error:
                job.state = JobState.FAILED
                job.finished_at = time.time()
                job.message = error
                self._active_job = None
                self._processing = False
        if error:
            self._ctx.events.emit(
                "input.error",
                message=f"无法获取视频地址：{error}",
                source=job.title(),
            )
            self._jobs_changed("input.job.failed")
            self._dispatch_jobs()
            return
        self._begin_job(job)

    def _probe_job_durations(self, job: _Job) -> None:
        with ThreadPoolExecutor(max_workers=max(1, job.parts)) as pool:
            pending = {
                pool.submit(probe_duration, streams[StreamView.MAIN]): video
                for video, streams in job.recording_streams.items()
                if video not in job.part_durations and streams.get(StreamView.MAIN)
            }
            for future in as_completed(pending):
                try:
                    duration = future.result()
                except OSError as exc:
                    _log.warning("Cannot probe recording duration: %s", exc)
                    continue
                with self._class_lock:
                    if self._active_job is not job:
                        return
                    self._record_part_duration(pending[future], duration)
                self._ctx.ui.refresh("input.duration")

    def _begin_job(self, job: _Job) -> None:
        with self._class_lock:
            if self._active_job is not job or job.state not in {
                JobState.PREPARING,
                JobState.RUNNING,
            }:
                return
            cls = dict(job.cls)
            index = job.input_part_index
            streams = dict(job.streams)
            current_session = self._ctx.streams.current.session_id
            continue_session = job.session_id == current_session and bool(
                job.session_meta
            )
            job.state = JobState.RUNNING
            if job.kind != SourceKind.LIVE and not job.durations_probing:
                job.durations_probing = True
                threading.Thread(
                    target=self._probe_job_durations, args=(job,), daemon=True
                ).start()
            # A user-initiated play always wins over a pending gap-fill pass.
            self._fill_queue = []
            self._fill_target = ""
        url = streams.get(StreamView.MAIN, "")
        if not url:
            self._fail_job(job, "没有可用的视频流")
            return

        live = job.kind == SourceKind.LIVE
        label = job.label
        videos = cls.get("videos") or []
        spec = SourceSpec(
            kind=SourceKind.LIVE if live else SourceKind.RECORDING,
            url=url,
            title=label or job.course,
            continue_session=continue_session,
            # Recordings are always processed at full speed; live is 1:1.
            realtime=live,
            download=not live and not url.startswith(("file://", "/")),
            audio_buffer_limit=self._buffer_limit(MediaKind.AUDIO),
            video_buffer_limit=self._buffer_limit(MediaKind.VIDEO) if not live else 0.0,
            course={
                "class_id": job.id,
                "course_id": str(cls.get("course", "")),
                "course_name": job.course,
                "class_label": label,
                "video_id": str(videos[index]) if index < len(videos) else "",
                "live_id": str(cls.get("live_id", "")),
            },
        )
        if not live:
            recording_parts = {
                str(part.get("video_id")): part
                for part in cls.get("recording_parts") or []
            }
            part = recording_parts.get(str(spec.course.get("video_id")), {})
            spec.final_part = index + 1 >= len(videos)
            spec.meta.update(
                recording_begin=part.get("begin", ""),
                recording_end=part.get("end", ""),
            )
            tail = spec
            for next_index in range(index + 1, len(videos)):
                next_id = str(videos[next_index])
                next_url = job.recording_streams.get(next_id, {}).get(
                    StreamView.MAIN, ""
                )
                if not next_url:
                    break
                part = recording_parts.get(next_id, {})
                tail.next_source = SourceSpec(
                    kind=SourceKind.RECORDING,
                    url=next_url,
                    title=spec.title,
                    continue_session=True,
                    realtime=False,
                    download=not next_url.startswith(("file://", "/")),
                    audio_buffer_limit=spec.audio_buffer_limit,
                    video_buffer_limit=spec.video_buffer_limit,
                    course={**spec.course, "video_id": next_id},
                    meta={
                        "recording_begin": part.get("begin", ""),
                        "recording_end": part.get("end", ""),
                    },
                    final_part=next_index + 1 >= len(videos),
                )
                tail = tail.next_source
        self._pending_message = ""
        with self._class_lock:
            if self._active_job is not job:
                return
        self._jobs_changed("input.class.processing")

        def _run() -> None:
            try:
                self._start_source(spec)
                session = self._ctx.streams.current
                with self._class_lock:
                    if self._active_job is not job or job.input_part_index != index:
                        return
                    job.session_id = session.session_id
                    if job.total_seconds:
                        session.meta["total_media_seconds"] = job.total_seconds
                    job.session_meta = dict(session.meta)
            except Exception as exc:  # noqa: BLE001 - surface startup failures
                self._ctx.events.emit(
                    "input.error",
                    message=f"启动处理失败：{exc}",
                    source=spec.title,
                )
                self._fail_job(job, str(exc))
            self._ctx.ui.refresh("input.class.started")

        # Downloading + starting ffmpeg can take minutes; never block the
        # HTTP action that triggered it.
        threading.Thread(target=_run, name="input-start", daemon=True).start()

    def _fail_job(self, job: _Job, message: str) -> None:
        with self._class_lock:
            if self._active_job is not job:
                return
        with self._class_lock:
            job.state = JobState.FAILED
            job.error = message or job.error
            job.message = message
            job.finished_at = time.time()
            if self._active_job is job:
                self._active_job = None
            self._processing = False
        session = self._ctx.streams.current
        session.meta["input_failed"] = True
        vlm = self._ctx.service("vlm")
        if vlm is not None:
            vlm.cancel(session.session_id)
        notes = self._ctx.service("notes")
        if notes is not None:
            notes.stop(session.session_id)
        self._sources.stop()
        self._end_open_session("processing failed")
        self._jobs_changed("input.job.failed")
        self._dispatch_jobs()

    def _class_from_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        class_id = str(payload.get("id") or current_navigation().class_id)
        if not class_id:
            return None
        if class_id == current_navigation().class_id and self._current_class:
            return dict(self._current_class)
        try:
            return self._resolve_class(class_id)
        except KeyError:
            return None

    def _act_delete(self, payload: dict[str, Any]) -> dict[str, Any]:
        cls = self._class_from_payload(payload)
        if cls is None:
            return {"ok": False, "message": "未选择课次"}
        if str(cls["id"]) in {self._fill_target, self._playback_class_id}:
            self._act_stop({"id": str(cls["id"])})
        with self._class_lock:
            job = self._jobs.get(str(cls.get("id", "")))
            was_active = job is not None and self._active_job is job
            if was_active:
                self._active_job = None
                self._processing = False
            if job is not None:
                self._jobs.pop(job.id, None)
                while job.id in self._queue:
                    self._queue.remove(job.id)
        sid = str(cls["id"])
        session = self._ctx.streams.get_session(sid)
        if session is not None:
            session.meta["processing_cancelled"] = True
        if was_active:
            self._sources.stop()
            self._end_open_session("results deleted")
        vlm = self._ctx.service("vlm")
        if vlm is not None:
            vlm.cancel(sid)
        consumers = [
            service
            for name in ("stt", "vlm")
            if (service := self._ctx.service(name)) is not None
        ]
        while any(service.processing(sid) for service in consumers):
            time.sleep(0.05)
        notes = self._ctx.service("notes")
        if notes is not None:
            notes.reset(sid)
        directory = self._ctx.data_dir / "sessions" / sid
        for name in ("transcript.jsonl", "visual.jsonl", "coverage.json"):
            (directory / name).unlink(missing_ok=True)
        shutil.rmtree(directory / "vlm", ignore_errors=True)
        path = self._session_meta_path(sid)
        if path.exists():
            try:
                record = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                record = {"session_id": sid, "meta": {}}
            record["ended_at"] = 0
            meta = record.get("meta") or {}
            for key in ("processing_cancelled", "input_failed"):
                meta.pop(key, None)
            record["meta"] = meta
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), "utf-8"
            )
            temporary.replace(path)
        self._ctx.streams.clear_results(sid)
        for name in ("stt", "vlm"):
            service = self._ctx.service(name)
            if service is not None:
                service.reset(sid)
        with self._class_lock:
            self._courses[sid] = _Course(cls=dict(cls))
        self._jobs_changed("input.results.deleted")
        return {"ok": True, "message": "已删除板书"}

    def _act_start_processing(self, payload: dict[str, Any]) -> dict[str, Any]:
        resume = False
        position = 0
        busy = ""
        cls = self._class_from_payload(payload)
        with self._class_lock:
            if cls is None:
                return {"ok": False, "message": "未选择课次"}
            class_id = str(cls["id"])
            if class_id in {self._playback_class_id, self._fill_target}:
                if self._processing:
                    return {"ok": True, "message": "正在处理"}
                return self._act_resume({"id": class_id})
            if class_id.startswith("manual_"):
                return self._act_play_url(
                    {
                        "url": cls.get("url", ""),
                        "title": cls.get("name", ""),
                        "kind": cls.get("kind", SourceKind.URL),
                    }
                )
            job = self._jobs.get(str(cls.get("id", "")))
            if job is not None and job.state in {JobState.PREPARING, JobState.RUNNING}:
                if self._active_job is job:
                    return {"ok": True, "message": "正在处理"}
                # A stale preparing job: put it back in line.
                job.state = JobState.QUEUED
            if job is not None and job.state == JobState.PAUSED:
                if self._active_job is job:
                    job.state = JobState.RUNNING
                    self._processing = True
                    resume = job.kind != SourceKind.LIVE
                else:
                    job.state = JobState.QUEUED
                    if job.id not in self._queue:
                        self._queue.insert(0, job.id)
            elif job is not None and job.state == JobState.QUEUED:
                # Clicking "处理" on a waiting class moves it to the front.
                if job.id in self._queue:
                    self._queue.remove(job.id)
                self._queue.insert(0, job.id)
            elif job is None or job.state in _JOB_TERMINAL:
                if job is not None:
                    self._jobs.pop(job.id, None)
                course = self._courses.get(str(cls["id"])) or _Course(cls=dict(cls))
                job = _Job(
                    id=str(cls.get("id", "")),
                    cls=cls,
                    state=JobState.QUEUED,
                    streams=dict(course.streams),
                    streams_loading=course.streams_loading,
                    recording_streams=dict(course.recording_streams),
                )
                job.streams_loading = not bool(job.streams.get(StreamView.MAIN))
                self._enqueue_job_locked(job)
            if not resume:
                position = self._queue.index(job.id) + 1 if job.id in self._queue else 0
                busy = self._active_job.title() if self._active_job else ""

        if resume:
            self._sources.resume()
            self._jobs_changed("input.class.resumed")
            return {"ok": True, "message": "已继续处理"}
        self._dispatch_jobs()
        with self._class_lock:
            if job.state == JobState.FAILED:
                return {"ok": False, "message": job.error or "处理失败"}
            if self._jobs.get(job.id) is not job:
                return {"ok": True}
            if self._active_job is job:
                if job.state == JobState.RUNNING:
                    return {"ok": True, "message": "开始处理"}
                if job.state == JobState.PREPARING:
                    return {"ok": True, "message": "正在准备视频…"}
        where = f"已加入队列（第 {position} 位）" if position else "已加入队列"
        if busy:
            where += f"；当前正在处理：{busy}"
        return {"ok": True, "message": where}

    # ------------------------------------------------------------------
    # Queue actions (per-job, do not depend on the selected class)
    # ------------------------------------------------------------------
    def _job_from_payload(self, payload: dict[str, Any]) -> _Job | None:
        job_id = str(payload.get("id", "") or "")
        with self._class_lock:
            if job_id:
                return self._jobs.get(job_id)
            return self._selected_job_locked()

    def _act_queue_pause(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self._job_from_payload(payload)
        if job is None:
            return {"ok": False, "message": "没有可暂停的任务"}
        with self._class_lock:
            if self._active_job is not job or job.state != JobState.RUNNING:
                return {"ok": False, "message": "该任务尚未运行"}
            job.state = JobState.PAUSED
            self._processing = False
            kind = job.kind
        if kind == SourceKind.LIVE:
            return self._act_queue_cancel({"id": job.id})
        self._sources.pause()
        self._jobs_changed("input.queue.paused")
        return {"ok": True, "message": "已暂停处理"}

    def _act_queue_resume(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self._job_from_payload(payload)
        if job is None:
            return {"ok": False, "message": "没有可继续的任务"}
        with self._class_lock:
            if self._active_job is not job or job.state != JobState.PAUSED:
                return {"ok": False, "message": "该任务没有暂停"}
            job.state = JobState.RUNNING
            self._processing = True
        self._sources.resume()
        self._jobs_changed("input.queue.resumed")
        return {"ok": True, "message": "已继续处理"}

    def _act_queue_cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self._job_from_payload(payload)
        if job is None:
            return {"ok": False, "message": "没有可取消的任务"}
        with self._class_lock:
            was_active = self._active_job is job
            self._jobs.pop(job.id, None)
            while job.id in self._queue:
                self._queue.remove(job.id)
            if was_active:
                self._active_job = None
                self._processing = False
                course = self._courses.get(job.id)
                if course is not None:
                    course.finished = False
        if was_active:
            session = self._ctx.streams.get_session(job.session_id or job.id)
            if session is not None:
                session.meta["processing_cancelled"] = True
            vlm = self._ctx.service("vlm")
            if vlm is not None:
                vlm.cancel(job.session_id or job.id)
            notes = self._ctx.service("notes")
            if notes is not None:
                notes.stop(job.session_id or job.id)
            self._sources.stop()
            self._end_open_session("processing canceled")
        self._jobs_changed("input.queue.canceled")
        self._dispatch_jobs()
        return {"ok": True}

    def _act_queue_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self._job_from_payload(payload)
        if job is None:
            return {"ok": False, "message": "任务记录已清理"}
        return self._act_start_processing({"id": job.id})

    def _act_queue_clear(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._class_lock:
            done = [
                job_id
                for job_id, job in self._jobs.items()
                if job.state in _JOB_TERMINAL
            ]
            for job_id in done:
                self._jobs.pop(job_id, None)
        self._jobs_changed("input.queue.cleared")
        return {"ok": True, "message": f"已清理 {len(done)} 条记录"}

    def _audio_enabled(self) -> bool:
        service = self._ctx.service("stt")
        return bool(service is not None and service.enabled())

    def _buffer_limit(self, kind: MediaKind) -> float:
        """Return the buffer limit published by registered media consumers."""

        ctx = self._ctx
        hub = ctx.streams if ctx is not None else None
        if hub is None:
            return 0.0
        try:
            return max(0.0, float(hub.consumer_buffer_limit(kind) or 0.0))
        except Exception:  # noqa: BLE001 - consumer buffer limits are best effort
            return 0.0

    def _continue_fill(self, recording: dict[str, Any]) -> None:
        """Start the next recording part of a gap-fill pass."""

        if self._start_fill_part(recording, continue_session=True):
            return
        while self._fill_queue:
            if self._start_fill_part(self._fill_queue.pop(0), continue_session=True):
                return
        self._end_fill_or_playback()

    def _end_fill_or_playback(self) -> None:
        with self._class_lock:
            self._processing = False
            class_id = self._fill_target or self._playback_class_id
            course = self._courses.get(class_id)
            if course is not None:
                session = self._ctx.streams.current
                course.finished = True
                course.session_id = session.session_id
                course.session_meta = dict(session.meta)
            self._playback_class_id = ""
            self._fill_queue = []
            self._fill_target = ""
        self._sources.stop()
        self._end_open_session("playback finished")
        self._prune_media()
        self._ctx.ui.refresh("input.finished")
        self._dispatch_jobs()

    def _record_part_duration(self, video_id: str, duration: float) -> None:
        if not video_id or duration <= 0:
            return
        with self._class_lock:
            job = self._active_job
            if job is None:
                return
            cls = job.cls
            videos = [str(video) for video in cls.get("videos") or []]
            if video_id not in videos:
                return
            durations = job.part_durations
            durations[video_id] = duration
            # A partial sum is not the duration of the class.
            total = (
                sum(durations[video] for video in videos)
                if all(video in durations for video in videos)
                else 0.0
            )
            job.total_seconds = total
            session = self._ctx.streams.get_session(job.session_id)
            if session is not None and total:
                session.meta["total_media_seconds"] = total

    def _act_pause_processing(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self._job_from_payload(payload)
        if job is not None:
            if job.state in {JobState.QUEUED, JobState.PREPARING}:
                return self._act_queue_cancel({"id": job.id})
            if self._active_job is job and job.state == JobState.RUNNING:
                return self._act_queue_pause({"id": job.id})
            return {"ok": True, "message": "当前没有处理任务"}
        with self._class_lock:
            class_id = str(payload.get("id") or current_navigation().class_id)
            if not class_id or class_id not in {
                self._fill_target,
                self._playback_class_id,
            }:
                return {"ok": True, "message": "当前没有处理任务"}
            kind = self._ctx.streams.current.meta.get("kind")
            self._processing = False
        if kind == SourceKind.LIVE:
            # A live stream cannot be paused: stop our ingestion and resume at
            # the live edge later.
            with self._class_lock:
                self._fill_target = ""
                self._fill_queue = []
                self._playback_class_id = ""
            self._sources.stop()
            self._end_open_session("live paused")
            self._jobs_changed("input.class.paused")
            self._dispatch_jobs()
            return {"ok": True, "message": "已停止直播处理"}
        self._sources.pause()
        self._jobs_changed("input.class.paused")
        return {"ok": True, "message": "已暂停处理"}

    def _on_class_ended(self) -> None:
        """Natural end of the current recording period."""

        with self._class_lock:
            job = self._active_job
            processing = self._processing

        if not processing:
            return

        if job is not None:
            videos = job.cls.get("videos") or []
            if job.input_part_index + 1 < len(videos):
                with self._class_lock:
                    job.input_part_index += 1
                    next_id = str(videos[job.input_part_index])
                    job.streams = dict(job.recording_streams.get(next_id, {}))
                    job.streams_loading = not bool(job.streams.get(StreamView.MAIN))
                    job.state = (
                        JobState.PREPARING if job.streams_loading else JobState.RUNNING
                    )
                self._ctx.ui.refresh("input.class.next")
                if job.streams.get(StreamView.MAIN):
                    self._begin_job(job)
                else:
                    threading.Thread(
                        target=self._resolve_job_streams,
                        args=(job,),
                        name="job-next-part",
                        daemon=True,
                    ).start()
                return
            if job.kind != SourceKind.LIVE:
                self._wait_for_consumers()
            self._finish_job(job, "playback finished")
            return

        # Manual URL playback or a gap-fill pass.
        if self._fill_queue:
            nxt = self._fill_queue.pop(0)
            threading.Thread(
                target=self._continue_fill,
                args=(nxt,),
                name="fill-next-part",
                daemon=True,
            ).start()
            return
        self._wait_for_consumers()
        self._end_fill_or_playback()

    def _wait_for_consumers(self, timeout: float = 3600.0) -> bool:
        """Block until every registered media consumer went idle."""

        ctx = self._ctx
        hub = ctx.streams if ctx is not None else None
        if hub is None:
            return True
        deadline = time.monotonic() + max(1.0, float(timeout))
        _log.info("draining media consumers before closing the session")
        while time.monotonic() < deadline:
            with self._class_lock:
                if not self._processing:
                    return True
            try:
                busy = list(hub.consumers_busy() or [])
            except Exception:  # noqa: BLE001 - a broken probe must not block
                _log.exception("consumer busy check failed")
                busy = []
            if not busy:
                _log.info("consumers drained")
                return True
            time.sleep(0.5)
        self._ctx.events.emit(
            "input.error",
            message="板书/语音处理超时，将按已完成的内容生成笔记．",
            source="",
        )
        return False

    def _finish_job(self, job: _Job, reason: str) -> None:
        """Mark one job done, close its session, and start the next in queue."""

        hub = self._ctx.streams
        with self._class_lock:
            job.state = JobState.DONE
            job.finished_at = time.time()
            job.message = "处理完成"
            if hub.current.session_id != "boot":
                job.session_id = hub.current.session_id
                job.session_meta = dict(hub.current.meta)
            if self._active_job is job:
                self._active_job = None
            self._processing = False
            course = self._courses.setdefault(job.id, _Course(cls=dict(job.cls)))
            course.finished = True
            course.session_id = job.session_id
            course.session_meta = dict(job.session_meta)
        self._sources.stop()
        self._end_open_session(reason)
        self._prune_media()
        self._jobs_changed("input.class.finished")
        self._dispatch_jobs()

    def _prune_media(self) -> None:
        """Keep only the most recent N recording folders."""

        try:
            keep = int(self._ctx.require("settings").get("input.keep_recordings", 4))
        except Exception:  # noqa: BLE001
            keep = 4
        keep = max(0, keep)
        root = self._ctx.data_dir / "sessions"
        try:
            folders = [path for path in root.glob("*/media") if path.is_dir()]
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
        kind = SourceKind(payload.get("kind") or SourceKind.URL)
        class_id = f"manual_{source_id(url)}"

        live = kind == SourceKind.LIVE
        spec = SourceSpec(
            kind=kind,
            url=url,
            title=title,
            realtime=live,
            download=not live and url.startswith(("http://", "https://")),
            audio_buffer_limit=self._buffer_limit(MediaKind.AUDIO),
            video_buffer_limit=self._buffer_limit(MediaKind.VIDEO) if not live else 0.0,
            course={"class_id": class_id},
            meta={"url": url},
        )
        with self._class_lock:
            if (
                self._active_job is not None
                or self._processing
                or self._playback_class_id
                or self._fill_target
            ):
                return {"ok": False, "message": "输入源正在使用中"}
            self._processing = True
            self._processing_started = time.time()
            self._playback_class_id = class_id
            cls = {
                "id": class_id,
                "name": title,
                "kind": kind,
                "videos": [],
                "url": url,
            }
            course = _Course(cls=cls, streams={StreamView.MAIN: url})
            self._courses[class_id] = course
        navigation = current_navigation()
        navigation.class_id = class_id
        navigation.class_meta = dict(cls)
        self._ctx.ui.open_page("input:session")

        def _run() -> None:
            try:
                self._start_source(spec)
                with self._class_lock:
                    session = self._ctx.streams.current
                    course.session_id = session.session_id
                    course.session_meta = dict(session.meta)
            except Exception as exc:  # noqa: BLE001 - surface startup failures
                with self._class_lock:
                    self._processing = False
                    self._playback_class_id = ""
                    course.error = str(exc)
                self._ctx.events.emit(
                    "input.error",
                    message=f"启动播放失败：{exc}",
                    source=title,
                )
                self._ctx.ui.refresh("input.error")
                self._dispatch_jobs()

        # Downloading + starting ffmpeg must never block the action request.
        threading.Thread(target=_run, name="input-start-url", daemon=True).start()
        return {"ok": True, "message": "开始播放", "open_page": "input:session"}

    def _act_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._pending_message = ""
        self._last_error = ""
        job = self._job_from_payload(payload)
        if job is not None:
            return self._act_queue_cancel({"id": job.id})
        class_id = str(payload.get("id") or current_navigation().class_id)
        with self._class_lock:
            if not class_id or class_id not in {
                self._fill_target,
                self._playback_class_id,
            }:
                return {"ok": False, "message": "当前没有处理任务"}
            self._processing = False
            self._fill_queue = []
            self._fill_target = ""
            self._playback_class_id = ""
        session = self._ctx.streams.current
        session.meta["processing_cancelled"] = True
        vlm = self._ctx.service("vlm")
        if vlm is not None:
            vlm.cancel(session.session_id)
        notes = self._ctx.service("notes")
        if notes is not None:
            notes.stop(session.session_id)
        self._sources.stop()
        self._end_open_session("stopped")
        self._jobs_changed("input.stopped")
        self._dispatch_jobs()
        return {"ok": True, "message": "已停止"}

    def _act_pause(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._act_pause_processing(payload)

    def _act_resume(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self._job_from_payload(payload)
        if job is not None:
            return self._act_queue_resume({"id": job.id})
        class_id = str(payload.get("id") or current_navigation().class_id)
        with self._class_lock:
            if not class_id or class_id not in {
                self._fill_target,
                self._playback_class_id,
            }:
                return {"ok": False, "message": "当前没有处理任务"}
            self._processing = True
        self._sources.resume()
        return {"ok": True, "message": "已继续"}


# ---------------------------------------------------------------------------
# Settings specs
# ---------------------------------------------------------------------------

_INPUT_SETTINGS: list[SettingsSpec] = [
    SettingsSpec(
        key="input.canvas_token",
        label="访问令牌",
        type=SettingType.SECRET,
        default="",
        group="Canvas",
        section="连接与认证",
        description="SJTU Canvas 的 Bearer Token．留空表示未配置．",
        order=0,
        secret=True,
        placeholder="Canvas 访问令牌",
        env=("CANVAS_TOKEN",),
    ),
    SettingsSpec(
        key="input.canvas_base_url",
        label="Canvas 地址",
        type=SettingType.STRING,
        default="https://oc.sjtu.edu.cn",
        group="Canvas",
        section="连接与认证",
        description="Canvas 实例的根地址，例如 https://oc.sjtu.edu.cn．",
        order=1,
        env=("CANVAS_BASE_URL",),
    ),
    SettingsSpec(
        key="input.video_fps",
        label="视频帧率（帧/秒）",
        type=SettingType.FLOAT,
        default=4.0,
        group="播放",
        section="视频输出",
        description="下次播放生效．",
        order=10,
    ),
    SettingsSpec(
        key="input.video_width",
        label="视频宽度（像素）",
        type=SettingType.INT,
        default=0,
        group="播放",
        section="视频输出",
        description="0 表示保持原分辨率，不缩放画面．下次播放生效．",
        order=11,
    ),
    SettingsSpec(
        key="input.audio_rate",
        label="音频采样率（Hz）",
        type=SettingType.INT,
        default=16000,
        group="播放",
        section="音频输出",
        description="下次播放生效．",
        order=12,
    ),
    SettingsSpec(
        key="input.keep_recordings",
        label="录像保留数量（节）",
        type=SettingType.INT,
        default=4,
        group="录播",
        section="本地存储",
        description="超出保留数量时先删除最旧录像，当前会话除外．0 表示不保留历史录像．",
        order=13,
    ),
    SettingsSpec(
        key="input.live_device",
        label="默认设备索引",
        type=SettingType.SELECT,
        default="0",
        group="直播",
        section="直播来源",
        description="",
        options=("0", "1", "2", "3"),
        order=20,
    ),
    SettingsSpec(
        key="input.reconnect",
        label="自动重连",
        type=SettingType.BOOL,
        default=True,
        group="直播",
        section="断线处理",
        description="直播连接中断后自动尝试重新连接．下次播放生效．",
        order=21,
    ),
]
