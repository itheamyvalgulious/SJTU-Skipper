"""STT plugin: Google Cloud Speech-to-Text v2 over the audio stream.

The plugin consumes ``hub.current.audio`` in fixed-length chunks and appends
recognized text to ``hub.current.transcript`` with ``source="stt"``.  It only
talks to the streams, so the behaviour layer keeps seeing plain text.

Authentication is OAuth (service-account JSON, authorized-user JSON, or ADC):
v2 does not accept API keys, and ``chirp_*`` models only exist in v2.  The
model is one explicit choice from the settings; Google's optional "model
adaptation" (uploading phrase sets / custom classes) is not used.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from core.plugin import Plugin, PluginContext
from plugins.settings.spec import SettingsSpec

from .credentials import GoogleTokenProvider
from .errors import SttError
from .google_cloud import (
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    MODEL_OPTIONS,
    GoogleSpeechClient,
    SpeechConfig,
    SpeechSegment,
)

LOGGER_NAME = "lecture_agent.plugin.stt"

TAB_ID = "classroom"
ITEM_ID = "stt.transcript"
PAGE_ID = "stt:transcript"

BACKEND = "google-cloud-stt-v2"

_DEFAULT_CHUNK_SECONDS = 20.0
_MIN_CHUNK_SECONDS = 1.0
# v2 Recognize accepts at most 60 s / 10 MB per request; keep a margin so
# WAV/base64/JSON overhead cannot push a request over the limit.
_MAX_CHUNK_SECONDS = 55.0
_MIN_FLUSH_SECONDS = 1.0
_MAX_PARAGRAPHS = 30

_STATE_LABELS = {
    "disabled": "已关闭",
    "needs_credentials": "缺少凭据",
    "error": "错误",
    "idle": "等待音频",
    "running": "运行中",
}


class STTService:
    """Public ``stt`` service."""

    def __init__(self, plugin: "SttPlugin") -> None:
        self._plugin = plugin

    def status(self) -> dict[str, Any]:
        return self._plugin.status()

    def transcribe(
        self, start: float | None = None, end: float | None = None
    ) -> dict[str, Any]:
        """Transcribe one audio range now (used by tests and manual calls)."""

        return self._plugin.transcribe_range(start, end)


class SttPlugin(Plugin):
    """Google Cloud Speech-to-Text v2 worker over the audio stream."""

    id = "stt"
    requires = ("settings",)
    provides = ("stt",)

    def __init__(self) -> None:
        self._ctx: PluginContext | None = None
        self._logger = logging.getLogger(LOGGER_NAME)
        self._lock = threading.RLock()
        self._auth_lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._auth: GoogleTokenProvider | None = None
        self._auth_signature: tuple[str, str, str, float] | None = None
        self._cursor = 0.0
        self._last_session = ""
        self._segments = 0
        self._chars = 0
        self._last_error = ""
        self._reported_error = ""
        self._consecutive_errors = 0
        self._page_added = False
        self._item_added = False
        self._tab_added = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self._register_settings()
        ctx.register_service("stt", STTService(self))
        ctx.meta.register("stt", self._meta, plugin_id=self.id)
        self._setup_ui(ctx)
        ctx.events.on("session.started", self._on_session_started, plugin_id=self.id)
        ctx.events.on("session.stopped", self._on_session_stopped, plugin_id=self.id)
        ctx.events.on("settings.changed", self._on_settings_changed, plugin_id=self.id)
        if self._setting("stt.enabled", True):
            self._start_worker()
        ctx.logger.info("stt ready: google cloud speech-to-text v2 (oauth)")

    def teardown(self) -> None:
        with self._lock:
            worker = self._worker
            self._worker = None
        self._stop_event.set()
        if worker is not None and worker.is_alive():  # pragma: no cover - race
            worker.join(timeout=3.0)
        with self._auth_lock:
            auth = self._auth
            self._auth = None
            self._auth_signature = None
        if auth is not None:
            auth.close()
        ctx = self._ctx
        if ctx is not None:
            if self._page_added:
                try:
                    ctx.ui.remove_page(PAGE_ID)
                except Exception:  # noqa: BLE001 - UI may already be gone
                    pass
                self._page_added = False
            if getattr(self, "_command_added", False):
                try:
                    ctx.ui.unregister_command("stt.debug")
                except Exception:  # noqa: BLE001
                    pass
                self._command_added = False
        self._ctx = None

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _register_settings(self) -> None:
        specs = [
            SettingsSpec(
                key="stt.enabled",
                label="启用语音转写",
                type="bool",
                default=True,
                group="STT",
                description="是否启动 Google Cloud Speech-to-Text v2 工作线程",
                group_note=(
                    "使用 Google Cloud Speech-to-Text **v2**（OAuth）。"
                    "v2 不接受 API Key，需要服务账号 JSON（或 gcloud ADC / "
                    "authorized_user JSON），并在项目中启用 Speech-to-Text API、"
                    "授予 speech.recognizers.recognize 权限。"
                    "凭据里没有 project 时请填写项目 ID。"
                ),
                order=0,
            ),
            SettingsSpec(
                key="stt.credentials_json",
                label="服务账号 JSON",
                type="text",
                secret=True,
                default="",
                group="STT",
                description=(
                    "粘贴 service_account 或 authorized_user JSON。"
                    "以 secret 形式保存在 settings.json，不会回显。"
                ),
                order=1,
            ),
            SettingsSpec(
                key="stt.credentials_file",
                label="凭据文件路径",
                type="string",
                default="",
                group="STT",
                description=(
                    "可选的 JSON 文件路径（上项为空时使用）；"
                    "留空且上项为空时回退到 GOOGLE_APPLICATION_CREDENTIALS / ADC。"
                ),
                env=("GOOGLE_APPLICATION_CREDENTIALS",),
                order=2,
            ),
            SettingsSpec(
                key="stt.project_id",
                label="Project ID",
                type="string",
                default="",
                group="STT",
                description="Google Cloud 项目 ID；留空则从凭据 JSON 读取。",
                order=3,
            ),
            SettingsSpec(
                key="stt.location",
                label="区域（location）",
                type="string",
                default=DEFAULT_LOCATION,
                group="STT",
                description=(
                    "v2 recognizer 区域，例如 us、eu、asia-northeast1、"
                    "asia-southeast1；chirp_3 需在支持的区域使用。"
                ),
                order=4,
            ),
            SettingsSpec(
                key="stt.model",
                label="转写模型",
                type="select",
                default=DEFAULT_MODEL,
                group="STT",
                description=(
                    "v2 模型，每块音频固定使用这一项。"
                    "暂不使用「模型自适应」（Model adaptation：上传 phrase sets / "
                    "custom classes 预制名词来提升专业术语准确率）。"
                    "模型是否支持某语言/区域由 Google 决定，不支持的组合会返回 400。"
                ),
                options=MODEL_OPTIONS,
                order=5,
            ),
            SettingsSpec(
                key="stt.language_code",
                label="语言代码",
                type="string",
                default="cmn-Hans-CN",
                group="STT",
                description=(
                    "BCP-47 语言代码；中文课堂用 cmn-Hans-CN，英文用 en-US。"
                    "所选模型需受该语言支持，不支持的组合会返回 400。"
                ),
                order=6,
            ),
            SettingsSpec(
                key="stt.chunk_seconds",
                label="分块时长（秒）",
                type="float",
                default=_DEFAULT_CHUNK_SECONDS,
                group="STT",
                description=(
                    "每次请求发送的音频长度。v2 同步 Recognize 硬上限 60 秒"
                    f"（单请求音频 ≤10 MB），代码限制在 {_MIN_CHUNK_SECONDS:g}–"
                    f"{_MAX_CHUNK_SECONDS:g} 秒预留处理余量。"
                ),
                order=7,
            ),
            SettingsSpec(
                key="stt.enable_punctuation",
                label="自动标点",
                type="bool",
                default=True,
                group="STT",
                description="请求 Google 自动添加标点。",
                order=8,
            ),
            SettingsSpec(
                key="stt.timeout",
                label="请求超时（秒）",
                type="float",
                default=60.0,
                group="STT",
                description="单次 v2 Recognize 请求的超时时间。",
                order=9,
            ),
        ]
        settings = self._ctx.service("settings") if self._ctx is not None else None
        if settings is None:
            return
        for spec in specs:
            try:
                settings.register(spec)
            except Exception:  # noqa: BLE001 - duplicate registration
                self._logger.warning("settings register failed for %s", spec.key)

    def _setting(self, key: str, default: Any = None) -> Any:
        ctx = self._ctx
        settings = ctx.service("settings") if ctx is not None else None
        if settings is None:
            return default
        return settings.get(key, default)

    def _credentials_configured(self) -> bool:
        if str(self._setting("stt.credentials_json", "") or "").strip():
            return True
        if str(self._setting("stt.credentials_file", "") or "").strip():
            return True
        return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip())

    def _credentials(self) -> GoogleTokenProvider:
        json_text = str(self._setting("stt.credentials_json", "") or "").strip()
        file_path = str(self._setting("stt.credentials_file", "") or "").strip()
        project = str(self._setting("stt.project_id", "") or "").strip()
        try:
            timeout = float(self._setting("stt.timeout", 60.0) or 60.0)
        except (TypeError, ValueError):
            timeout = 60.0
        signature = (json_text, file_path, project, round(timeout, 3))
        with self._auth_lock:
            if self._auth is None or self._auth_signature != signature:
                old = self._auth
                self._auth = None
                self._auth_signature = None
                if old is not None:
                    old.close()
                self._auth = GoogleTokenProvider(
                    credentials_json=json_text,
                    credentials_file=file_path,
                    project_id=project,
                    timeout=timeout,
                )
                self._auth_signature = signature
            return self._auth

    def _speech_config(self) -> tuple[SpeechConfig, GoogleTokenProvider]:
        provider = self._credentials()
        project = str(self._setting("stt.project_id", "") or "").strip()
        if not project:
            project = provider.project_id
        location = str(
            self._setting("stt.location", DEFAULT_LOCATION) or DEFAULT_LOCATION
        ).strip()
        model = str(self._setting("stt.model", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
        language = str(
            self._setting("stt.language_code", "cmn-Hans-CN") or "cmn-Hans-CN"
        ).strip()
        try:
            timeout = float(self._setting("stt.timeout", 60.0) or 60.0)
        except (TypeError, ValueError):
            timeout = 60.0
        config = SpeechConfig(
            project_id=project,
            location=location or DEFAULT_LOCATION,
            model=model or DEFAULT_MODEL,
            language_code=language or "cmn-Hans-CN",
            punctuation=bool(self._setting("stt.enable_punctuation", True)),
            timeout=timeout,
        )
        return config, provider

    def _chunk_seconds(self) -> float:
        try:
            value = float(self._setting("stt.chunk_seconds", _DEFAULT_CHUNK_SECONDS))
        except (TypeError, ValueError):
            value = _DEFAULT_CHUNK_SECONDS
        return min(_MAX_CHUNK_SECONDS, max(_MIN_CHUNK_SECONDS, value))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        ctx = self._ctx
        session = ctx.streams.current if ctx is not None else None
        enabled = bool(self._setting("stt.enabled", True))
        configured = self._credentials_configured()
        with self._lock:
            segments = self._segments
            chars = self._chars
            cursor = self._cursor
            last_error = self._last_error
            alive = self._worker is not None and self._worker.is_alive()

        if not enabled:
            state = "disabled"
            error = "已关闭转写"
        elif not configured:
            state = "needs_credentials"
            error = "未配置 Google 凭据"
        elif last_error:
            state = "error"
            error = last_error
        elif alive:
            state = "running"
            error = ""
        else:
            state = "idle"
            error = ""

        return {
            "state": state,
            "error": error,
            "backend": BACKEND,
            "model": str(self._setting("stt.model", DEFAULT_MODEL) or DEFAULT_MODEL),
            "location": str(self._setting("stt.location", DEFAULT_LOCATION) or DEFAULT_LOCATION),
            "session_id": session.session_id if session is not None else "",
            "audio_seconds": (
                round(session.audio.duration, 3) if session is not None else 0.0
            ),
            "cursor": round(cursor, 3),
            "segments": segments,
            "chars": chars,
            "worker_alive": alive,
        }

    def _meta(self) -> dict[str, Any]:
        status = self.status()
        return {
            "backend": status["backend"],
            "state": status["state"],
            "model": status["model"],
            "location": status["location"],
            "session_id": status["session_id"],
            "audio_seconds": status["audio_seconds"],
            "segments": status["segments"],
            "chars": status["chars"],
        }

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def transcribe_range(
        self, start: float | None = None, end: float | None = None
    ) -> dict[str, Any]:
        """Transcribe ``[start, end)`` of the current audio stream now."""

        ctx = self._ctx
        if ctx is None:
            raise SttError("STT 插件尚未初始化")
        session = ctx.streams.current
        audio = session.audio
        if start is None:
            with self._lock:
                begin = self._cursor
        else:
            begin = float(start)
        begin = max(0.0, begin)
        finish = (
            float(end)
            if end is not None
            else min(audio.duration, begin + self._chunk_seconds())
        )
        finish = min(float(finish), audio.duration)
        if finish - begin <= 1e-6:
            return {
                "ok": True,
                "skipped": True,
                "start": begin,
                "end": finish,
                "segments": 0,
                "chars": 0,
            }
        if audio.sample_width != 2:
            raise SttError("Google STT 仅支持 16-bit PCM 音频")

        wav = audio.read_as_wav(begin, finish)
        if not wav:
            return {
                "ok": True,
                "skipped": True,
                "start": begin,
                "end": finish,
                "segments": 0,
                "chars": 0,
            }

        config, provider = self._speech_config()
        with GoogleSpeechClient(config, provider.token) as client:
            segments = client.recognize(wav, duration=finish - begin)

        appended, chars = self._append_segments(segments, base=begin, model=config.model)
        if appended:
            ctx.events.emit(
                "stt.segment",
                session_id=session.session_id,
                start=begin,
                end=finish,
                segments=appended,
                chars=chars,
                model=config.model,
            )
            ctx.ui.refresh("stt.segment")
        return {
            "ok": True,
            "start": begin,
            "end": finish,
            "segments": appended,
            "chars": chars,
        }

    def _append_segments(
        self,
        segments: list[SpeechSegment],
        *,
        base: float,
        model: str,
    ) -> tuple[int, int]:
        ctx = self._ctx
        if ctx is None:
            return 0, 0
        transcript = ctx.streams.current.transcript
        appended = 0
        chars = 0
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            item_start = base + max(0.0, segment.start)
            item_end = base + max(segment.start, segment.end)
            item = transcript.append(
                text,
                start=item_start,
                end=item_end,
                source="stt",
                meta={
                    "model": model,
                    "confidence": round(float(segment.confidence), 4),
                    "language_code": segment.language_code,
                },
            )
            if item is None:
                continue
            appended += 1
            chars += len(item.text)
        with self._lock:
            self._segments += appended
            self._chars += chars
            self._consecutive_errors = 0
            self._last_error = ""
            self._reported_error = ""
        return appended, chars

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _start_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._worker_loop, name="stt-worker", daemon=True
            )
            self._worker.start()

    def _stop_worker(self) -> None:
        with self._lock:
            worker = self._worker
            self._worker = None
        self._stop_event.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)

    def _worker_loop(self) -> None:
        self._logger.info("STT worker started")
        while not self._stop_event.is_set():
            try:
                self._process_next_window()
            except Exception as exc:  # noqa: BLE001 - worker must survive
                self._handle_error(exc)
        try:
            self._flush_remaining()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            self._logger.exception("STT flush on shutdown failed")
        self._logger.info("STT worker stopped")

    def _process_next_window(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        session = ctx.streams.current
        sid = session.session_id
        with self._lock:
            if sid != self._last_session:
                self._last_session = sid
                self._cursor = 0.0
                self._segments = 0
                self._chars = 0
                self._last_error = ""
                self._reported_error = ""
            cursor = self._cursor

        if not bool(self._setting("stt.enabled", True)):
            self._stop_event.wait(1.0)
            return
        if not self._credentials_configured():
            self._stop_event.wait(2.0)
            return

        needed = cursor + self._chunk_seconds()
        if session.audio.duration + 1e-9 < needed:
            if session.ended_at is not None:
                self._flush_remaining()
            else:
                self._stop_event.wait(0.5)
            return

        self.transcribe_range(cursor, needed)
        with self._lock:
            if self._last_session == session.session_id:
                self._cursor = needed

    def _flush_remaining(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        session = ctx.streams.current
        duration = session.audio.duration
        with self._lock:
            cursor = self._cursor
        if duration - cursor < _MIN_FLUSH_SECONDS:
            return
        self.transcribe_range(cursor, duration)
        with self._lock:
            if self._last_session == session.session_id:
                self._cursor = duration

    def _handle_error(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._consecutive_errors += 1
            self._last_error = message
            report = message != self._reported_error
            self._reported_error = message
            backoff = min(10.0, 0.5 * (2 ** min(self._consecutive_errors, 5)))
        if report:
            self._logger.warning("STT request failed: %s", message)
            ctx = self._ctx
            if ctx is not None:
                ctx.events.emit(
                    "stt.error",
                    session_id=ctx.streams.current.session_id,
                    error=message,
                )
                ctx.ui.refresh("stt.error")
        self._stop_event.wait(backoff)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_session_started(self, event: Any) -> None:
        with self._lock:
            sid = str(event.get("session_id", "") or "")
            if sid:
                self._last_session = sid
            self._cursor = 0.0
            self._segments = 0
            self._chars = 0
            self._last_error = ""
            self._reported_error = ""
            self._consecutive_errors = 0

    def _on_session_stopped(self, _event: Any) -> None:
        try:
            self._flush_remaining()
        except Exception:  # noqa: BLE001 - session teardown is best effort
            self._logger.exception("STT flush on session stop failed")

    def _on_settings_changed(self, event: Any) -> None:
        if event.get("key") != "stt.enabled":
            return
        if bool(event.get("new")):
            with self._lock:
                self._last_error = ""
                self._reported_error = ""
                self._consecutive_errors = 0
            self._start_worker()
        else:
            self._stop_worker()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self, ctx: PluginContext) -> None:
        ui = ctx.ui
        # Debug view only: no sidebar surface, opened from the command palette.
        ui.add_page(
            PAGE_ID,
            "语音转写",
            render=self._render_page,
            icon="mic",
            order=20,
            plugin_id=self.id,
        )
        self._page_added = True
        ui.register_command(
            "stt.debug",
            "语音转写（调试）",
            subtitle="查看实时转写片段与 STT 状态",
            category="调试",
            icon="🎙",
            order=20,
            page_id=PAGE_ID,
            keywords="stt transcript debug 转写",
            plugin_id=self.id,
        )
        self._command_added = True

    def _render_page(self) -> list[dict[str, Any]]:
        ctx = self._ctx
        blocks: list[dict[str, Any]] = [
            {"type": "keyvalue", "items": self._status_rows()},
        ]
        if ctx is not None:
            transcript = ctx.streams.current.transcript
            items = [
                item
                for item in transcript.items_since(max(0, len(transcript) - 200))
                if item.source == "stt"
            ][-_MAX_PARAGRAPHS:]
            for item in items:
                stamp = ""
                if item.start is not None and item.end is not None:
                    stamp = (
                        f"**{_fmt_time(item.start)}–{_fmt_time(item.end)}**\n\n"
                    )
                blocks.append({"type": "markdown", "text": f"{stamp}{item.text}"})
            if not items:
                blocks.append({"type": "text", "text": "暂无转写结果"})
        blocks.append(
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "STT 设置",
                        "action": "ui.open_page",
                        "value": {"page_id": "settings:main"},
                        "class": "btn btn-sm",
                    }
                ],
            }
        )
        return blocks

    def _status_rows(self) -> list[list[str]]:
        status = self.status()
        return [
            ["状态", _STATE_LABELS.get(str(status["state"]), str(status["state"]))],
            ["后端", "Google Cloud Speech-to-Text v2"],
            ["模型", f"{status['model']} @ {status['location']}"],
            ["会话", status["session_id"] or "—"],
            ["音频", f"{float(status['audio_seconds']):.1f}s"],
            [
                "已转写",
                f"{status['segments']} 段 / {status['chars']} 字",
            ],
        ]


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


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
