"""STT plugin: consume the audio stream and grow the transcript stream.

The worker keeps a per-session cursor in seconds.  It transcribes windows of
``stt.chunk_seconds`` with ``stt.overlap`` seconds of shared audio, then trims
the duplicated text against the previously appended line before appending a
``TextItem`` with ``source="stt"``.  Session switches flush the tail of the old
session and reset the cursor; a session ``ended_at`` flushes the tail too.

Failures surface as an error status (visible in the page and in ``stt.error``).
The same window is retried with exponential backoff instead of being skipped,
so a broken backend is reported loudly and no audio is silently dropped.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from core.plugin import Plugin, PluginContext
from core.streams import StreamSet
from plugins.settings.spec import SettingsSpec
from .audio import Window, dedupe_overlap, pcm_to_wav
from .backends import (
    PROVIDERS,
    STTBackend,
    STTError,
    STTResult,
    STTUnavailableError,
    create_backend,
)

LOGGER_NAME = "lecture_agent.plugin.stt"

EVENT_SEGMENT = "stt.segment"
EVENT_ERROR = "stt.error"

TAB_ID = "classroom"
ITEM_ID = "stt.transcript"
PAGE_ID = "stt:transcript"

MIN_OVERLAP_CHARS = 4


def _specs() -> tuple[SettingsSpec, ...]:
    return (
        SettingsSpec(
            key="stt.enabled",
            label="启用语音识别",
            type="bool",
            default=True,
            group="STT",
            description="关闭后不再消费音频流，已转写内容保留。",
            group_note="provider=none 时只接收音频流、不产生文字；其余参数仅在接入后端时显示。",
            order=10,
        ),
        SettingsSpec(
            key="stt.provider",
            label="STT 后端",
            type="select",
            default="none",
            group="STT",
            options=PROVIDERS,
            description="none = 只接入音频流、不输出文字；openai = 兼容接口；faster-whisper = 本地依赖。",
            order=11,
        ),
        SettingsSpec(
            key="stt.model",
            visible_when=("stt.provider", ("openai", "faster-whisper")),
            label="模型",
            type="string",
            default="whisper-1",
            group="STT",
            description="openai 后端使用 whisper-1 之类；faster-whisper 使用 base/small/medium 等。",
            order=12,
        ),
        SettingsSpec(
            key="stt.language",
            visible_when=("stt.provider", ("openai", "faster-whisper")),
            label="语言",
            type="string",
            default="zh",
            group="STT",
            description="ISO-639-1 语言代码，留空则自动检测。",
            order=13,
        ),
        SettingsSpec(
            key="stt.chunk_seconds",
            visible_when=("stt.provider", ("openai", "faster-whisper")),
            label="切片时长（秒）",
            type="float",
            default=30.0,
            group="STT",
            description="每次送入后端的音频长度。",
            order=14,
        ),
        SettingsSpec(
            key="stt.overlap",
            visible_when=("stt.provider", ("openai", "faster-whisper")),
            label="重叠时长（秒）",
            type="float",
            default=0.5,
            group="STT",
            description="相邻切片共享的音频长度，用于避免切断词语。",
            order=15,
        ),
        SettingsSpec(
            key="stt.min_seconds",
            visible_when=("stt.provider", ("openai", "faster-whisper")),
            label="最短转写时长（秒）",
            type="float",
            default=1.0,
            group="STT",
            description="会话结束或切换时，剩余音频短于该值则丢弃。",
            order=16,
        ),
        SettingsSpec(
            key="stt.base_url",
            visible_when=("stt.provider", ("openai",)),
            label="转写 Base URL",
            type="string",
            default="",
            group="STT",
            description="OpenAI 兼容转写接口的 Base URL。",
            placeholder="https://api.openai.com/v1",
            order=17,
        ),
        SettingsSpec(
            key="stt.api_key",
            visible_when=("stt.provider", ("openai",)),
            label="转写 API Key",
            type="secret",
            default="",
            group="STT",
            description="OpenAI 兼容转写接口的 API Key。",
            order=18,
        ),
    )


@dataclass(frozen=True, slots=True)
class STTConfig:
    """Snapshot of the STT settings."""

    enabled: bool = True
    provider: str = "none"
    model: str = "whisper-1"
    language: str = "zh"
    chunk_seconds: float = 30.0
    overlap: float = 0.5
    min_seconds: float = 1.0
    base_url: str = ""
    api_key: str = ""


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


class STTService:
    """Public ``stt`` service consumed by UI/other plugins.

    The service is a thin facade over the plugin so callers never need the
    worker internals: ``status()`` for the UI, ``transcribe_once()`` for
    one-off audio and ``backends()`` to list supported providers.
    """

    def __init__(self, plugin: "SttPlugin") -> None:
        self._plugin = plugin

    @property
    def plugin(self) -> "SttPlugin":
        return self._plugin

    def status(self) -> dict[str, Any]:
        """Current provider/session/worker state."""

        return self._plugin.status()

    def backends(self) -> list[str]:
        """Names accepted by ``stt.provider``."""

        return list(PROVIDERS)

    def transcribe_once(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> STTResult:
        """Transcribe a single WAV payload without touching the streams."""

        return self._plugin.transcribe_once(
            wav_bytes,
            language=language,
            model=model,
            provider=provider,
        )


class SttPlugin(Plugin):
    """Speech recognition plugin; see module docstring for behaviour."""

    id = "stt"
    requires: tuple[str, ...] = ("settings", "llm")
    provides: tuple[str, ...] = ("stt",)

    def __init__(
        self,
        *,
        auto_start: bool = True,
        backend_factory: Callable[[STTConfig], STTBackend] | None = None,
        idle_poll: float = 0.2,
        base_backoff: float = 1.0,
        max_backoff: float = 15.0,
        min_overlap_chars: int = MIN_OVERLAP_CHARS,
    ) -> None:
        self.auto_start = bool(auto_start)
        self._backend_factory = backend_factory
        self.idle_poll = max(0.05, float(idle_poll))
        self.base_backoff = max(0.0, float(base_backoff))
        self.max_backoff = max(self.base_backoff, float(max_backoff))
        self.min_overlap_chars = max(1, int(min_overlap_chars))

        self._ctx: PluginContext | None = None
        self._logger: logging.Logger = logging.getLogger(LOGGER_NAME)
        self._settings: Any | None = None

        self._flow_lock = threading.RLock()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

        self._session: StreamSet | None = None
        self._session_id: str = ""
        self._cursor: float = 0.0
        self._last_text: str = ""
        self._window_failures: int = 0
        self._consecutive_failures: int = 0
        self._next_attempt: float = 0.0
        self._state: str = "idle"
        self._last_error: str = ""
        self._segments: int = 0
        self._chars: int = 0
        self._backend: STTBackend | None = None
        self._backend_key: tuple[Any, ...] | None = None
        self._page_open = False

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------
    def setup(self, ctx: PluginContext) -> None:
        """Register settings/service/UI/events and start the worker."""

        self._ctx = ctx
        self._logger = ctx.logger
        ctx.events.define(EVENT_SEGMENT, "STT 追加了一段转写")
        ctx.events.define(EVENT_ERROR, "STT 转写失败")

        self._settings = ctx.services.get("settings")
        if self._settings is None:  # pragma: no cover - requires guarantees it
            ctx.logger.warning("settings service missing; STT uses defaults")
        else:
            for spec in _specs():
                register = getattr(self._settings, "register", None)
                if callable(register):
                    register(spec)

        ctx.register_service("stt", STTService(self))
        ctx.meta.register("stt", self.status, plugin_id=self.id)
        ctx.events.on(
            "settings.changed",
            self._on_settings_changed,
            plugin_id=self.id,
        )

        self._session = ctx.streams.current
        self._session_id = self._session.session_id
        self._build_ui()
        ctx.logger.info(
            "stt ready: provider=%s chunk=%.1fs overlap=%.1fs",
            self.config().provider,
            self.config().chunk_seconds,
            self.config().overlap,
        )
        if self.auto_start:
            self.start()

    def teardown(self) -> None:
        """Stop the worker, flush the remaining tail and drop UI entries."""

        self.stop()
        try:
            self.pump(flush=True, force=True)
        except Exception:  # noqa: BLE001 - shutdown is best effort
            self._logger.exception("STT final flush failed")
        self._close_backend()
        ctx = self._ctx
        if ctx is not None:
            try:
                ctx.ui.remove_page(PAGE_ID)
            except Exception:  # noqa: BLE001
                pass
            try:
                ctx.ui.remove_toolbox_item(ITEM_ID)
            except Exception:  # noqa: BLE001
                pass

    def _on_settings_changed(self, event: Any) -> None:
        if event.get("key") != "stt.enabled":
            return
        enabled = bool(event.get("new"))
        if enabled:
            self.start()
        else:
            self.stop()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the consumer thread (idempotent)."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name=f"stt-worker-{id(self):x}",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker and join it; safe to call twice."""

        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop.set()
        self._wake.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self._wake.clear()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.pump()
            except Exception:  # noqa: BLE001 - worker must survive anything
                self._logger.exception("STT worker iteration failed")
                self._stop.wait(0.5)
                continue
            if self._stop.is_set():
                break
            self._wake.wait(self.idle_poll)
            self._wake.clear()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        settings = self._settings
        if settings is None:
            return default
        try:
            return settings.get(key, default)
        except Exception:  # noqa: BLE001 - a broken settings service must not kill STT
            return default

    def config(self) -> STTConfig:
        """Read the current settings; no implicit fallback to another profile."""

        base_url = _as_text(self._get("stt.base_url", ""))
        api_key = _as_text(self._get("stt.api_key", ""))
        model = _as_text(self._get("stt.model", ""))
        return STTConfig(
            enabled=_as_bool(self._get("stt.enabled", True), True),
            provider=_as_text(self._get("stt.provider", "none"), "none").lower(),
            model=model or "whisper-1",
            language=_as_text(self._get("stt.language", "zh")),
            chunk_seconds=_as_float(self._get("stt.chunk_seconds", 30.0), 30.0),
            overlap=_as_float(self._get("stt.overlap", 0.5), 0.5),
            min_seconds=_as_float(self._get("stt.min_seconds", 1.0), 1.0),
            base_url=base_url,
            api_key=api_key,
        )

    def _config_error(self, cfg: STTConfig) -> str:
        if cfg.provider not in PROVIDERS:
            return f"未知 STT provider：{cfg.provider}"
        if cfg.provider == "openai":
            if not cfg.base_url:
                return "缺少 STT base_url（请在设置中填写 stt.base_url）"
            if not cfg.api_key:
                return "缺少 STT API Key（请在设置中填写 stt.api_key）"
        if cfg.chunk_seconds <= 0:
            return "stt.chunk_seconds 必须为正数"
        if cfg.overlap < 0 or cfg.overlap >= cfg.chunk_seconds:
            return "stt.overlap 必须大于等于 0 且小于 stt.chunk_seconds"
        return ""

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _backend_for(self, cfg: STTConfig) -> STTBackend:
        key = (cfg.provider, cfg.base_url, cfg.api_key, cfg.model, id(self._backend_factory))
        with self._lock:
            if self._backend is not None and self._backend_key == key:
                return self._backend
            old = self._backend
            self._backend = None
            self._backend_key = None
        if self._backend_factory is not None:
            backend = self._backend_factory(cfg)
        else:
            backend = create_backend(
                cfg.provider,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                model=cfg.model,
            )
        with self._lock:
            if self._backend is None:
                self._backend = backend
                self._backend_key = key
                replaced = old
            else:  # a concurrent call won the race; keep theirs
                replaced = backend
                backend = self._backend
        self._close(replaced)
        return backend

    def _close(self, backend: STTBackend | None) -> None:
        if backend is None:
            return
        closer = getattr(backend, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                self._logger.exception("failed to close STT backend")

    def _close_backend(self) -> None:
        with self._lock:
            backend = self._backend
            self._backend = None
            self._backend_key = None
        self._close(backend)

    def transcribe_once(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> STTResult:
        """Transcribe one WAV payload immediately (no stream side effects)."""

        cfg = self.config()
        if provider:
            cfg = replace(cfg, provider=provider.strip().lower())
        error = self._config_error(cfg)
        if error:
            raise STTUnavailableError(error)
        backend = self._backend_for(cfg)
        return backend.transcribe(
            wav_bytes,
            language=language if language is not None else (cfg.language or None),
            model=model or cfg.model or None,
        )

    # ------------------------------------------------------------------
    # Audio consumption
    # ------------------------------------------------------------------
    def pump(self, *, flush: bool = False, force: bool = False) -> int:
        """Process every complete window available right now.

        Returns the number of transcript items appended.  ``flush`` also
        transcribes the trailing partial window (used on session end/switch and
        at shutdown); ``force`` ignores the failure backoff delay.
        """

        ctx = self._ctx
        if ctx is None:
            return 0
        with self._flow_lock:
            session = ctx.streams.current
            flushed = 0
            if session.session_id != self._session_id:
                previous = self._session
                if previous is not None:
                    try:
                        flushed = self._pump_session(previous, flush=True, force=force)
                    except Exception:  # noqa: BLE001 - old session must not block the new one
                        self._logger.exception("STT flush of previous session failed")
                self._reset_for(session)
            ended = session.ended_at is not None
            return flushed + self._pump_session(session, flush=flush or ended, force=force)

    def _reset_for(self, session: StreamSet) -> None:
        with self._lock:
            self._session = session
            self._session_id = session.session_id
            self._cursor = 0.0
            self._last_text = ""
            self._window_failures = 0
            self._consecutive_failures = 0
            self._next_attempt = 0.0
            self._state = "idle"
            self._last_error = ""
            self._segments = 0
            self._chars = 0

    def _pump_session(self, session: StreamSet, *, flush: bool, force: bool) -> int:
        cfg = self.config()
        if not cfg.enabled:
            self._set_state("disabled", "")
            return 0
        if cfg.provider == "none":
            # Audio is connected, but no transcription backend is attached:
            # deliberately produce no text and do not move the cursor, so a
            # later backend can transcribe the whole session.
            self._set_state("idle", "")
            return 0
        error = self._config_error(cfg)
        if error:
            self._set_state("error", error, emit=True)
            return 0
        if not force and time.monotonic() < self._next_attempt:
            return 0

        chunk = max(0.05, cfg.chunk_seconds)
        overlap = min(max(0.0, cfg.overlap), chunk * 0.5)
        min_seconds = max(0.0, cfg.min_seconds)
        duration = session.audio.duration
        appended = 0

        while True:
            with self._lock:
                cursor = self._cursor
            remaining = duration - cursor
            if remaining <= 1e-9:
                break
            if flush:
                if remaining < min_seconds - 1e-9:
                    break
                end = min(cursor + chunk, duration)
            else:
                if remaining < chunk - 1e-9:
                    break
                end = cursor + chunk
            window = Window(start=cursor, end=end)
            outcome = self._process_window(session, window, cfg)

            if outcome == "failed":
                # No silent skip: keep the cursor and surface the error.  The
                # next pump retries the same window after the backoff delay.
                with self._lock:
                    self._window_failures += 1
                break

            if outcome == "empty":
                break
            if outcome == "appended":
                appended += 1
            with self._lock:
                self._window_failures = 0
                self._cursor = end if end >= duration - 1e-9 else end - overlap
                if self._cursor <= cursor + 1e-9 and end < duration - 1e-9:
                    self._cursor = end  # pragma: no cover - degenerate overlap
        return appended

    def _process_window(
        self,
        session: StreamSet,
        window: Window,
        cfg: STTConfig,
    ) -> str:
        """Transcribe one window; returns appended/skipped/empty/failed."""

        pcm = session.audio.read(window.start, window.end)
        if not pcm:
            return "empty"
        try:
            backend = self._backend_for(cfg)
            wav = pcm_to_wav(
                pcm,
                sample_rate=session.audio.sample_rate,
                channels=session.audio.channels,
                sample_width=session.audio.sample_width,
            )
            result = backend.transcribe(
                wav,
                language=cfg.language or None,
                model=cfg.model or None,
            )
        except STTUnavailableError as exc:
            self._fail(str(exc), cfg)
            return "failed"
        except STTError as exc:
            self._fail(str(exc), cfg)
            return "failed"
        except Exception as exc:  # noqa: BLE001 - backend code is untrusted
            self._fail(f"{type(exc).__name__}: {exc}", cfg)
            return "failed"

        text = (result.text or "").strip()
        with self._lock:
            previous = self._last_text
        tail = dedupe_overlap(previous, text, min_overlap=self.min_overlap_chars) if text else ""
        if not tail:
            with self._lock:
                if text:
                    self._last_text = text
                self._state = "running"
                self._last_error = ""
                self._consecutive_failures = 0
            return "skipped"

        item = session.transcript.append(
            tail,
            start=window.start,
            end=window.end,
            source="stt",
            meta={
                "model": cfg.model,
                "language": cfg.language,
                "provider": cfg.provider,
                "segments": len(result.segments),
            },
        )
        with self._lock:
            self._last_text = text
            self._state = "running"
            self._last_error = ""
            self._consecutive_failures = 0
            self._next_attempt = 0.0
            self._segments += 1
            self._chars += len(tail)
            segments_total = self._segments
        self._emit(
            EVENT_SEGMENT,
            session_id=session.session_id,
            text=tail,
            start=window.start,
            end=window.end,
            provider=cfg.provider,
            model=cfg.model,
            language=cfg.language,
            chars=len(tail),
            item_index=item.index if item is not None else None,
            segments_total=segments_total,
        )
        ctx = self._ctx
        if ctx is not None:
            ctx.ui.refresh("stt.segment")
        return "appended"

    # ------------------------------------------------------------------
    # Errors and status
    # ------------------------------------------------------------------
    def _fail(self, message: str, cfg: STTConfig) -> None:
        with self._lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
            delay = min(self.max_backoff, self.base_backoff * (2 ** (failures - 1)))
            self._next_attempt = time.monotonic() + delay
            self._state = "error"
            self._last_error = message
        self._logger.warning("STT window failed (%d): %s", failures, message)
        self._emit(
            EVENT_ERROR,
            error=message,
            provider=cfg.provider,
            model=cfg.model,
            failures=failures,
            session_id=self._session_id,
            retry_in=delay,
        )

    def _set_state(self, state: str, error: str, *, emit: bool = False) -> None:
        with self._lock:
            changed = (self._state, self._last_error) != (state, error)
            self._state = state
            self._last_error = error
        if emit and changed:
            self._emit(
                EVENT_ERROR,
                error=error,
                provider=self.config().provider,
                session_id=self._session_id,
                failures=self._consecutive_failures,
            )

    def _emit(self, event_type: str, **payload: Any) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        try:
            ctx.events.emit(event_type, **payload)
        except Exception:  # noqa: BLE001 - observability must not break STT
            self._logger.exception("failed to emit %s", event_type)

    def status(self) -> dict[str, Any]:
        """Snapshot for the UI, the service and the meta registry."""

        cfg = self.config()
        session = self._ctx.streams.current if self._ctx is not None else None
        with self._lock:
            state = self._state
            error = self._last_error
            cursor = self._cursor
            segments = self._segments
            chars = self._chars
            last_text = self._last_text
            failures = self._consecutive_failures
            session_id = self._session_id
        if not cfg.enabled:
            state, error = "disabled", ""
        else:
            config_error = self._config_error(cfg)
            if config_error:
                state, error = "error", config_error
        return {
            "enabled": cfg.enabled,
            "state": state,
            "error": error,
            "provider": cfg.provider,
            "model": cfg.model,
            "language": cfg.language,
            "chunk_seconds": cfg.chunk_seconds,
            "overlap": cfg.overlap,
            "min_seconds": cfg.min_seconds,
            "has_base_url": bool(cfg.base_url),
            "has_api_key": bool(cfg.api_key),
            "session_id": session_id or (session.session_id if session else ""),
            "cursor": round(cursor, 3),
            "audio_seconds": round(session.audio.duration, 3) if session else 0.0,
            "segments": segments,
            "chars": chars,
            "last_text": last_text,
            "failures": failures,
            "worker": self.running,
            "backends": list(PROVIDERS),
        }

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        ui = ctx.ui
        tabs = getattr(ui, "snapshot", lambda: {})().get("tabs", {})
        if TAB_ID not in tabs:
            try:
                ui.add_tab(
                    "left",
                    TAB_ID,
                    "课堂",
                    icon="🎙",
                    order=10,
                    plugin_id=self.id,
                )
            except ValueError:
                pass  # another plugin created it concurrently
        ui.add_toolbox_item(
            TAB_ID,
            ITEM_ID,
            "转写",
            description="实时语音转写文字流",
            order=20,
            render=self._render_item,
            plugin_id=self.id,
        )
        ui.add_page(
            PAGE_ID,
            "实时转写",
            render=self._render_page,
            icon="mic",
            order=20,
            plugin_id=self.id,
        )

    def _state_label(self, status: dict[str, Any]) -> str:
        if not status["enabled"]:
            return "已关闭"
        if status.get("provider") == "none":
            return "未接入 STT 后端（仅接收音频流）"
        return {
            "idle": "等待音频",
            "running": "转写中",
            "error": "错误",
            "disabled": "已关闭",
        }.get(status["state"], status["state"])

    def _recent_text(self, limit: int = 40) -> str:
        ctx = self._ctx
        if ctx is None:
            return ""
        try:
            items = ctx.streams.current.transcript.items_since(0)
        except Exception:  # noqa: BLE001 - render must never raise
            return ""
        texts = [item.text for item in items if item.source == "stt"][-limit:]
        return "\n".join(texts)

    def _status_rows(self, status: dict[str, Any]) -> list[list[str]]:
        rows = [
            ["状态", self._state_label(status)],
            ["后端", "未接入" if status["provider"] == "none" else status["provider"]],
            ["模型", status["model"] or "-"],
            ["语言", status["language"] or "自动"],
            ["会话", status["session_id"] or "-"],
            ["已转写", f"{status['segments']} 段 / {status['chars']} 字"],
        ]
        if status["error"]:
            rows.append(["错误", status["error"]])
        return rows

    def _render_item(self) -> list[dict[str, Any]]:
        status = self.status()
        blocks: list[dict[str, Any]] = [
            {"type": "keyvalue", "items": self._status_rows(status)},
        ]
        text = self._recent_text(limit=2)
        if text:
            blocks.append({"type": "text", "text": text})
        blocks.append(
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "打开转写页",
                        "action": "ui.open_page",
                        "value": {"page_id": PAGE_ID},
                        "class": "btn btn-sm btn-primary",
                    },
                    {
                        "label": "STT 设置",
                        "action": "ui.open_page",
                        "value": {"page_id": "settings:main"},
                        "class": "btn btn-sm",
                    },
                ],
            }
        )
        return blocks

    def _render_page(self) -> list[dict[str, Any]]:
        status = self.status()
        duration = float(status["audio_seconds"] or 0.0)
        cursor = float(status["cursor"] or 0.0)
        progress = 0.0 if duration <= 0 else min(1.0, max(0.0, cursor / duration))
        blocks: list[dict[str, Any]] = [
            {"type": "keyvalue", "items": self._status_rows(status)},
            {
                "type": "progress",
                "value": progress,
                "label": f"已消费 {cursor:.1f}s / {duration:.1f}s",
            },
        ]
        if status["error"]:
            blocks.append(
                {"type": "html", "html": f'<div class="ui-error">⚠️ {_escape(status["error"])}</div>'}
            )
        body = self._recent_text(limit=80) or "（等待音频…）"
        blocks.append({"type": "markdown", "text": f"### 实时转写\n\n{body}"})
        return blocks


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Friendly alias for plugin registries that use upper-case acronyms.
STTPlugin = SttPlugin

__all__ = [
    "EVENT_ERROR",
    "EVENT_SEGMENT",
    "ITEM_ID",
    "PAGE_ID",
    "STTConfig",
    "STTPlugin",
    "STTService",
    "SttPlugin",
    "TAB_ID",
]
