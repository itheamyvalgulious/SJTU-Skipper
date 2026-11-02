"""Media sources: FFmpeg-backed audio/video ingestion with lifecycle."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from .ffmpeg import (
    build_audio_cmd,
    build_reconnect_args,
    build_video_cmd,
    has_complete_frame,
    split_jpeg_frames,
)

_log = logging.getLogger("lecture_agent.plugins.input.sources")

Kind = Literal["live", "recording", "url", "file"]


@dataclass(slots=True)
class SourceSpec:
    """A resolved media source ready for playback."""

    kind: Kind = "url"
    url: str = ""
    title: str = ""
    course: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    start_at: float = 0.0


# ---------------------------------------------------------------------------
# MediaSource protocol
# ---------------------------------------------------------------------------


class MediaSource(Protocol):
    """A running media source feeding audio/video to stream hub."""

    spec: SourceSpec

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    @property
    def running(self) -> bool: ...

    @property
    def paused(self) -> bool: ...

    @property
    def started_at(self) -> float: ...

    def status(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# FFmpegSource
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK = 4096
_AUDIO_RECONNECT_DELAY = 3.0  # seconds between reconnect attempts


class FFmpegSource:
    """Ingest a URL/file via two ffmpeg subprocesses → audio + video streams.

    * Audio subprocess: s16le 16kHz mono → ``ctx.streams.current.audio``
    * Video subprocess: MJPEG frames → ``ctx.streams.current.video``

    Both subprocesses are read in background threads.  JPEG frames are
    split on SOI/EOI boundaries.
    """

    def __init__(
        self,
        spec: SourceSpec,
        *,
        hub: Any,  # StreamHub
        emit: Callable[..., Any],
        sample_rate: int = 16000,
        video_fps: float = 4.0,
        video_width: int = 0,
        playback_speed: float = 1.0,
        reconnect: bool = True,
        reconnect_max_retries: int = 10,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        buffer_size: int = 65536,
    ) -> None:
        self.spec = spec
        self._hub = hub
        self._emit = emit
        self._sample_rate = int(sample_rate)
        self._video_fps = float(video_fps)
        self._video_width = int(video_width)
        self._playback_speed = float(playback_speed)
        self._reconnect = bool(reconnect)
        self._reconnect_max_retries = int(reconnect_max_retries)
        self._reconnect_base_delay = float(reconnect_base_delay)
        self._reconnect_max_delay = float(reconnect_max_delay)
        self._buffer_size = int(buffer_size)

        self._audio_proc: subprocess.Popen[bytes] | None = None
        self._video_proc: subprocess.Popen[bytes] | None = None
        self._audio_thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None
        self._audio_stop = threading.Event()
        self._video_stop = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially
        self._lock = threading.RLock()
        self._started_at: float = 0.0
        self._running = False
        self._paused = False

        # JPEG frame buffer for incomplete tail bytes
        self._jpeg_buffer = b""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def started_at(self) -> float:
        return self._started_at

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._paused = False
            self._started_at = time.time()
            self._pause_event.set()
            self._audio_stop.clear()
            self._video_stop.clear()
            self._jpeg_buffer = b""

        is_live = self.spec.kind == "live"
        is_http = self.spec.url.startswith(("http://", "https://"))
        headers = dict(self.spec.headers) if is_http else {}
        if is_http:
            headers.setdefault("User-Agent", "sjtu-skipper/0.1")

        audio_cmd = build_audio_cmd(
            self.spec.url,
            sample_rate=self._sample_rate,
            headers=headers,
        )
        if not is_live:
            # VOD: use -re for real-time pacing
            audio_cmd.insert(1, "-re")
        # Playback speed
        if self._playback_speed != 1.0:
            pos = audio_cmd.index("-i") + 2 if "-i" in audio_cmd else len(audio_cmd)
            audio_cmd.insert(pos, f"atempo={self._playback_speed}")
            audio_cmd.insert(pos, "-filter:a")
        if self._reconnect:
            pos = 1
            for arg in reversed(build_reconnect_args(self._reconnect_max_retries)):
                audio_cmd.insert(pos, arg)

        video_cmd = build_video_cmd(
            self.spec.url,
            fps=self._video_fps,
            width=self._video_width,
            headers=headers,
            live=is_live,
        )
        if not is_live:
            video_cmd.insert(1, "-re")
        if self._reconnect:
            pos = 1
            for arg in reversed(build_reconnect_args(self._reconnect_max_retries)):
                video_cmd.insert(pos, arg)

        _log.info("Starting ffmpeg audio: %s", " ".join(audio_cmd))
        _log.info("Starting ffmpeg video: %s", " ".join(video_cmd))

        try:
            self._audio_proc = subprocess.Popen(
                audio_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self._running = False
            self._emit("input.error", message="ffmpeg 未安装或不在 PATH 中", source=self.spec.title)
            return
        except Exception as exc:
            self._running = False
            self._emit("input.error", message=f"启动音频进程失败：{exc}", source=self.spec.title)
            return

        try:
            self._video_proc = subprocess.Popen(
                video_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            self._running = False
            if self._audio_proc:
                self._audio_proc.terminate()
                self._audio_proc = None
            self._emit("input.error", message=f"启动视频进程失败：{exc}", source=self.spec.title)
            return

        self._audio_thread = threading.Thread(
            target=self._read_audio,
            name=f"input-audio-{self.spec.title}",
            daemon=True,
        )
        self._video_thread = threading.Thread(
            target=self._read_video,
            name=f"input-video-{self.spec.title}",
            daemon=True,
        )
        self._audio_thread.start()
        self._video_thread.start()

        self._emit(
            "input.status",
            status="playing",
            source=self.spec.title,
            kind=self.spec.kind,
            meta=dict(self.spec.meta),
        )

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        self._audio_stop.set()
        self._video_stop.set()
        self._pause_event.set()  # unpause so threads can exit

        for proc, label in [
            (self._audio_proc, "audio"),
            (self._video_proc, "video"),
        ]:
            if proc is None:
                continue
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            except Exception:
                pass

        self._audio_proc = None
        self._video_proc = None
        self._emit(
            "input.status",
            status="stopped",
            source=self.spec.title,
        )

    def pause(self) -> None:
        with self._lock:
            if not self._running or self._paused:
                return
            self._paused = True
            self._pause_event.clear()
        self._emit("input.status", status="paused", source=self.spec.title)

    def resume(self) -> None:
        with self._lock:
            if not self._running or not self._paused:
                return
            self._paused = False
            self._pause_event.set()
        self._emit("input.status", status="playing", source=self.spec.title)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "paused": self._paused,
                "kind": self.spec.kind,
                "title": self.spec.title,
                "url": self.spec.url,
            }

    # ------------------------------------------------------------------
    # Reader threads
    # ------------------------------------------------------------------

    def _read_audio(self) -> None:
        """Read PCM from stdout → append to audio stream.  Reconnect on EOF."""
        retries = 0
        while not self._audio_stop.is_set():
            proc = self._audio_proc
            if proc is None or proc.stdout is None:
                break

            try:
                # read1 returns whatever is currently available instead of
                # blocking until the full buffer is filled (real-time audio).
                read = getattr(proc.stdout, "read1", proc.stdout.read)
                chunk = read(self._buffer_size)
            except Exception:
                chunk = None

            if chunk:
                retries = 0
                self._pause_event.wait()
                if self._audio_stop.is_set():
                    break
                try:
                    current = self._hub.current
                    current.audio.append(chunk)
                except Exception:
                    _log.exception("audio append failed")
            else:
                # EOF – process exited
                rc = proc.poll()
                if rc is not None:
                    _log.warning("Audio ffmpeg exited with code %s", rc)
                    if not self._reconnect or self._audio_stop.is_set():
                        break
                    retries += 1
                    if retries > self._reconnect_max_retries:
                        _log.error("Audio reconnect max retries exceeded")
                        self._emit(
                            "input.error",
                            message="音频流重连失败，已达最大重试次数。",
                            source=self.spec.title,
                        )
                        break
                    delay = min(
                        self._reconnect_base_delay * (2 ** (retries - 1)),
                        self._reconnect_max_delay,
                    )
                    _log.info("Audio reconnecting in %.1fs (attempt %d)", delay, retries)
                    time.sleep(delay)
                    if self._audio_stop.is_set():
                        break
                    self._restart_audio_process()
                else:
                    # process still alive but no data yet? small sleep
                    time.sleep(0.1)

        _log.info("Audio reader thread exiting")

    def _restart_audio_process(self) -> None:
        """Restart the audio ffmpeg subprocess."""
        if self._audio_proc is not None:
            try:
                self._audio_proc.terminate()
            except Exception:
                pass
        is_live = self.spec.kind == "live"
        headers = dict(self.spec.headers) if self.spec.url.startswith(("http://", "https://")) else {}
        cmd = build_audio_cmd(
            self.spec.url,
            sample_rate=self._sample_rate,
            headers=headers,
        )
        if not is_live:
            cmd.insert(1, "-re")
        if self._reconnect:
            pos = 1
            for arg in reversed(build_reconnect_args(self._reconnect_max_retries)):
                cmd.insert(pos, arg)
        try:
            self._audio_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            _log.exception("Failed to restart audio process")
            self._audio_proc = None

    def _read_video(self) -> None:
        """Read JPEG frames from stdout → split → append to video stream."""
        buffer = b""
        retries = 0
        while not self._video_stop.is_set():
            proc = self._video_proc
            if proc is None or proc.stdout is None:
                break

            try:
                # read1 returns whatever is currently available instead of
                # blocking until the full buffer is filled (real-time audio).
                read = getattr(proc.stdout, "read1", proc.stdout.read)
                chunk = read(self._buffer_size)
            except Exception:
                chunk = None

            if chunk:
                retries = 0
                buffer += chunk
                # Extract complete JPEG frames
                frames = split_jpeg_frames(buffer)
                if frames:
                    # Keep any trailing data (incomplete frame)
                    last_soi = buffer.rfind(b"\xff\xd8")
                    if last_soi >= 0:
                        last_eoi = buffer.rfind(b"\xff\xd9", last_soi)
                        if last_eoi < 0:
                            # Trailing incomplete: keep from last SOI
                            buffer = buffer[last_soi:]
                        else:
                            buffer = b""
                    else:
                        buffer = b""

                    self._pause_event.wait()
                    if self._video_stop.is_set():
                        break
                    for frame_data in frames:
                        try:
                            current = self._hub.current
                            current.video.append_frame(frame_data)
                        except Exception:
                            _log.exception("video frame append failed")
            else:
                rc = proc.poll()
                if rc is not None:
                    _log.warning("Video ffmpeg exited with code %s", rc)
                    if not self._reconnect or self._video_stop.is_set():
                        break
                    retries += 1
                    if retries > self._reconnect_max_retries:
                        _log.error("Video reconnect max retries exceeded")
                        self._emit(
                            "input.error",
                            message="视频流重连失败，已达最大重试次数。",
                            source=self.spec.title,
                        )
                        break
                    delay = min(
                        self._reconnect_base_delay * (2 ** (retries - 1)),
                        self._reconnect_max_delay,
                    )
                    time.sleep(delay)
                    if self._video_stop.is_set():
                        break
                    self._restart_video_process()
                else:
                    time.sleep(0.1)

        _log.info("Video reader thread exiting")

    def _restart_video_process(self) -> None:
        if self._video_proc is not None:
            try:
                self._video_proc.terminate()
            except Exception:
                pass
        is_live = self.spec.kind == "live"
        headers = dict(self.spec.headers) if self.spec.url.startswith(("http://", "https://")) else {}
        cmd = build_video_cmd(
            self.spec.url,
            fps=self._video_fps,
            width=self._video_width,
            headers=headers,
            live=is_live,
        )
        if not is_live:
            cmd.insert(1, "-re")
        if self._reconnect:
            pos = 1
            for arg in reversed(build_reconnect_args(self._reconnect_max_retries)):
                cmd.insert(pos, arg)
        try:
            self._video_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            _log.exception("Failed to restart video process")
            self._video_proc = None


# ---------------------------------------------------------------------------
# SourceManager
# ---------------------------------------------------------------------------


class SourceManager:
    """At most one active source at a time.  Owns StreamHub session lifecycle."""

    def __init__(
        self,
        *,
        hub: Any,
        emit: Callable[..., Any],
        sample_rate: int = 16000,
        video_fps: float = 4.0,
        video_width: int = 0,
        playback_speed: float = 1.0,
        reconnect: bool = True,
    ) -> None:
        self._hub = hub
        self._emit = emit
        self._sample_rate = int(sample_rate)
        self._video_fps = float(video_fps)
        self._video_width = int(video_width)
        self._playback_speed = float(playback_speed)
        self._reconnect = bool(reconnect)
        self._lock = threading.RLock()
        self._current: FFmpegSource | None = None
        self._state: Literal["idle", "playing", "paused", "stopped"] = "idle"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def start(self, spec: SourceSpec) -> None:
        """Start a new source, stopping any existing one."""
        with self._lock:
            self.stop()
            meta = dict(spec.meta)
            meta.update(spec.course)
            meta.setdefault("title", spec.title)
            meta.setdefault("kind", spec.kind)
            # Keep the stream timeline metadata in sync with the source.
            try:
                self._hub.video_fps = self._video_fps
                self._hub.video_width = self._video_width or 1920
                self._hub.video_height = 1080
                self._hub.audio_sample_rate = self._sample_rate
            except Exception:
                pass
            self._hub.start_session(meta=meta)

            source = FFmpegSource(
                spec,
                hub=self._hub,
                emit=self._emit,
                sample_rate=self._sample_rate,
                video_fps=self._video_fps,
                video_width=self._video_width,
                playback_speed=self._playback_speed,
                reconnect=self._reconnect,
            )
            self._current = source
            self._state = "playing"
        source.start()

    def stop(self) -> None:
        with self._lock:
            source = self._current
            self._current = None
            self._state = "idle"
        if source is not None:
            source.stop()
            current = self._hub.current
            if current.session_id != "boot" and current.ended_at is None:
                self._hub.end_session("source stopped")

    def pause(self) -> None:
        with self._lock:
            source = self._current
        if source is not None:
            source.pause()
            with self._lock:
                self._state = "paused"

    def resume(self) -> None:
        with self._lock:
            source = self._current
        if source is not None:
            source.resume()
            with self._lock:
                self._state = "playing"

    def status(self) -> dict[str, Any]:
        with self._lock:
            source = self._current
            return {
                "state": self._state,
                "source": source.status() if source is not None else None,
            }


# ---------------------------------------------------------------------------
# Fake source for tests
# ---------------------------------------------------------------------------


class FakeSource:
    """A source that appends fake audio/video data without ffmpeg."""

    def __init__(self, spec: SourceSpec, *, hub: Any, emit: Callable[..., Any]):
        self.spec = spec
        self._hub = hub
        self._emit = emit
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._running = False
        self._started_at: float = 0.0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return not self._pause.is_set()

    @property
    def started_at(self) -> float:
        return self._started_at

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_at = time.time()
        self._pause.set()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"fake-input-{self.spec.title}",
            daemon=True,
        )
        self._thread.start()
        self._emit("input.status", status="playing", source=self.spec.title)

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        self._pause.set()
        self._emit("input.status", status="stopped", source=self.spec.title)

    def pause(self) -> None:
        self._pause.clear()
        self._emit("input.status", status="paused", source=self.spec.title)

    def resume(self) -> None:
        self._pause.set()
        self._emit("input.status", status="playing", source=self.spec.title)

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "paused": self.paused,
            "kind": self.spec.kind,
            "title": self.spec.title,
        }

    def _run(self) -> None:
        sample_rate = 16000
        fake_audio_chunk = b"\x00" * (sample_rate * 2)  # 1 second of silence
        fake_jpeg = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00\x43\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07"
            b"\x09\x09\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
            b"\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c\x20\x24\x2e\x27\x20\x22"
            b"\x2c\x23\x1c\x1c\x28\x37\x29\x2c\x30\x31\x34\x34\x34\x1f\x27"
            b"\x39\x3d\x38\x32\x3c\x2e\x33\x34\x32\xff\xc0\x00\x0b\x08\x00"
            b"\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05"
            b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01"
            b"\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xc4\x00\xb5\x10"
            b"\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01"
            b"\x7d\x01\x02\x03\x00\x04\x11\x05\x12\x21\x31\x41\x06\x13\x51"
            b"\x61\x07\x22\x71\x14\x32\x81\x91\xa1\x08\x23\x42\xb1\xc1\x15"
            b"\x52\xd1\xf0\x24\x33\x62\x72\x82\x09\x0a\x16\x17\x18\x19\x1a"
            b"\x25\x26\x27\x28\x29\x2a\x34\x35\x36\x37\x38\x39\x3a\x43\x44"
            b"\x45\x46\x47\x48\x49\x4a\x53\x54\x55\x56\x57\x58\x59\x5a\x63"
            b"\x64\x65\x66\x67\x68\x69\x6a\x73\x74\x75\x76\x77\x78\x79\x7a"
            b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98"
            b"\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5"
            b"\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2"
            b"\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7"
            b"\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda"
            b"\x00\x08\x01\x01\x00\x00\x3f\x00\xd2\xff\xd9"
        )
        while not self._stop.is_set():
            self._pause.wait()
            if self._stop.is_set():
                break
            try:
                current = self._hub.current
                current.audio.append(fake_audio_chunk)
                current.video.append_frame(fake_jpeg)
            except Exception:
                _log.exception("fake source append")
            time.sleep(0.05)