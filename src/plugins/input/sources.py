from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from enum import StrEnum
from typing import Any, Callable, Protocol

from core.lecture import source_id
from core.media import MediaKind, ProcessingMode, SourceKind
from urllib.parse import urlsplit

from .download import MediaDownloader
from .ffmpeg import (
    build_audio_cmd,
    build_reconnect_args,
    build_video_cmd,
    has_complete_frame,
    split_jpeg_frames,
    probe_duration,
)

_log = logging.getLogger("lecture_agent.plugins.input.sources")


class SourceState(StrEnum):
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


class DownloadState(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    DONE = "done"


@dataclass(slots=True)
class SourceSpec:
    """A resolved media source ready for playback."""

    kind: SourceKind = SourceKind.URL
    url: str = ""
    title: str = ""
    course: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    start_at: float = 0.0
    # Keep appending to the current stream session (used for the next period
    # of a merged class so one class yields one transcript / one note set).
    continue_session: bool = False
    # ``False`` disables ffmpeg's ``-re`` pacing: the recording is ingested as
    # fast as the decoder allows (live sources are always realtime).
    realtime: bool = True
    # Save the recording to the session's ``media/`` directory before decoding
    # it.  Ignored for live sources and local files.
    download: bool = False
    # Backlog limits (media seconds) applied to the session's stream gates.
    audio_gate_limit: float = 0.0
    video_gate_limit: float = 0.0
    # Prefetch only the next file. Consumer queues continue across part EOF.
    next_source: SourceSpec | None = None
    final_part: bool = True


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
        hub: Any,
        emit: Callable[..., Any],
        sample_rate: int = 16000,
        video_fps: float = 4.0,
        video_width: int = 0,
        reconnect: bool = True,
        reconnect_max_retries: int = 10,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        buffer_size: int = 65536,
        realtime: bool = True,
        downloader: MediaDownloader | None = None,
        media_dir: Path | None = None,
        audio_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self.spec = spec
        self._hub = hub
        self._emit = emit
        self._sample_rate = int(sample_rate)
        self._video_fps = float(video_fps)
        self._video_width = int(video_width)
        self._reconnect = bool(reconnect)
        self._reconnect_max_retries = int(reconnect_max_retries)
        self._reconnect_base_delay = float(reconnect_base_delay)
        self._reconnect_max_delay = float(reconnect_max_delay)
        self._buffer_size = int(buffer_size)
        self._realtime = bool(realtime) or spec.kind == SourceKind.LIVE
        self._downloader = downloader
        if downloader is not None and spec.headers:
            self._downloader = downloader.for_headers(spec.headers)
        self._media_dir = Path(media_dir) if media_dir is not None else None
        self._input_url = spec.url
        self._media_path: Path | None = None
        self._download_handle = None
        self._download_thread = None
        self._download_done = threading.Event()
        self._download_error = ""
        self._audio_enabled = audio_enabled
        self._audio_done = False
        self._video_done = False
        self._session_id = ""
        self._prepared = False
        self._prepare_done: threading.Event | None = None
        self._prepare_thread = None
        self._duration = 0.0
        self._cancelled = False
        self._audio_offset = 0.0
        self._video_offset = 0.0

        self._audio_proc: subprocess.Popen[bytes] | None = None
        self._video_proc: subprocess.Popen[bytes] | None = None
        self._audio_thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None
        self._audio_stop = threading.Event()
        self._video_stop = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._lock = threading.RLock()
        self._started_at: float = 0.0
        self._running = False
        self._paused = False
        self._ended_emitted = False

        self._jpeg_buffer = b""

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

    @property
    def finished(self) -> bool:
        """True once both readers reached EOF on a recording."""

        with self._lock:
            return self._ended_emitted

    def _prepare_input(self) -> str:
        """Return a seekable cache URL for remote recordings."""

        url = self.spec.url
        if (
            not self.spec.download
            or self.spec.kind == SourceKind.LIVE
            or self._downloader is None
            or self._media_dir is None
            or not url.startswith(("http://", "https://"))
        ):
            return url

        name = self._media_name(url)
        video_id = str(self.spec.course.get("video_id") or "")
        if video_id:
            name = f"{video_id}_{name}"
        dest = self._media_dir / name
        self._emit(
            "input.download",
            status=DownloadState.STARTED,
            source=self.spec.title,
            session_id=self._session_id,
            video_id=video_id,
            total=0,
        )

        def _progress(received: int, total: int) -> None:
            self._emit(
                "input.download",
                status=DownloadState.PROGRESS,
                source=self.spec.title,
                session_id=self._session_id,
                video_id=video_id,
                received=int(received),
                total=int(total),
            )

        self._download_handle = self._downloader.start(
            url,
            dest,
            stop_event=self._audio_stop,
            on_progress=_progress,
        )
        self._download_handle.set_prefetch(self._prepare_done is not None)
        if self._cancelled:
            self._download_handle.close()
            raise OSError("cancelled")
        return self._download_handle.input_url

    def _monitor_download(self) -> None:
        handle = self._download_handle
        assert handle is not None
        try:
            result = handle.wait()
            if not result.ok or result.path is None:
                if self._cancelled:
                    return
                self._download_error = result.error
                self._emit(
                    "input.error",
                    session_id=self._session_id,
                    message=f"录像下载失败：{result.error}",
                    source=self.spec.title,
                )
                self.stop()
                return
            self._media_path = result.path
            session = self._hub.get_session(self._session_id)
            if session is not None:
                with self._lock:
                    session.meta["media_path"] = str(result.path)
                    session.meta["media_name"] = result.path.name
                    parts = session.meta.setdefault("media_parts", [])
                    if not any(part["path"] == str(result.path) for part in parts):
                        parts.append(
                            {
                                "path": str(result.path),
                                "video_id": str(self.spec.course.get("video_id") or ""),
                                "offset": self._audio_offset,
                                "duration": self._duration,
                                "recording_begin": self.spec.meta.get(
                                    "recording_begin", ""
                                ),
                                "recording_end": self.spec.meta.get(
                                    "recording_end", ""
                                ),
                            }
                        )
            self._emit(
                "input.download",
                status=DownloadState.DONE,
                session_id=self._session_id,
                video_id=str(self.spec.course.get("video_id") or ""),
                source=self.spec.title,
                received=result.total,
                total=result.total,
                path=str(result.path),
            )
        finally:
            self._download_done.set()

    def prepare(self) -> None:
        """Download one part and report its measured duration before decoding."""
        self._input_url = self._prepare_input()
        try:
            self._duration = (
                probe_duration(self._input_url)
                if self.spec.kind != SourceKind.LIVE
                else 0.0
            )
        except Exception:
            if self._download_handle is not None:
                self._download_handle.close()
            raise
        self._prepared = True
        self._emit(
            "input.prepared",
            video_id=str(self.spec.course.get("video_id") or ""),
            duration=self._duration,
            session_id=self._session_id,
            path=str(self._media_path) if self._media_path else "",
        )

    def _media_name(self, url: str) -> str:
        raw = ""
        try:
            raw = Path(urlsplit(url).path).name
        except Exception:  # noqa: BLE001 - fall back to the class id
            raw = ""
        if not raw or not raw.lower().endswith((".mp4", ".flv", ".m4a", ".mkv")):
            video_id = str(self.spec.course.get("video_id") or "")
            raw = f"{video_id or 'recording'}.mp4"
        return raw

    def start(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            if self._running:
                return
            self._running = True
            self._paused = False
            self._started_at = time.time()
            self._pause_event.set()
            self._audio_stop.clear()
            self._video_stop.clear()
            self._jpeg_buffer = b""
            self._audio_done = False
            self._video_done = False
            self._ended_emitted = False

        is_live = self.spec.kind == SourceKind.LIVE
        with self._lock:
            self._session_id = self._hub.current.session_id
        session = self._hub.get_session(self._session_id) or self._hub.current
        self._audio_offset = session.audio.duration
        self._video_offset = session.video.duration
        if self._prepare_done is not None:
            self._prepare_done.wait()
        try:
            if not self._prepared:
                self.prepare()
        except Exception as exc:
            self._emit(
                "input.error",
                session_id=self._session_id,
                message=f"准备录像失败：{exc}",
                source=self.spec.title,
            )
            self.stop()
            return
        if self._download_handle is not None:
            self._download_handle.set_prefetch(False)
            self._download_thread = threading.Thread(
                target=self._monitor_download,
                name="input-download-monitor",
                daemon=True,
            )
            self._download_thread.start()
        session.meta.pop("input_failed", None)
        session.meta.pop("processing_cancelled", None)
        session = self._hub.get_session(self._session_id) or self._hub.current
        if self._media_path is not None:
            session.meta["media_path"] = str(self._media_path)
            session.meta["media_name"] = self._media_path.name
            parts = session.meta.setdefault("media_parts", [])
            if str(self._media_path) not in [part["path"] for part in parts]:
                parts.append(
                    {
                        "path": str(self._media_path),
                        "video_id": str(self.spec.course.get("video_id") or ""),
                        "offset": self._audio_offset,
                        "duration": self._duration,
                        "recording_begin": self.spec.meta.get("recording_begin", ""),
                        "recording_end": self.spec.meta.get("recording_end", ""),
                    }
                )
        if self._audio_stop.is_set() or not self.running:
            return
        is_http = self._input_url.startswith(("http://", "https://"))
        headers = (
            dict(self.spec.headers) if is_http and self._download_handle is None else {}
        )
        if is_http:
            headers.setdefault("User-Agent", "sjtu-skipper/0.1")

        audio_cmd = build_audio_cmd(
            self._input_url,
            sample_rate=self._sample_rate,
            headers=headers,
        )
        if not is_live and self._realtime:
            audio_cmd.insert(1, "-re")
        if self._reconnect and is_http:
            pos = 1
            for arg in reversed(build_reconnect_args(self._reconnect_max_retries)):
                audio_cmd.insert(pos, arg)

        video_cmd = build_video_cmd(
            self._input_url,
            fps=self._video_fps,
            width=self._video_width,
            headers=headers,
            live=is_live,
        )
        if not is_live and self._realtime:
            video_cmd.insert(1, "-re")
        if self._reconnect and is_http:
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
            if self._download_handle is not None:
                self._download_handle.close()
            self._running = False
            self._emit(
                "input.error",
                message="ffmpeg 未安装或不在 PATH 中",
                source=self.spec.title,
            )
            return
        except Exception as exc:
            if self._download_handle is not None:
                self._download_handle.close()
            self._running = False
            self._emit(
                "input.error",
                message=f"启动音频进程失败：{exc}",
                source=self.spec.title,
            )
            return

        try:
            self._video_proc = subprocess.Popen(
                video_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            if self._download_handle is not None:
                self._download_handle.close()
            self._running = False
            if self._audio_proc:
                self._audio_proc.terminate()
                self._audio_proc = None
            self._emit(
                "input.error",
                message=f"启动视频进程失败：{exc}",
                source=self.spec.title,
            )
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
            status=SourceState.PLAYING,
            source=self.spec.title,
            kind=self.spec.kind,
            meta=dict(self.spec.meta),
        )

    def stop(self) -> None:
        self._cancelled = True
        if self._download_handle is not None:
            self._download_handle.close()
        with self._lock:
            self._cancelled = True
            self._audio_stop.set()
            self._video_stop.set()
            if not self._running:
                return
            self._running = False
        self._audio_stop.set()
        self._video_stop.set()
        self._pause_event.set()  # unpause so threads can exit
        session = self._hub.get_session(self._session_id)
        if session is not None and not self.finished:
            session.audio_gate.close()
            session.video_gate.close()

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
            status=SourceState.STOPPED,
            source=self.spec.title,
        )

    def pause(self) -> None:
        with self._lock:
            if not self._running or self._paused:
                return
            self._paused = True
            self._pause_event.clear()
        self._emit("input.status", status=SourceState.PAUSED, source=self.spec.title)

    def resume(self) -> None:
        with self._lock:
            if not self._running or not self._paused:
                return
            self._paused = False
            self._pause_event.set()
        self._emit("input.status", status=SourceState.PLAYING, source=self.spec.title)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "paused": self._paused,
                "kind": self.spec.kind,
                "title": self.spec.title,
                "url": self.spec.url,
            }

    def _mark_reader_done(self, stream: MediaKind) -> None:
        """Called by each reader at EOF; emits ``input.ended`` once for both.

        A recording's audio and video are decoded by independent ffmpeg
        processes.  At full speed the audio one finishes minutes before the
        video one, so ending the session on the first EOF would cut the video
        (and the VLM results) off.
        """

        with self._lock:
            if stream == MediaKind.AUDIO:
                self._audio_done = True
            else:
                self._video_done = True
            session = self._hub.get_session(self._session_id)
            if session is not None and self.spec.final_part:
                if stream == MediaKind.AUDIO:
                    session.audio_input_done = True
                else:
                    session.video_input_done = True
            _log.info(
                "reader %s reached EOF (audio_done=%s video_done=%s)",
                stream,
                self._audio_done,
                self._video_done,
            )
            if not (self._audio_done and self._video_done):
                return
            if self._ended_emitted:
                return
            self._ended_emitted = True

        if self._download_handle is not None:
            self._download_done.wait()
            if self._download_error:
                return
        try:
            current = self._hub.get_session(self._session_id) or self._hub.current
            current.input_done = self.spec.final_part
            current.input_done_at = time.time() if self.spec.final_part else None
        except Exception:  # noqa: BLE001 - metadata is best effort
            pass
        self._emit(
            "input.ended",
            source=self.spec.title,
            course=dict(self.spec.course),
            media_path=str(self._media_path) if self._media_path else "",
            session_id=self._session_id,
            video_id=str(self.spec.course.get("video_id") or ""),
            duration=(
                current.audio.duration - self._audio_offset
                or current.video.duration - self._video_offset
            ),
        )

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
                chunk = read(min(self._buffer_size, 6400))
            except Exception:
                chunk = None

            if chunk:
                retries = 0
                self._pause_event.wait()
                if self._audio_stop.is_set():
                    break
                current = (
                    self._hub.get_session(self._session_id)
                    if self._session_id
                    else self._hub.current
                )
                if current is None:
                    return
                for offset in range(0, len(chunk), 6400):
                    self._pause_event.wait()
                    if self._audio_stop.is_set():
                        return
                    payload = chunk[offset : offset + 6400]
                    duration = current.audio.bytes_to_duration(len(payload))
                    gate = current.audio_gate
                    if self._audio_enabled is not None and not self._audio_enabled():
                        current.audio.discard(duration)
                        gate.release(current.audio.duration)
                        continue
                    if not gate.wait(current.audio.duration + duration):
                        if not self._audio_stop.is_set():
                            self._emit(
                                "input.error",
                                message="Audio backlog limit timed out. Input stopped.",
                                source=self.spec.title,
                            )
                            self.stop()
                        return
                    if self._audio_stop.is_set():
                        return
                    current.audio.append(payload)
            else:
                rc = proc.poll()
                if rc is not None:
                    _log.info("Audio ffmpeg exited with code %s", rc)
                    if self.spec.kind in (SourceKind.RECORDING, SourceKind.FILE):
                        self._mark_reader_done(MediaKind.AUDIO)
                        break
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
                    _log.info(
                        "Audio reconnecting in %.1fs (attempt %d)", delay, retries
                    )
                    time.sleep(delay)
                    if self._audio_stop.is_set():
                        break
                    self._restart_audio_process()
                else:
                    time.sleep(0.1)

        _log.info("Audio reader thread exiting")

    def _restart_audio_process(self) -> None:
        if self._audio_proc is not None:
            try:
                self._audio_proc.terminate()
            except Exception:
                pass
        is_live = self.spec.kind == SourceKind.LIVE
        is_http = self._input_url.startswith(("http://", "https://"))
        headers = (
            dict(self.spec.headers) if is_http and self._download_handle is None else {}
        )
        cmd = build_audio_cmd(
            self._input_url,
            sample_rate=self._sample_rate,
            headers=headers,
        )
        if not is_live and self._realtime:
            cmd.insert(1, "-re")
        if self._reconnect and is_http:
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
                frames = split_jpeg_frames(buffer)
                if frames:
                    last_soi = buffer.rfind(b"\xff\xd8")
                    if last_soi >= 0:
                        last_eoi = buffer.rfind(b"\xff\xd9", last_soi)
                        if last_eoi < 0:
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
                            current = (
                                self._hub.get_session(self._session_id)
                                if self._session_id
                                else self._hub.current
                            )
                            if current is None:
                                return
                            gate = getattr(current, "video_gate", None)
                            if gate is not None and not gate.wait(
                                current.video.duration
                            ):
                                return
                            current.video.append_frame(frame_data)
                        except Exception:
                            _log.exception("video frame append failed")
            else:
                rc = proc.poll()
                if rc is not None:
                    _log.info("Video ffmpeg exited with code %s", rc)
                    if self.spec.kind in (SourceKind.RECORDING, SourceKind.FILE):
                        self._mark_reader_done(MediaKind.VIDEO)
                        break
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
        is_live = self.spec.kind == SourceKind.LIVE
        is_http = self._input_url.startswith(("http://", "https://"))
        headers = (
            dict(self.spec.headers) if is_http and self._download_handle is None else {}
        )
        cmd = build_video_cmd(
            self._input_url,
            fps=self._video_fps,
            width=self._video_width,
            headers=headers,
            live=is_live,
        )
        if not is_live and self._realtime:
            cmd.insert(1, "-re")
        if self._reconnect and is_http:
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
        reconnect: bool = True,
        media_root: Path | None = None,
        downloader: MediaDownloader | None = None,
        audio_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._hub = hub
        self._emit = emit
        self._sample_rate = int(sample_rate)
        self._video_fps = float(video_fps)
        self._video_width = int(video_width)
        self._reconnect = bool(reconnect)
        self._media_root = Path(media_root) if media_root is not None else None
        self._downloader = downloader or MediaDownloader()
        self._audio_enabled = audio_enabled
        self._lock = threading.RLock()
        self._current: FFmpegSource | None = None
        self._state: SourceState = SourceState.IDLE
        self._prefetched: FFmpegSource | None = None

    @property
    def state(self) -> SourceState:
        with self._lock:
            return self._state

    def configure(
        self,
        *,
        sample_rate: int | None = None,
        video_fps: float | None = None,
        video_width: int | None = None,
        reconnect: bool | None = None,
    ) -> None:
        """Apply live settings to the next source start."""

        with self._lock:
            if sample_rate is not None:
                self._sample_rate = int(sample_rate)
            if video_fps is not None:
                self._video_fps = float(video_fps)
            if video_width is not None:
                self._video_width = int(video_width)
            if reconnect is not None:
                self._reconnect = bool(reconnect)

    def start(self, spec: SourceSpec) -> None:
        """Start a source and retain the session for subsequent parts."""
        with self._lock:
            prepared = self._prefetched
            if prepared is not None and prepared.spec.url == spec.url:
                self._prefetched = None
            else:
                prepared = None
        with self._lock:
            self.stop()
            meta = dict(spec.meta)
            meta.update(spec.course)
            meta.setdefault("title", spec.title)
            meta.setdefault("kind", spec.kind)
            meta["mode"] = (
                ProcessingMode.LIVE
                if spec.kind == SourceKind.LIVE
                else (ProcessingMode.REALTIME if spec.realtime else ProcessingMode.FAST)
            )
            try:
                self._hub.video_fps = self._video_fps
                self._hub.video_width = self._video_width or 1920
                self._hub.video_height = 1080
                self._hub.audio_sample_rate = self._sample_rate
            except Exception:
                pass
            session_id = meta.get("class_id") or source_id(spec.url)
            current = self._hub.current
            reuse = (
                bool(spec.continue_session)
                and current.session_id == session_id
                and current.ended_at is None
            )
            if not reuse:
                current = self._hub.start_session(session_id, meta=meta)
            else:
                current.meta.update(meta)
            current.input_done = False
            current.input_done_at = None
            current.audio_input_done = False
            current.video_input_done = False
            current.audio_gate.configure(
                max(spec.audio_gate_limit, current.audio_gate.limit)
                if reuse
                else spec.audio_gate_limit,
                released=current.audio_gate.released if reuse else 0.0,
                fail_closed=True,
            )
            current.video_gate.configure(
                spec.video_gate_limit,
                released=current.video_gate.released if reuse else 0.0,
            )

            media_dir = None
            if self._media_root is not None:
                media_dir = self._media_root / current.session_id / "media"
            source = prepared or self._make_source(spec, media_dir)
            source.spec = spec
            self._current = source
            self._state = SourceState.PLAYING
        if spec.next_source is not None:
            self.prefetch(spec.next_source)
        source.start()
        if not source.running:
            # ffmpeg failed to start (missing binary, bad URL, ...).  The
            # source has already emitted input.error; do not keep a phantom
            # "playing" state in the UI.
            with self._lock:
                if self._current is source:
                    self._current = None
                    self._state = SourceState.IDLE
            current = self._hub.current
            if current.session_id != "boot" and current.ended_at is None:
                current.meta["input_failed"] = True
                self._hub.end_session("source failed to start")

    def _make_source(self, spec: SourceSpec, media_dir: Path | None) -> FFmpegSource:
        return FFmpegSource(
            spec,
            hub=self._hub,
            emit=self._emit,
            sample_rate=self._sample_rate,
            video_fps=self._video_fps,
            video_width=self._video_width,
            reconnect=self._reconnect,
            realtime=spec.realtime,
            downloader=self._downloader,
            media_dir=media_dir,
            audio_enabled=self._audio_enabled,
        )

    def prefetch(self, spec: SourceSpec) -> None:
        """Download the next part while consumers analyze the active part."""
        with self._lock:
            session = self._hub.current
            media_dir = (
                self._media_root / session.session_id / "media"
                if self._media_root is not None
                else None
            )
            source = self._make_source(spec, media_dir)
            source._session_id = session.session_id
            source._prepare_done = threading.Event()
            self._prefetched = source

        def prepare() -> None:
            try:
                source.prepare()
            except Exception as exc:
                if not source._cancelled:
                    self._emit(
                        "input.error",
                        message=f"准备录像失败：{exc}",
                        source=spec.title,
                        session_id=source._session_id,
                        video_id=str(spec.course.get("video_id") or ""),
                    )
            finally:
                assert source._prepare_done is not None
                source._prepare_done.set()

        source._prepare_thread = threading.Thread(
            target=prepare, name="input-prefetch", daemon=True
        )
        source._prepare_thread.start()

    def _join_source(self, source) -> None:
        for thread in (
            source._prepare_thread,
            source._download_thread,
            source._audio_thread,
            source._video_thread,
        ):
            if thread is not None and thread is not threading.current_thread():
                thread.join()

    def stop(self) -> None:
        with self._lock:
            source = self._current
            self._current = None
            self._state = SourceState.IDLE
            prefetched = self._prefetched
            self._prefetched = None
        if prefetched is not None:
            prefetched.stop()
            self._join_source(prefetched)
        if source is not None:
            finished = bool(getattr(source, "finished", False))
            source.stop()
            self._join_source(source)
            current = self._hub.current
            if (
                not finished
                and current.session_id != "boot"
                and current.ended_at is None
            ):
                self._hub.end_session("source stopped")

    def pause(self) -> None:
        with self._lock:
            source = self._current
        if source is not None:
            source.pause()
            with self._lock:
                self._state = SourceState.PAUSED

    def resume(self) -> None:
        with self._lock:
            source = self._current
        if source is not None:
            source.resume()
            with self._lock:
                self._state = SourceState.PLAYING

    def status(self) -> dict[str, Any]:
        with self._lock:
            source = self._current
            return {
                "state": self._state,
                "source": source.status() if source is not None else None,
            }
