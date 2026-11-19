"""Append-only media and text streams.

* Every stream is append-only.  Existing chunks/items are immutable.
* Time is measured in seconds from the start of the current session.
* Audio is stored as raw PCM (s16le by default) and can be sliced exactly.
* Video is stored as encoded JPEG frames; ``read()`` returns concatenated
  frame bytes, while ``read_frames()`` exposes timestamps.
* Text is a growing sequence of immutable :class:`TextItem` objects.
* ``StreamHub`` owns four streams per session and coalesces high-frequency
  audio/video notifications into ``stream.appended`` events.
"""

from __future__ import annotations

import bisect
import logging
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable

from .media import MediaKind

Notify = Callable[..., None]
_log = logging.getLogger("lecture_agent.streams")


class StreamKind(StrEnum):
    AUDIO = "audio"
    VIDEO = "video"
    TRANSCRIPT = "transcript"
    VISUAL = "visual"


class TextSource(StrEnum):
    STT = "stt"
    VLM = "vlm"


@dataclass(frozen=True, slots=True)
class AudioChunk:
    start: float
    end: float
    data: bytes


@dataclass(frozen=True, slots=True)
class VideoFrame:
    index: int
    start: float
    end: float
    data: bytes


@dataclass(frozen=True, slots=True)
class TextItem:
    index: int
    text: str
    start: float | None = None
    end: float | None = None
    source: str | TextSource = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.meta, MappingProxyType):
            object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))


class _TimeIndexedStream:
    """Shared timeline/index machinery for audio and video streams."""

    stream_name: StreamKind

    def __init__(
        self,
        *,
        session_id: str,
        notify: Notify | None = None,
    ) -> None:
        self.session_id = session_id
        self._notify = notify
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._starts: list[float] = []
        self._chunks: list[Any] = []
        self._total = 0.0

    def __len__(self) -> int:
        """Whole seconds currently buffered.

        Python's ``len()`` protocol requires an ``int``.  Use the
        :attr:`duration` property when sub-second precision is needed.
        """

        with self._lock:
            return int(self._total)

    @property
    def duration(self) -> float:
        """Precise buffered duration in seconds."""

        with self._lock:
            return self._total

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def snapshot(self) -> tuple[float, int]:
        with self._lock:
            return self._total, len(self._chunks)

    def wait_for(self, end: float, timeout: float | None = None) -> bool:
        target = max(0.0, float(end))
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._total + 1e-9 < target:
                if deadline is None:
                    self._cv.wait(0.1)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cv.wait(min(0.1, remaining))
            return True

    def _append(self, chunk: Any, *, start: float, end: float, chars: int = 0) -> None:
        if end <= start:
            raise ValueError("chunk end must be greater than start")
        with self._lock:
            old_total = self._total
            old_count = len(self._chunks)
            if start < old_total - 1e-6:
                raise ValueError(
                    f"append start {start:.3f} overlaps existing end {old_total:.3f}"
                )
            if start > old_total + 1e-6:
                # Allow intentional gaps (e.g. live reconnect).
                self._starts.append(start)
            else:
                self._starts.append(old_total)
                start = old_total
            if hasattr(chunk, "index"):
                chunk = type(chunk)(
                    index=len(self._chunks), start=start, end=end, data=chunk.data
                )
            else:
                chunk = type(chunk)(start=start, end=end, data=chunk.data)
            self._chunks.append(chunk)
            self._total = max(self._total, end)
            self._cv.notify_all()
            new_total = self._total
            new_count = len(self._chunks)
        if self._notify is not None:
            self._notify(
                stream=self.stream_name,
                session_id=self.session_id,
                old_len=old_total,
                new_len=new_total,
                old_count=old_count,
                new_count=new_count,
                first_index=new_count - 1,
                chars=chars,
            )

    def trim(self, before: float) -> int:
        """Drop chunks that end at or before *before* (already consumed).

        Media streams are append-only for reads, but processed audio/video has
        no remaining consumer: without trimming a 2-hour lecture keeps every
        decoded frame in RAM (several GB).  ``duration`` keeps growing as if
        nothing was dropped, so timestamps and cursors stay valid.
        """

        before = float(before)
        if before <= 0.0:
            return 0
        with self._lock:
            count = 0
            for chunk in self._chunks:
                if chunk.end > before + 1e-9:
                    break
                count += 1
            if count <= 0:
                return 0
            del self._chunks[:count]
            del self._starts[:count]
            return count

    @property
    def kept_seconds(self) -> float:
        """Media seconds still buffered (after trimming)."""

        with self._lock:
            if not self._chunks:
                return 0.0
            return max(0.0, self._total - self._chunks[0].start)

    def _overlapping(self, start: float, end: float) -> list[Any]:
        start = max(0.0, float(start))
        end = float(end)
        if end <= start:
            return []
        with self._lock:
            if not self._chunks:
                return []
            idx = bisect.bisect_right(self._starts, start) - 1
            idx = max(0, idx)
            out: list[Any] = []
            for chunk in self._chunks[idx:]:
                if chunk.start >= end:
                    break
                if chunk.end <= start:
                    continue
                out.append(chunk)
            return out

    def read_frames(self, start: float, end: float) -> list[VideoFrame]:
        return list(self._overlapping(start, end))

    def read(self, start: float, end: float) -> bytes:
        return b"".join(chunk.data for chunk in self._overlapping(start, end))


class AudioStream(_TimeIndexedStream):
    """Append-only PCM audio stream."""

    stream_name = StreamKind.AUDIO

    def __init__(
        self,
        session_id: str,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        notify: Notify | None = None,
    ) -> None:
        super().__init__(session_id=session_id, notify=notify)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_width = int(sample_width)

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width

    def bytes_to_duration(self, size: int) -> float:
        return float(size) / float(self.bytes_per_second)

    def append(self, pcm: bytes, duration: float | None = None) -> None:
        if not pcm:
            return
        if not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise TypeError("pcm must be bytes-like")
        data = bytes(pcm)
        dur = self.bytes_to_duration(len(data)) if duration is None else float(duration)
        if dur <= 0:
            raise ValueError("audio duration must be positive")
        total, _ = self.snapshot()
        self._append(
            AudioChunk(start=total, end=total + dur, data=data),
            start=total,
            end=total + dur,
        )

    def discard(self, duration: float) -> None:
        """Advance the timeline without retaining PCM or inserting silence."""
        duration = float(duration)
        if duration <= 0:
            return
        with self._cv:
            old_total = self._total
            self._total += duration
            new_total = self._total
            count = len(self._chunks)
            self._cv.notify_all()
        if self._notify is not None:
            self._notify(
                stream=self.stream_name,
                session_id=self.session_id,
                old_len=old_total,
                new_len=new_total,
                old_count=count,
                new_count=count,
                first_index=count,
                chars=0,
            )

    @property
    def earliest_time(self) -> float:
        """Return the first retained PCM time, or the logical end when empty."""
        with self._lock:
            return self._chunks[0].start if self._chunks else self._total

    def buffered_ranges(
        self, start: float = 0.0, end: float | None = None
    ) -> list[tuple[float, float]]:
        """Return continuous intervals that contain retained PCM."""
        with self._lock:
            end = self._total if end is None else float(end)
            ranges: list[tuple[float, float]] = []
            for chunk in self._overlapping(start, end):
                lo, hi = max(start, chunk.start), min(end, chunk.end)
                if ranges and lo <= ranges[-1][1] + 1e-9:
                    ranges[-1] = (ranges[-1][0], hi)
                else:
                    ranges.append((lo, hi))
            return ranges

    def read(self, start: float, end: float) -> bytes:
        bps = self.bytes_per_second
        parts: list[bytes] = []
        for chunk in self._overlapping(start, end):
            lo = max(float(start), chunk.start)
            hi = min(float(end), chunk.end)
            if hi <= lo:
                continue
            b0 = max(0, int(round((lo - chunk.start) * bps)))
            b1 = min(len(chunk.data), int(round((hi - chunk.start) * bps)))
            if b1 > b0:
                parts.append(chunk.data[b0:b1])
        return b"".join(parts)

    def read_as_wav(self, start: float, end: float) -> bytes:
        """Return a complete WAV container for ``[start, end)``."""

        import io
        import wave

        pcm = self.read(start, end)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()


class VideoStream(_TimeIndexedStream):
    """Append-only encoded-frame video stream (JPEG by default)."""

    stream_name = StreamKind.VIDEO

    def __init__(
        self,
        session_id: str,
        *,
        fps: float = 1.0,
        width: int = 1280,
        height: int = 720,
        codec: str = "mjpeg",
        notify: Notify | None = None,
    ) -> None:
        super().__init__(session_id=session_id, notify=notify)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.codec = codec

    def append_frame(
        self,
        jpeg: bytes,
        duration: float | None = None,
        *,
        start: float | None = None,
    ) -> VideoFrame:
        if not jpeg:
            raise ValueError("empty video frame")
        data = bytes(jpeg)
        total, _ = self.snapshot()
        begin = total if start is None else max(float(start), total)
        dur = (1.0 / self.fps) if duration is None else float(duration)
        if dur <= 0:
            raise ValueError("frame duration must be positive")
        frame = VideoFrame(index=0, start=begin, end=begin + dur, data=data)
        self._append(frame, start=begin, end=begin + dur)
        with self._lock:
            return self._chunks[-1]

    def read(self, start: float, end: float) -> bytes:
        return b"".join(frame.data for frame in self._overlapping(start, end))


class TextStream(Sequence[TextItem]):
    """Append-only text stream; ``len()`` is the number of items."""

    def __init__(
        self,
        session_id: str,
        name: str | StreamKind = "text",
        *,
        notify: Notify | None = None,
    ) -> None:
        self.session_id = session_id
        self.name = name
        self._notify = notify
        self._lock = threading.RLock()
        self._items: list[TextItem] = []
        self._char_count = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __getitem__(self, index: int | slice) -> TextItem | list[TextItem]:
        with self._lock:
            return self._items[index]

    def __iter__(self) -> Iterator[TextItem]:
        with self._lock:
            items = list(self._items)
        return iter(items)

    @property
    def char_count(self) -> int:
        with self._lock:
            return self._char_count

    def snapshot_index(self) -> int:
        with self._lock:
            return len(self._items)

    def append(
        self,
        text: str,
        *,
        start: float | None = None,
        end: float | None = None,
        source: str | TextSource = "",
        meta: Mapping[str, Any] | None = None,
    ) -> TextItem | None:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if text == "":
            return None
        with self._lock:
            index = len(self._items)
            item = TextItem(
                index=index,
                text=text,
                start=start,
                end=end,
                source=source,
                meta=dict(meta or {}),
            )
            self._items.append(item)
            self._char_count += len(text)
            count = len(self._items)
        if self._notify is not None:
            self._notify(
                stream=self.name,
                session_id=self.session_id,
                old_len=count - 1,
                new_len=count,
                old_count=count - 1,
                new_count=count,
                first_index=index,
                chars=len(text),
                start=start,
                end=end,
            )
        return item

    def text_since(self, index: int = 0) -> str:
        with self._lock:
            start = max(0, int(index))
            return "".join(item.text for item in self._items[start:])

    def items_since(self, index: int = 0) -> list[TextItem]:
        with self._lock:
            return list(self._items[max(0, int(index)) :])

    def char_count_since(self, index: int = 0) -> int:
        with self._lock:
            start = max(0, int(index))
            return sum(len(item.text) for item in self._items[start:])

    def to_text(self, separator: str = "\n") -> str:
        with self._lock:
            return separator.join(item.text for item in self._items)


class BacklogGate:
    """Throttle a producer so unprocessed media stays bounded.

    Recording mode ingests media as fast as ffmpeg can decode it.  Without a
    gate a whole lecture (hours of PCM/MJPEG) would pile up in memory before
    the (slower) STT/VLM consumers catch up.  Readers call :meth:`wait` before
    appending; consumers call :meth:`release` after committing a window.

    ``limit <= 0`` disables the gate. A stalled consumer disables the gate
    unless ``fail_closed`` requires the producer to stop instead.
    """

    def __init__(
        self,
        name: str,
        *,
        limit: float = 0.0,
        stall_timeout: float = 300.0,
    ) -> None:
        self.name = name
        self._limit = max(0.0, float(limit))
        self._stall_timeout = max(0.0, float(stall_timeout))
        self._lock = threading.Condition(threading.Lock())
        self._released = 0.0
        self._last_release = time.monotonic()
        self._closed = False
        self._fail_closed = False

    @property
    def limit(self) -> float:
        with self._lock:
            return self._limit

    @property
    def released(self) -> float:
        with self._lock:
            return self._released

    def configure(
        self, limit: float, *, released: float = 0.0, fail_closed: bool = False
    ) -> None:
        with self._lock:
            self._limit = max(0.0, float(limit))
            self._released = max(0.0, float(released))
            self._fail_closed = bool(fail_closed)
            self._last_release = time.monotonic()
            self._closed = False
            self._lock.notify_all()

    def release(self, seconds: float) -> None:
        with self._lock:
            self._released = max(self._released, float(seconds))
            self._last_release = time.monotonic()
            self._lock.notify_all()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    def wait(self, media_end: float) -> bool:
        """Block until ``media_end - released <= limit``.

        Returns False when the gate closes or a fail-closed gate times out.
        """

        with self._lock:
            while not self._closed:
                if self._limit <= 0:
                    return True
                if float(media_end) - self._released <= self._limit + 1e-9:
                    return True
                if (
                    self._stall_timeout > 0
                    and time.monotonic() - self._last_release > self._stall_timeout
                ):
                    if self._fail_closed:
                        _log.error(
                            "backlog gate %s stalled; stopping producer", self.name
                        )
                        self._closed = True
                        return False
                    _log.warning(
                        "backlog gate %s stalled for %.0fs; disabling",
                        self.name,
                        self._stall_timeout,
                    )
                    self._limit = 0.0
                    return True
                self._lock.wait(0.2)
            return False


@dataclass(slots=True)
class SourceProgress:
    before: float = 0.0
    final: bool = False
    enabled: bool = True
    errors: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class StreamSet:
    session_id: str
    started_at: float
    meta: dict[str, Any]
    audio: AudioStream
    video: VideoStream
    transcript: TextStream
    visual: TextStream
    ended_at: float | None = None
    end_reason: str = ""
    audio_gate: BacklogGate = field(
        default_factory=lambda: BacklogGate(MediaKind.AUDIO)
    )
    video_gate: BacklogGate = field(
        default_factory=lambda: BacklogGate(MediaKind.VIDEO)
    )
    # Set by the input plugin once all ffmpeg readers reached EOF; consumers
    # use it to drain their queues before the session is ended.
    input_done: bool = False
    input_done_at: float | None = None
    audio_input_done: bool = False
    video_input_done: bool = False
    transcript_progress: SourceProgress = field(default_factory=SourceProgress)
    visual_progress: SourceProgress = field(default_factory=SourceProgress)

    def stream(self, name: StreamKind) -> AudioStream | VideoStream | TextStream:
        if not isinstance(name, StreamKind):
            raise TypeError("name must be a StreamKind")
        streams: dict[StreamKind, AudioStream | VideoStream | TextStream] = {
            StreamKind.AUDIO: self.audio,
            StreamKind.VIDEO: self.video,
            StreamKind.TRANSCRIPT: self.transcript,
            StreamKind.VISUAL: self.visual,
        }
        return streams[name]

    @property
    def duration(self) -> float:
        return max(self.audio.duration, self.video.duration)

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "meta": dict(self.meta),
            "audio_seconds": self.audio.duration,
            "video_seconds": self.video.duration,
            "transcript_items": len(self.transcript),
            "transcript_chars": self.transcript.char_count,
            "visual_items": len(self.visual),
            "visual_chars": self.visual.char_count,
            "input_done": self.input_done,
            "buffered_audio_seconds": round(self.audio.kept_seconds, 1),
            "buffered_video_seconds": round(self.video.kept_seconds, 1),
        }


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    started_at: float
    ended_at: float | None
    meta: Mapping[str, Any]
    audio_seconds: float
    transcript_items: int
    transcript_chars: int
    visual_items: int
    visual_chars: int


class StreamHub:
    """Owns the current session's four streams and emits stream events."""

    def __init__(
        self,
        *,
        emit: Callable[..., Any] | None = None,
        audio_emit_interval: float = 0.5,
        video_emit_interval: float = 1.0,
        audio_sample_rate: int = 16000,
        video_fps: float = 1.0,
        video_width: int = 1280,
        video_height: int = 720,
    ) -> None:
        self._emit = emit
        self.audio_emit_interval = max(0.05, float(audio_emit_interval))
        self.video_emit_interval = max(0.1, float(video_emit_interval))
        self.audio_sample_rate = int(audio_sample_rate)
        self.video_fps = float(video_fps)
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        self._lock = threading.RLock()
        self._sessions: list[StreamSet] = []
        self._last_emit: dict[tuple[str, StreamKind], float] = {}
        # (plugin_id, kind) -> callbacks describing one media consumer.
        # Producers ask the hub about cursors/backpressure instead of reaching
        # into the consumer plugins directly (no ``input -> stt/vlm`` imports).
        self._consumers: dict[tuple[str, MediaKind], dict[str, Any]] = {}
        self._current = self._create("boot", {})
        self._sessions.append(self._current)
        if self._emit is not None:
            self._emit(
                "session.started",
                session_id=self._current.session_id,
                meta={},
                previous_session_id=None,
                started_at=self._current.started_at,
            )

    def _create(self, session_id: str, meta: Mapping[str, Any]) -> StreamSet:
        sid = session_id
        notify = self._on_append
        return StreamSet(
            session_id=sid,
            started_at=time.time(),
            meta=dict(meta),
            audio=AudioStream(
                sid,
                sample_rate=self.audio_sample_rate,
                notify=notify,
            ),
            video=VideoStream(
                sid,
                fps=self.video_fps,
                width=self.video_width,
                height=self.video_height,
                notify=notify,
            ),
            transcript=TextStream(sid, StreamKind.TRANSCRIPT, notify=notify),
            visual=TextStream(sid, StreamKind.VISUAL, notify=notify),
        )

    @property
    def current(self) -> StreamSet:
        with self._lock:
            return self._current

    def start_session(
        self, session_id: str, *, meta: Mapping[str, Any] | None = None
    ) -> StreamSet:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", session_id) is None:
            raise ValueError("session_id must be a safe path component")
        stopped_payload: dict[str, Any] | None = None
        with self._lock:
            previous = self._current
            previous_id = previous.session_id if previous.session_id != "boot" else None
            has_content = (
                previous.duration > 0
                or len(previous.transcript) > 0
                or len(previous.visual) > 0
            )
            if previous.ended_at is None and (previous_id is not None or has_content):
                self._end_locked(previous, "replaced by new session")
                stopped_payload = {
                    "session_id": previous.session_id,
                    "reason": previous.end_reason,
                    "ended_at": previous.ended_at,
                    "started_at": previous.started_at,
                    "meta": dict(previous.meta),
                }
            new_set = self._create(session_id, meta or {})
            self._current = new_set
            self._sessions[:] = [
                session
                for session in self._sessions
                if session.session_id != session_id
            ]
            self._sessions.append(new_set)
            started_payload = {
                "session_id": new_set.session_id,
                "meta": dict(new_set.meta),
                "previous_session_id": previous_id,
                "started_at": new_set.started_at,
            }
        if self._emit is not None:
            if stopped_payload is not None:
                self._emit("session.stopped", **stopped_payload)
            self._emit("session.started", **started_payload)
        return new_set

    def end_session(self, reason: str = "") -> None:
        with self._lock:
            current = self._current
            if current.ended_at is not None:
                return
            self._end_locked(current, reason)
        if self._emit is not None:
            self._emit(
                "session.stopped",
                session_id=current.session_id,
                reason=reason,
                ended_at=current.ended_at,
                started_at=current.started_at,
                meta=dict(current.meta),
            )

    def _end_locked(self, session: StreamSet, reason: str) -> None:
        session.ended_at = time.time()
        session.end_reason = reason
        session.audio_gate.close()
        session.video_gate.close()
        for stream_name in (StreamKind.AUDIO, StreamKind.VIDEO):
            stream = session.stream(stream_name)
            key = (session.session_id, stream_name)
            last = self._last_emit.get(key, 0.0)
            new_len = stream.duration
            if new_len > last + 1e-9 and self._emit is not None:
                self._emit(
                    "stream.appended",
                    stream=stream_name,
                    session_id=session.session_id,
                    old_len=last,
                    new_len=new_len,
                    delta=new_len - last,
                    start=last,
                    end=new_len,
                    chars=0,
                    first_index=0,
                )
                self._last_emit[key] = new_len
        self._last_emit.pop((session.session_id, StreamKind.AUDIO), None)
        self._last_emit.pop((session.session_id, StreamKind.VIDEO), None)

    def _on_append(
        self,
        *,
        stream: StreamKind,
        session_id: str,
        old_len: float | int,
        new_len: float | int,
        old_count: int,
        new_count: int,
        first_index: int,
        chars: int = 0,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        if not isinstance(stream, StreamKind):
            raise TypeError("stream must be a StreamKind")
        if self._emit is None:
            return
        if stream in (StreamKind.AUDIO, StreamKind.VIDEO):
            interval = (
                self.audio_emit_interval
                if stream is StreamKind.AUDIO
                else self.video_emit_interval
            )
            key = (session_id, stream)
            with self._lock:
                last = self._last_emit.get(key, 0.0)
                if float(new_len) - last + 1e-9 < interval:
                    return
                self._last_emit[key] = float(new_len)
            self._emit(
                "stream.appended",
                stream=stream,
                session_id=session_id,
                old_len=last,
                new_len=float(new_len),
                delta=float(new_len) - last,
                start=last,
                end=float(new_len),
                chars=0,
                first_index=0,
            )
            return
        self._emit(
            "stream.appended",
            stream=stream,
            session_id=session_id,
            old_len=old_count,
            new_len=new_count,
            delta=new_count - old_count,
            start=first_index,
            end=first_index,
            chars=chars,
            first_index=first_index,
            item_start=start,
            item_end=end,
        )

    def list_sessions(self) -> list[SessionInfo]:
        with self._lock:
            sessions = list(self._sessions)
        out: list[SessionInfo] = []
        for s in sessions:
            out.append(
                SessionInfo(
                    session_id=s.session_id,
                    started_at=s.started_at,
                    ended_at=s.ended_at,
                    meta=dict(s.meta),
                    audio_seconds=s.audio.duration,
                    transcript_items=len(s.transcript),
                    transcript_chars=s.transcript.char_count,
                    visual_items=len(s.visual),
                    visual_chars=s.visual.char_count,
                )
            )
        return out

    def get_session(self, session_id: str) -> StreamSet | None:
        with self._lock:
            for s in self._sessions:
                if s.session_id == session_id:
                    return s
        return None

    def _text_progress_locked(
        self, session_id: str, stream: StreamKind
    ) -> SourceProgress:
        if not isinstance(stream, StreamKind):
            raise TypeError("stream must be a StreamKind")
        if stream not in (StreamKind.TRANSCRIPT, StreamKind.VISUAL):
            raise ValueError(f"Invalid text stream: {stream}")
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        return getattr(session, f"{stream}_progress")

    def text_progress(self, session_id: str, stream: StreamKind) -> SourceProgress:
        with self._lock:
            state = self._text_progress_locked(session_id, stream)
            return SourceProgress(
                state.before,
                state.final,
                state.enabled,
                [dict(error) for error in state.errors],
            )

    def seal_text(
        self,
        session_id: str,
        stream: StreamKind,
        before: float,
        *,
        final: bool = False,
        error: dict | None = None,
    ) -> None:
        with self._lock:
            state = self._text_progress_locked(session_id, stream)
            state.before = max(state.before, float(before))
            state.final = state.final or final
            if error is not None:
                state.errors.append(dict(error))
            payload = dict(
                session_id=session_id,
                stream=stream,
                before=state.before,
                final=state.final,
                enabled=state.enabled,
                error=error,
            )
        if self._emit is not None:
            self._emit("stream.sealed", **payload)

    def set_text_enabled(
        self, session_id: str, stream: StreamKind, enabled: bool
    ) -> None:
        with self._lock:
            state = self._text_progress_locked(session_id, stream)
            state.enabled = bool(enabled)
            payload = dict(
                session_id=session_id,
                stream=stream,
                before=state.before,
                final=state.final,
                enabled=state.enabled,
                error=None,
            )
        if self._emit is not None:
            self._emit("stream.sealed", **payload)

    def status(self) -> dict[str, Any]:
        return self.current.status()

    def shutdown(self) -> None:
        self.end_session("runtime shutdown")

    def register_consumer(
        self,
        plugin_id: str,
        kind: MediaKind,
        *,
        cursor_fn: Callable[[], float] | None = None,
        busy_fn: Callable[[], bool] | None = None,
        gate_limit_fn: Callable[[], float] | None = None,
    ) -> None:
        """Register a media consumer for the specified media kind.

        Producers use the consumer callbacks to control media backlog.
        The input plugin does not need access to the consumer plugin.
        """

        if not isinstance(kind, MediaKind):
            raise TypeError("kind must be a MediaKind")
        key = (str(plugin_id), kind)
        with self._lock:
            self._consumers[key] = {
                "cursor_fn": cursor_fn,
                "busy_fn": busy_fn,
                "gate_limit_fn": gate_limit_fn,
            }

    def unregister_consumer(
        self, plugin_id: str, kind: MediaKind | None = None
    ) -> None:
        if kind is not None and not isinstance(kind, MediaKind):
            raise TypeError("kind must be a MediaKind")
        with self._lock:
            for key in list(self._consumers):
                if key[0] != plugin_id:
                    continue
                if kind is not None and key[1] is not kind:
                    continue
                self._consumers.pop(key, None)

    def remove_plugin(self, plugin_id: str) -> None:
        """Drop every consumer registered by *plugin_id* (plugin teardown)."""

        self.unregister_consumer(plugin_id)

    def consumers(self) -> list[dict[str, str | MediaKind]]:
        """Introspection: registered consumers as ``{plugin_id, kind}``."""

        with self._lock:
            keys = list(self._consumers)
        return [{"plugin_id": pid, "kind": kind} for pid, kind in keys]

    def _consumer_fns(self, kind: MediaKind, name: str) -> list[Callable[..., Any]]:
        if not isinstance(kind, MediaKind):
            raise TypeError("kind must be a MediaKind")
        with self._lock:
            consumers = [
                dict(entry)
                for (pid, entry_kind), entry in self._consumers.items()
                if entry_kind is kind
            ]
        return [c[name] for c in consumers if callable(c.get(name))]

    def consumer_cursor(self, kind: MediaKind) -> float:
        """Furthest-back position consumed by the ``kind`` consumers.

        Returns ``0.0`` when nothing has been consumed yet (or when there are no
        consumers at all), which is what a producer treats as "no backpressure".
        """

        values: list[float] = []
        for fn in self._consumer_fns(kind, "cursor_fn"):
            try:
                values.append(float(fn() or 0.0))
            except Exception:  # noqa: BLE001 - a broken consumer must not stall input
                _log.exception("consumer cursor failed for kind=%s", kind)
        return min(values) if values else 0.0

    def consumer_gate_limit(self, kind: MediaKind) -> float:
        """Return the strictest positive consumer backlog limit, or zero."""

        limits: list[float] = []
        for fn in self._consumer_fns(kind, "gate_limit_fn"):
            try:
                limit = float(fn() or 0.0)
            except Exception:  # noqa: BLE001 - a broken consumer means "no limit"
                _log.exception("consumer gate limit failed for kind=%s", kind)
                continue
            if limit > 0.0:
                limits.append(limit)
        return min(limits) if limits else 0.0

    def consumers_busy(self) -> list[str]:
        """Plugin ids whose consumer currently reports itself busy."""

        with self._lock:
            entries = [
                (pid, dict(entry)) for (pid, _kind), entry in self._consumers.items()
            ]
        busy: list[str] = []
        for pid, entry in entries:
            fn = entry.get("busy_fn")
            if not callable(fn):
                continue
            try:
                if fn():
                    busy.append(pid)
            except Exception:  # noqa: BLE001 - busy is best effort
                _log.exception("consumer busy check failed for plugin=%s", pid)
        return list(dict.fromkeys(busy))
