"""Append-only media and text streams.

Design notes
------------
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
import secrets
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable

Notify = Callable[..., None]


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
    source: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.meta, MappingProxyType):
            object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))


class _TimeIndexedStream:
    """Shared timeline/index machinery for audio and video streams."""

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
    def total_time(self) -> float:
        return self.duration

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
                chunk = type(chunk)(index=len(self._chunks), start=start, end=end, data=chunk.data)
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

    stream_name = "audio"

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

    def duration_to_bytes(self, duration: float) -> int:
        return max(0, int(round(float(duration) * self.bytes_per_second)))

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
        self._append(AudioChunk(start=total, end=total + dur, data=data), start=total, end=total + dur)

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

    stream_name = "video"

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

    def __init__(self, session_id: str, name: str = "text", *, notify: Notify | None = None) -> None:
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
        source: str = "",
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

    def item_at(self, index: int) -> TextItem:
        with self._lock:
            return self._items[index]

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

    def stream(self, name: str) -> AudioStream | VideoStream | TextStream:
        value = getattr(self, name, None)
        if value is None:
            raise KeyError(f"unknown stream {name!r}")
        return value

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
        self._last_emit: dict[tuple[str, str], float] = {}
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

    # ------------------------------------------------------------------
    # Creation / lifecycle
    # ------------------------------------------------------------------
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
            transcript=TextStream(sid, "transcript", notify=notify),
            visual=TextStream(sid, "visual", notify=notify),
        )

    @property
    def current(self) -> StreamSet:
        with self._lock:
            return self._current

    def start_session(self, *, meta: Mapping[str, Any] | None = None) -> StreamSet:
        previous_stopped: StreamSet | None = None
        with self._lock:
            previous = self._current
            has_content = (
                previous.duration > 0
                or len(previous.transcript) > 0
                or len(previous.visual) > 0
            )
            if previous.ended_at is None and has_content:
                self._end_locked(previous, "replaced by new session")
                previous_stopped = previous
                previous_id = previous.session_id
            else:
                previous_id = previous.session_id if previous.session_id != "boot" else None
            new_set = self._create(self._new_session_id(), meta or {})
            self._current = new_set
            self._sessions.append(new_set)
        if previous_stopped is not None and self._emit is not None:
            self._emit(
                "session.stopped",
                session_id=previous_stopped.session_id,
                reason=previous_stopped.end_reason,
                ended_at=previous_stopped.ended_at,
            )
        if self._emit is not None:
            self._emit(
                "session.started",
                session_id=new_set.session_id,
                meta=dict(new_set.meta),
                previous_session_id=previous_id,
                started_at=new_set.started_at,
            )
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
            )

    def _end_locked(self, session: StreamSet, reason: str) -> None:
        session.ended_at = time.time()
        session.end_reason = reason
        # Flush coalesced audio/video notifications.
        for stream_name in ("audio", "video"):
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
        self._last_emit.pop((session.session_id, "audio"), None)
        self._last_emit.pop((session.session_id, "video"), None)

    def _new_session_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    def _on_append(
        self,
        *,
        stream: str,
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
        if self._emit is None:
            return
        if stream in ("audio", "video"):
            interval = self.audio_emit_interval if stream == "audio" else self.video_emit_interval
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
        # Text streams notify immediately.
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

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
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

    def status(self) -> dict[str, Any]:
        return self.current.status()

    def shutdown(self) -> None:
        self.end_session("runtime shutdown")
