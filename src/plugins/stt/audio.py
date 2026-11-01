"""PCM/WAV helpers and chunk-window arithmetic for the STT plugin.

The audio stream stores raw PCM (s16le, mono, 16 kHz by default).  Speech
backends expect a complete WAV container, and long lessons must be cut into
overlapping windows so that no word is lost at a boundary.  This module owns
all of that arithmetic so the worker stays simple and the rules are unit
testable without any I/O.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2
MIN_OVERLAP_CHARS = 4


def pcm_to_wav(
    pcm: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> bytes:
    """Wrap raw PCM ``pcm`` into a complete WAV container.

    ``sample_width`` is the bytes per sample (2 for s16le).  An empty payload
    still produces a valid, empty WAV file instead of raising.
    """

    if not isinstance(pcm, (bytes, bytearray, memoryview)):
        raise TypeError("pcm must be bytes-like")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(int(channels))
        writer.setsampwidth(int(sample_width))
        writer.setframerate(int(sample_rate))
        writer.writeframes(bytes(pcm))
    return buffer.getvalue()


def wav_duration(wav_bytes: bytes) -> float:
    """Return the duration in seconds of a WAV container."""

    if not wav_bytes:
        raise ValueError("empty wav payload")
    try:
        with wave.open(io.BytesIO(bytes(wav_bytes)), "rb") as reader:
            frames = reader.getnframes()
            rate = reader.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"invalid wav payload: {exc}") from exc
    if rate <= 0:
        raise ValueError("invalid wav sample rate")
    return frames / float(rate)


def pcm_duration(
    pcm: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> float:
    """Duration of raw PCM given the stream format."""

    bytes_per_second = int(sample_rate) * int(channels) * int(sample_width)
    if bytes_per_second <= 0:
        raise ValueError("invalid audio format")
    return len(pcm) / float(bytes_per_second)


@dataclass(frozen=True, slots=True)
class Window:
    """A half-open time window ``[start, end)`` in session seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def compute_windows(
    total_duration: float,
    chunk_seconds: float,
    overlap: float = 0.0,
) -> list[Window]:
    """Split ``[0, total_duration)`` into overlapping transcribe windows.

    The first window starts at ``0`` and each following window starts
    ``chunk_seconds - overlap`` seconds later, so consecutive windows share
    ``overlap`` seconds of audio for context.  The final window is truncated at
    ``total_duration`` and the loop stops as soon as it is reached: no tiny
    trailing sliver that would only duplicate already transcribed audio.
    """

    total = max(0.0, float(total_duration))
    chunk = float(chunk_seconds)
    gap = max(0.0, float(overlap))
    if chunk <= 0:
        raise ValueError("chunk_seconds must be positive")
    if gap >= chunk:
        raise ValueError("overlap must be smaller than chunk_seconds")
    if total <= 0:
        return []

    windows: list[Window] = []
    start = 0.0
    while start < total - 1e-9:
        end = min(start + chunk, total)
        windows.append(Window(start=round(start, 6), end=round(end, 6)))
        if end >= total - 1e-9:
            break
        next_start = end - gap
        if next_start <= start + 1e-9:  # pragma: no cover - guarded by gap < chunk
            break
        start = next_start
    return windows


def common_overlap(previous: str, current: str, *, min_overlap: int = MIN_OVERLAP_CHARS) -> int:
    """Length of the longest suffix of ``previous`` that prefixes ``current``.

    Returns ``0`` when no common run of at least ``min_overlap`` characters
    exists.  Compared on raw characters: transcription backends are usually
    consistent inside the overlap region, and a stricter normalisation would
    risk removing text that merely looks similar.
    """

    if not previous or not current:
        return 0
    floor = max(1, int(min_overlap))
    upper = min(len(previous), len(current))
    for size in range(upper, floor - 1, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0


def dedupe_overlap(
    previous: str,
    current: str,
    *,
    min_overlap: int = MIN_OVERLAP_CHARS,
) -> str:
    """Return the not-yet-transcribed tail of ``current``.

    ``previous`` is the text already appended to the transcript, ``current``
    the raw transcription of a window that overlaps the previous one.  When the
    shared run is shorter than ``min_overlap`` no dedupe happens (it would be
    guesswork), and when the whole window is a repeat the result is ``""``.
    """

    if not current:
        return ""
    overlap = common_overlap(previous, current, min_overlap=min_overlap)
    if overlap == 0:
        return current
    return current[overlap:].lstrip()
