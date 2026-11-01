"""FFmpeg helpers: JPEG frame boundary parser and command builders."""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("lecture_agent.plugins.input.ffmpeg")

# JPEG markers
_SOI = bytes([0xFF, 0xD8])
_EOI = bytes([0xFF, 0xD9])
# Any 0xFF marker that is not a restart marker (D0-D7)
_MARKER_PREFIX = 0xFF


class JPEGParseError(ValueError):
    """Raised when JPEG frame parsing fails on corrupted data."""


def split_jpeg_frames(data: bytes) -> list[bytes]:
    """Extract complete JPEG frames from a raw byte stream.

    A JPEG frame starts with SOI (``FF D8``) and ends with EOI (``FF D9``).
    Data between EOI and the next SOI is discarded (padding/partial frames).
    Any leading bytes before the first SOI are silently dropped.

    Handles sticky/half-frame scenarios: if the stream ends without EOI after
    the last SOI, that trailing data is buffered for the next call (not
    returned).

    Returns the list of complete JPEG frames found.
    """
    frames: list[bytes] = []
    offset = 0
    n = len(data)

    while offset < n:
        # Find next SOI
        soi_pos = data.find(_SOI, offset)
        if soi_pos < 0:
            break
        # Look for EOI after this SOI
        eoi_pos = data.find(_EOI, soi_pos + 2)
        if eoi_pos < 0:
            # No EOI found — trailing incomplete frame, stop here
            break
        # Valid frame: SOI .. EOI inclusive
        frames.append(data[soi_pos : eoi_pos + 2])
        offset = eoi_pos + 2

    return frames


def iter_jpeg_frames(data: bytes) -> list[bytes]:
    """Alias for split_jpeg_frames; returns complete frames, drops garbage."""
    return split_jpeg_frames(data)


def has_complete_frame(data: bytes) -> bool:
    """Return True if *data* contains at least one SOI–EOI pair."""
    soi = data.find(_SOI)
    if soi < 0:
        return False
    return data.find(_EOI, soi + 2) >= 0


def _append_http_headers(cmd: list[str], headers: dict[str, str] | None) -> None:
    """Append HTTP-only ffmpeg options as a single ``-headers`` blob."""

    if not headers:
        return
    blob = "\r\n".join(f"{key}: {value}" for key, value in headers.items()) + "\r\n"
    cmd.extend(["-headers", blob])
    ua = headers.get("User-Agent") or headers.get("user-agent")
    if ua:
        cmd.extend(["-user_agent", ua])
    ref = headers.get("Referer") or headers.get("referer")
    if ref:
        cmd.extend(["-referer", ref])


def build_audio_cmd(
    url: str,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    extra_args: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> list[str]:
    """Build an ffmpeg command that outputs raw s16le PCM on stdout.

    Parameters
    ----------
    url:
        Input URL or file path.
    sample_rate:
        Output sample rate in Hz (default 16000).
    channels:
        Output channel count (default 1, mono).
    extra_args:
        Additional ffmpeg arguments inserted before output spec.
    headers:
        HTTP headers (e.g. ``{"Referer": "…"}``) passed to ffmpeg.
    """
    cmd = ["ffmpeg"]
    _append_http_headers(cmd, headers)
    cmd.extend(
        [
            "-hide_banner",
            "-loglevel", "error",
            "-i", url,
            "-vn",  # no video
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-f", "s16le",
        ]
    )
    if extra_args:
        cmd.extend(extra_args)
    cmd.append("pipe:1")
    return cmd


def build_video_cmd(
    url: str,
    *,
    fps: float = 1.0,
    width: int = 1280,
    extra_args: list[str] | None = None,
    headers: dict[str, str] | None = None,
    live: bool = False,
) -> list[str]:
    """Build an ffmpeg command that outputs MJPEG frames on stdout.

    Parameters
    ----------
    url:
        Input URL (FLV for live, mp4 for VOD).
    fps:
        Output frame rate.
    width:
        Output frame width (height is auto-scaled).
    extra_args:
        Additional ffmpeg arguments.
    headers:
        HTTP headers.
    live:
        If True, add low-latency flags for live streaming.
    """
    cmd = ["ffmpeg"]
    _append_http_headers(cmd, headers)
    cmd.extend(["-hide_banner", "-loglevel", "error", "-i", url])
    if live:
        # Low-latency flags for live FLV streams
        cmd.extend(
            [
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-strict", "experimental",
                "-probesize", "32",
                "-analyzeduration", "0",
            ]
        )
    video_filter = f"fps={fps}"
    if width > 0:
        video_filter += f",scale={width}:-1"
    cmd.extend(
        [
            "-an",  # no audio in video sub-process
            "-vf", video_filter,
            "-c:v", "mjpeg",
            "-q:v", "3",
            "-f", "image2pipe",
        ]
    )
    if extra_args:
        cmd.extend(extra_args)
    cmd.append("pipe:1")
    return cmd


def build_reconnect_args(
    max_retries: int = 10,
    stream: bool = True,
) -> list[str]:
    """FFmpeg reconnect arguments for resilient streaming."""
    return [
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", str(max(1, int(max_retries))),
    ]