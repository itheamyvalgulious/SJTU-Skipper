"""Blackboard frame fusion adapted from picture_parser for real-time windows.

Given a list of JPEG-encoded video frames, produces a single fused
grayscale image that suppresses teacher occlusion, camera PTZ drift
ghosting, and transient eraser lines while preserving persistent chalk.

The original algorithm lives in ``legacy/blackboard_lecture_agent.py``
(:func:`extract_registered_persistence_fusion`).  This module adapts it
to accept decoded frames and exposes tuneable parameters.
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
from PIL import Image

from core.streams import VideoFrame

logger = logging.getLogger("lecture_agent.plugin.vlm.fusion")


def fuse_frames(
    frames: Sequence[VideoFrame],
    *,
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    max_width: int = 0,
    align: bool = True,
    min_persistence: int | None = None,
) -> np.ndarray:
    """Fuse a window of JPEG video frames into one clean blackboard image.

    Parameters
    ----------
    frames:
        Video frames with JPEG ``data`` payloads.  Must all share the
        same resolution (the algorithm assumes a fixed board crop).
    crop:
        Normalised board crop ``(x0, y0, x1, y1)``, each in [0, 1].
        Defaults to the full frame.
    max_width:
        Resize the decoded image so its width does not exceed this value.
        Set to 0 or a very large number to skip resize.
    align:
        When True, use FFT phase correlation to register frames against
        the middle frame before fusion.
    min_persistence:
        Minimum number of frames a chalk pixel must appear in to be
        considered persistent.  Defaults to 5, matching the original
        picture_parser pipeline.

    Returns
    -------
    np.ndarray
        Fused grayscale image (uint8, 2-D).

    Notes
    -----
    - At least 3 valid frames are required; there is no silent fallback.
    - Decode/fusion errors are raised to the caller.
    """
    if len(frames) < 3:
        raise ValueError("at least 3 frames are required for fusion")

    return _fuse_impl(
        frames,
        crop=crop,
        max_width=max_width,
        align=align,
        min_persistence=min_persistence,
    )


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def _fuse_impl(
    frames: Sequence[VideoFrame],
    *,
    crop: tuple[float, float, float, float],
    max_width: int,
    align: bool,
    min_persistence: int | None,
) -> np.ndarray:
    # Decode
    decoded = [_decode_jpeg(f.data) for f in frames]
    decoded = [_apply_crop(d, crop) for d in decoded]
    decoded = [_resize_width(d, max_width) for d in decoded]
    N = len(decoded)
    bh, bw = decoded[0].shape
    board_frames = np.stack(decoded, axis=0).astype(np.float32)

    # --- 1. Optional FFT alignment ---
    if align:
        board_frames = _fft_align(board_frames)

    # --- 2. Background (30th percentile) ---
    bg = np.percentile(board_frames, 30, axis=0).astype(np.float32)

    # --- 3. Temporal activity ---
    temp_diff = np.abs(board_frames[1:] - board_frames[:-1])
    temp_diff = np.pad(temp_diff, ((1, 0), (0, 0), (0, 0)), mode="edge")
    d_pad = np.pad(temp_diff, ((2, 2), (0, 0), (0, 0)), mode="edge")
    mov_activity = (
        d_pad[0:-4] + d_pad[1:-3] + d_pad[2:-2] + d_pad[3:-1] + d_pad[4:]
    ) / 5.0
    is_static = mov_activity < 12.0

    # --- 4. Spatial gradient ---
    grad_y = np.abs(board_frames[:, 1:, :] - board_frames[:, :-1, :])
    grad_y = np.pad(grad_y, ((0, 0), (1, 0), (0, 0)))
    grad_x = np.abs(board_frames[:, :, 1:] - board_frames[:, :, :-1])
    grad_x = np.pad(grad_x, ((0, 0), (0, 0), (1, 0)))
    spatial_grad = grad_y + grad_x

    # --- 5. Chalk candidate mask ---
    is_chalk = (board_frames - bg > 22.0) & (spatial_grad > 14.0) & is_static

    # --- 6. Temporal persistence & monotonic edge ---
    persistence = min_persistence if min_persistence is not None else 5
    total_count = np.sum(is_chalk, axis=0)
    cond_persistent = total_count >= persistence
    cond_fresh_write = is_chalk[-1] & is_chalk[-2] & is_chalk[-3]
    cond_fresh_erase = is_chalk[0] & is_chalk[1] & is_chalk[2]
    valid_chalk = cond_persistent | cond_fresh_write | cond_fresh_erase

    # --- 7. robust median fusion ---
    # np.nanmedian is very slow on full-resolution stacks; sorting once and
    # taking the two middle valid values gives exactly the same result.
    counts = np.sum(is_chalk, axis=0)
    chalk_vals = np.where(is_chalk, board_frames, np.nan)
    ordered = np.sort(chalk_vals, axis=0)
    lo = np.maximum(0, (counts - 1) // 2)
    hi = np.maximum(0, counts // 2)
    low = np.take_along_axis(ordered, lo[None, ...], axis=0)[0]
    high = np.take_along_axis(ordered, hi[None, ...], axis=0)[0]
    chalk_robust = (low + high) / 2.0
    chalk_robust = np.where(counts == 0, np.nan, chalk_robust)
    chalk_robust = np.where(np.isnan(chalk_robust), bg, chalk_robust)

    fused = np.where(valid_chalk, np.maximum(bg, chalk_robust), bg).astype(np.uint8)
    return fused


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to a grayscale numpy array."""
    img = Image.open(io_bytes(data))
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8)


def _apply_crop(
    arr: np.ndarray, crop: tuple[float, float, float, float]
) -> np.ndarray:
    x0, y0, x1, y1 = crop
    h, w = arr.shape
    cx0 = max(0, int(round(x0 * w)))
    cy0 = max(0, int(round(y0 * h)))
    cx1 = min(w, int(round(x1 * w)))
    cy1 = min(h, int(round(y1 * h)))
    if cx0 >= cx1 or cy0 >= cy1:
        return arr  # degenerate crop – keep full frame
    return arr[cy0:cy1, cx0:cx1]


def _resize_width(arr: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return arr
    w = arr.shape[1]
    if w <= max_width:
        return arr
    ratio = max_width / w
    new_h = max(1, int(round(arr.shape[0] * ratio)))
    img = Image.fromarray(arr)
    img = img.resize((max_width, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _fft_align(frames: np.ndarray, *, estimate_width: int = 640) -> np.ndarray:
    """Register every frame to the middle frame via FFT phase correlation.

    ``frames``: (N, H, W) float32.

    The displacement is estimated on a smaller copy (at most
    ``estimate_width`` pixels wide) to keep real-time windows affordable; the
    returned frames, and therefore the fused output, stay at the original
    resolution.  This only affects motion estimation, not the image fed to the
    VLM.
    """
    N, bh, bw = frames.shape
    ref_idx = N // 2
    step = max(1, int(np.ceil(bw / max(1, estimate_width))))
    if step > 1:
        small = frames[:, ::step, ::step]
    else:
        small = frames
    _n, ah, aw = small.shape
    ref = small[ref_idx].astype(np.float32)
    win = np.outer(np.hanning(ah), np.hanning(aw)).astype(np.float32)
    F_ref = np.fft.fft2(ref * win)

    aligned = np.empty_like(frames)
    for i in range(N):
        F_cur = np.fft.fft2(small[i].astype(np.float32) * win)
        cp = (F_ref * np.conj(F_cur)) / (np.abs(F_ref * np.conj(F_cur)) + 1e-12)
        r = np.fft.ifft2(cp).real
        r = np.fft.fftshift(r)
        py, px = np.unravel_index(np.argmax(r), r.shape)
        dy = int(round((py - ah // 2) * (bh / ah)))
        dx = int(round((px - aw // 2) * (bw / aw)))
        aligned[i] = np.roll(frames[i], shift=(dy, dx), axis=(0, 1))

    return aligned


# ---------------------------------------------------------------------------
# Small BytesIO wrapper (avoids top-level PIL import side-effects)
# ---------------------------------------------------------------------------
import io as _io


def io_bytes(data: bytes):
    return _io.BytesIO(data)