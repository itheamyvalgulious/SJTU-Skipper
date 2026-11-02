"""VLM (Vision-Language Model) blackboard recognition plugin.

Reuses the picture_parser fusion pipeline adapted for real-time sliding
windows: decode JPEG at full resolution, optional FFT alignment, temporal
persistence filtering, nanmedian fusion, then send to a VLM for LaTeX
recognition.
"""

from __future__ import annotations

from .plugin import VlmPlugin

__all__ = ["VlmPlugin"]