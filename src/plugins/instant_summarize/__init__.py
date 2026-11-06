"""Instant summarize plugin: incremental (real-time) lecture notes."""

from __future__ import annotations

from .plugin import (
    INSTANT_NOTE_FILES,
    INSTANT_NOTE_HEADERS,
    InstantSummarizePlugin,
)

__all__ = ["InstantSummarizePlugin", "INSTANT_NOTE_FILES", "INSTANT_NOTE_HEADERS"]
