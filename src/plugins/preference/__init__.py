"""Shared student-preference memory plugin."""

from __future__ import annotations

from .plugin import PreferencePlugin, PreferenceService
from .store import Preference, PreferenceStore, Proposal

__all__ = [
    "Preference",
    "PreferencePlugin",
    "PreferenceService",
    "PreferenceStore",
    "Proposal",
]
