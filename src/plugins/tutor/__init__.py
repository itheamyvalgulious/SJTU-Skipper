"""Tutor plugin: shared teaching preferences and prompt layers for ask/notes.

Public API:
- ``TutorPlugin``: id="tutor", requires=("settings","ask")
- ``TutorService``: the ``tutor`` service (preference block + snapshot)
- ``PreferenceStore`` / ``Preference`` / ``Proposal``: profile storage
"""

from __future__ import annotations

from .plugin import TutorPlugin, TutorService
from .store import Preference, PreferenceStore, Proposal

__all__ = [
    "Preference",
    "PreferenceStore",
    "Proposal",
    "TutorPlugin",
    "TutorService",
]
