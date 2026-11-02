"""STT placeholder plugin package.

There is currently no speech-recognition backend and no ``stt.*`` setting.
"""

from __future__ import annotations

from .plugin import STTPlugin, STTService, SttPlugin

__all__ = [
    "STTPlugin",
    "STTService",
    "SttPlugin",
]
