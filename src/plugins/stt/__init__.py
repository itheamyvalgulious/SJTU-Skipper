"""STT plugin package.

Public entry points:

* :class:`~plugins.stt.plugin.SttPlugin` — the plugin itself.
* :class:`~plugins.stt.backends.STTBackend` — backend protocol.
* :func:`~plugins.stt.audio.compute_windows` — chunk arithmetic.
"""

from __future__ import annotations

from .audio import Window, compute_windows, dedupe_overlap, pcm_to_wav, wav_duration
from .backends import (
    PROVIDERS,
    FasterWhisperBackend,
    OpenAITranscriptionBackend,
    STTBackend,
    STTBackendError,
    STTError,
    STTResult,
    STTSegment,
    STTUnavailableError,
    create_backend,
    transcription_endpoint,
)
from .plugin import STTConfig, STTPlugin, STTService, SttPlugin

__all__ = [
    "PROVIDERS",
    "FasterWhisperBackend",
    "OpenAITranscriptionBackend",
    "STTBackend",
    "STTBackendError",
    "STTConfig",
    "STTError",
    "STTPlugin",
    "STTResult",
    "STTService",
    "STTSegment",
    "STTUnavailableError",
    "SttPlugin",
    "Window",
    "compute_windows",
    "create_backend",
    "dedupe_overlap",
    "pcm_to_wav",
    "transcription_endpoint",
    "wav_duration",
]
