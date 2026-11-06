"""Shared STT errors."""

from __future__ import annotations


class SttError(RuntimeError):
    """Raised when credentials, transport, or Google STT rejects a request."""


__all__ = ["SttError"]
