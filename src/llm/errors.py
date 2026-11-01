"""LLM-layer error hierarchy."""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base for every LLM-related error."""


class LLMAuthError(LLMError):
    """Authentication or authorization failure (HTTP 401 / 403)."""


class LLMRateLimitError(LLMError):
    """Rate-limit response (HTTP 429)."""


class LLMTimeoutError(LLMError):
    """Request timed out before the provider responded."""


class LLMStreamError(LLMError):
    """The SSE stream failed mid-response after the first meaningful delta."""


class LLMHTTPError(LLMError):
    """Non-retryable HTTP error with status code and response body."""

    def __init__(self, status_code: int, body: str = "", message: str = "") -> None:
        super().__init__(message or f"LLM HTTP {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


class LLMRetryExhausted(LLMError):
    """Every retry attempt failed; the cause is chained via ``__cause__``."""