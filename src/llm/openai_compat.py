"""OpenAI-compatible chat completions provider.

Handles base-url normalisation, image data-URL serialisation, non-streaming
requests, SSE streaming with incremental tool-call aggregation, and automatic
retries for transient failures.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from .errors import (
    LLMAuthError,
    LLMError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMRetryExhausted,
    LLMStreamError,
    LLMTimeoutError,
)
from .provider import Provider
from .types import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    Message,
    ProviderConfig,
    RetryPolicy,
    StreamEvent,
    TextPart,
    ToolCall,
    ToolDef,
    Usage,
)

logger = logging.getLogger("lecture_agent.llm.openai_compat")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.HTTPStatusError,
)

_MAX_LINE = 1_000_000  # SSE safety valve


def _normalise_base_url(raw: str) -> str:
    """Accept anything from a bare host to a full ``/chat/completions`` path.

    Returns a URL that ends with ``/v1`` (no trailing slash) so the caller
    appends ``/chat/completions``.
    """
    raw = raw.strip().rstrip("/")
    if raw.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")]
    if raw.endswith("/v1"):
        return raw
    # If it's a bare host (no /v1 path), append it.
    if "/v1" not in raw.split("/", 3)[-1]:
        raw = raw.rstrip("/") + "/v1"
    return raw


def _serialise_image(part: ImagePart) -> dict[str, Any]:
    """Turn an ``ImagePart`` into an OpenAI ``image_url`` block."""
    if part.data is not None:
        encoded = base64.b64encode(part.data).decode("ascii")
        url = f"data:{part.mime_type};base64,{encoded}"
    elif part.url is not None:
        url = part.url
    else:
        raise ValueError("ImagePart must have either `url` or `data`")
    return {"type": "image_url", "image_url": {"url": url, "detail": part.detail}}


def _serialise_message(msg: Message) -> dict[str, Any]:
    """Convert our ``Message`` to an OpenAI-compatible dict."""
    out: dict[str, Any] = {"role": msg.role}
    if isinstance(msg.content, str):
        out["content"] = msg.content
    else:
        parts: list[dict[str, Any]] = []
        for part in msg.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                parts.append(_serialise_image(part))
            else:
                parts.append({"type": "text", "text": str(part)})
        out["content"] = parts
    if msg.name:
        out["name"] = msg.name
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        out["tool_call_id"] = msg.tool_call_id
    return out


def _serialise_tool_defs(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _build_retry_delay(
    attempt: int, policy: RetryPolicy, retry_after: float | None
) -> float:
    """Return the sleep duration for retry *attempt* (0-indexed)."""
    if retry_after is not None:
        delay = float(retry_after)
    else:
        delay = min(policy.max_delay, policy.base_delay * (2**attempt))
    jitter = random.uniform(-policy.jitter, policy.jitter) * delay
    return max(0.0, delay + jitter)


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------


@dataclass
class OpenAICompatibleProvider:
    """OpenAI chat/completions client with retry and SSE streaming.

    Parameters
    ----------
    config : ProviderConfig
        Connection and identity parameters.
    retry_policy : RetryPolicy
        Controls automatic retries (default 3 attempts, 408/409/429/5xx).
    """

    config: ProviderConfig
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    name: str = "openai-compatible"
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = self._make_client(self.config)

    def _make_client(self, config: ProviderConfig) -> httpx.Client:
        base = _normalise_base_url(config.base_url)
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
        headers.update(config.extra_headers)
        return httpx.Client(
            base_url=base,
            headers=headers,
            timeout=httpx.Timeout(config.timeout),
        )

    def reconfigure(self, config: ProviderConfig) -> None:
        """Apply new settings (base URL/key/headers) without a restart."""

        if config == self.config:
            return
        old = self._client
        self.config = config
        self._client = self._make_client(config)
        try:
            old.close()
        except Exception:  # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request, stream=False)
        return self._retry(lambda: self._chat_once(payload, request))

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        payload = self._build_payload(request, stream=True)
        first_event_yielded = False
        last_exception: Exception | None = None

        for attempt in range(self.retry_policy.max_attempts):
            try:
                if attempt > 0 and first_event_yielded:
                    # Already started output — cannot restart.
                    raise LLMStreamError(
                        "stream started emitting before a retry could complete"
                    )
                last_exception = None
                response = None
                for event in self._stream_once(payload, request):
                    if not first_event_yielded and event.type not in ("done", "error"):
                        first_event_yielded = True
                    if event.type == "error":
                        last_exception = event.error
                    yield event

                # If we got here via normal completion, return.
                if last_exception is None:
                    return
                # Otherwise, the loop ended with an error on the last event.
                raise last_exception  # type: ignore[misc]

            except _RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
            except LLMAuthError:
                raise
            except (LLMRateLimitError, LLMStreamError):
                if first_event_yielded:
                    raise
                last_exception = last_exception  # already set

            if first_event_yielded:
                # Should not reach here for LLMStreamError, but be safe.
                break

            # Compute delay, respecting Retry-After if we have a response.
            retry_after = _parse_retry_after(response) if response is not None else None
            delay = _build_retry_delay(attempt, self.retry_policy, retry_after)
            logger.warning(
                "llm stream retry attempt %d/%d after %.2fs: %s",
                attempt + 1,
                self.retry_policy.max_attempts,
                delay,
                last_exception,
            )
            time.sleep(delay)

        raise LLMRetryExhausted(
            f"stream failed after {self.retry_policy.max_attempts} attempts"
        ) from last_exception

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _build_payload(
        self, request: ChatRequest, *, stream: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": [_serialise_message(m) for m in request.messages],
            "stream": stream,
        }
        if request.tools:
            payload["tools"] = _serialise_tool_defs(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        elif self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        elif self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        if request.response_format:
            payload["response_format"] = request.response_format
        extra_body = dict(self.config.extra_body)
        extra_body.update(request.extra)
        if extra_body:
            # ``extra`` is merged at the top level (e.g. DeepSeek's
            # ``thinking`` switch).  For compatibility, an explicit nested
            # ``extra_body`` key is merged as well.
            nested = extra_body.pop("extra_body", None)
            payload.update(extra_body)
            if isinstance(nested, dict):
                payload.update(nested)
        return payload

    def _chat_once(
        self, payload: dict[str, Any], _request: ChatRequest
    ) -> ChatResponse:
        response = self._client.post("/chat/completions", json=payload)
        if response.status_code in (401, 403):
            raise LLMAuthError(
                f"LLM auth failed ({response.status_code}): {response.text[:500]}"
            )
        if response.status_code == 429:
            raise LLMRateLimitError("LLM rate limited (429)")
        # 408, 409, and 5xx are retryable transient errors.
        if response.status_code in self.retry_policy.retry_statuses or response.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"retryable status {response.status_code}",
                request=response.request,
                response=response,
            )
        # Non-200 / non-retryable: read body safely, then raise.
        if response.status_code != 200:
            body = ""
            try:
                body = response.text
            except Exception:
                pass
            raise LLMHTTPError(
                status_code=response.status_code,
                body=body,
                message=f"LLM HTTP {response.status_code}: {body[:200]}",
            )
        body = response.json()
        return _parse_non_streaming_response(body)

    def _stream_once(
        self, payload: dict[str, Any], _request: ChatRequest
    ) -> Iterator[StreamEvent]:
        tool_call_acc: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        content_parts: list[str] = []
        model = ""

        with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code in (401, 403):
                raise LLMAuthError(
                    f"LLM auth failed ({resp.status_code}): {resp.text[:500]}"
                )
            if resp.status_code == 429:
                raise LLMRateLimitError("LLM rate limited (429)")
            if resp.status_code != 200:
                # Let the retry layer handle 5xx etc.
                resp.read()
                raise httpx.HTTPStatusError(
                    f"stream failed ({resp.status_code})",
                    request=resp.request,
                    response=resp,
                )
            try:
                for line in resp.iter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    model = chunk.get("model", model)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta", {})

                    # text
                    if delta.get("content"):
                        text = delta["content"]
                        content_parts.append(text)
                        yield StreamEvent(type="text", text=text)

                    # tool calls (incremental)
                    for tc_delta in (delta.get("tool_calls") or []):
                        idx = tc_delta.get("index", 0)
                        acc = tool_call_acc.setdefault(idx, {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        })
                        if "id" in tc_delta and tc_delta["id"]:
                            acc["id"] = tc_delta["id"]
                        func = tc_delta.get("function", {})
                        if "name" in func and func["name"]:
                            acc["name"] = func["name"]
                        if "arguments" in func:
                            acc["arguments"] += func["arguments"]

                # After the stream finishes, emit completed tool calls.
                for idx in sorted(tool_call_acc):
                    acc = tool_call_acc[idx]
                    try:
                        args: dict[str, Any] = (
                            json.loads(acc["arguments"]) if acc["arguments"] else {}
                        )
                    except json.JSONDecodeError:
                        args = {}
                    tc = ToolCall(id=acc["id"], name=acc["name"], arguments=args)
                    yield StreamEvent(type="tool_call", tool_call=tc)

                content = "".join(content_parts)
                tool_calls = [
                    ToolCall(
                        id=a["id"],
                        name=a["name"],
                        arguments=(
                            json.loads(a["arguments"])
                            if a["arguments"]
                            else {}
                        ),
                    )
                    for a in tool_call_acc.values()
                ]
                for a in tool_call_acc.values():
                    try:
                        a["arguments_parsed"] = json.loads(a["arguments"]) if a["arguments"] else {}
                    except json.JSONDecodeError:
                        a["arguments_parsed"] = {}
                usage_raw = {}
                response = ChatResponse(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    usage=Usage(),
                    raw={},
                    model=model,
                )
                yield StreamEvent(type="done", response=response)

            except (_RETRYABLE_EXCEPTIONS, LLMRateLimitError):
                raise
            except httpx.HTTPStatusError:
                raise
            except Exception as exc:
                yield StreamEvent(type="error", error=LLMStreamError(str(exc)))

    def _retry(self, fn: Any) -> ChatResponse:
        last_exception: Exception | None = None
        error_response: httpx.Response | None = None
        for attempt in range(self.retry_policy.max_attempts):
            try:
                return fn()
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exception = exc
                if isinstance(exc, httpx.HTTPStatusError):
                    error_response = exc.response
            except LLMAuthError:
                raise
            except LLMRateLimitError as exc:
                last_exception = exc

            if attempt + 1 >= self.retry_policy.max_attempts:
                break

            retry_after = _parse_retry_after(error_response) if error_response is not None else None
            delay = _build_retry_delay(attempt, self.retry_policy, retry_after)
            logger.warning(
                "llm retry attempt %d/%d after %.2fs: %s",
                attempt + 1,
                self.retry_policy.max_attempts,
                delay,
                last_exception,
            )
            time.sleep(delay)

        raise LLMRetryExhausted(
            f"request failed after {self.retry_policy.max_attempts} attempts"
        ) from last_exception


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------


def _parse_non_streaming_response(body: dict[str, Any]) -> ChatResponse:
    model = body.get("model", "")
    usage_raw = body.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
    )
    choices = body.get("choices", [])
    if not choices:
        return ChatResponse(content="", usage=usage, raw=body, model=model)

    choice = choices[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason")
    content = message.get("content") or ""
    tool_calls: list[ToolCall] = []
    for tc_raw in (message.get("tool_calls") or []):
        func = tc_raw.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args: dict[str, Any] = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(
            ToolCall(id=tc_raw.get("id", ""), name=func.get("name", ""), arguments=args)
        )
    return ChatResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        raw=body,
        model=model,
    )


def _parse_retry_after(response: httpx.Response | None) -> float | None:
    """Extract ``Retry-After`` seconds from response headers, if present."""
    if response is None:
        return None
    value = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # Could be a HTTP-date; skip for simplicity.
        return None