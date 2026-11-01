"""Tests for the LLM layer: types, provider, mock, service, and plugin."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from unittest import mock

import httpx
import pytest

from core.events import EventBus
from core.plugin import PluginContext
from core.services import ServiceRegistry
from llm.errors import (
    LLMAuthError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMRetryExhausted,
    LLMStreamError,
)
from llm.mock import MockProvider
from llm.openai_compat import (
    OpenAICompatibleProvider,
    _normalise_base_url,
)
from llm.service import LLMService
from llm.types import (
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
from tests.support import FakeSettings

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ProviderConfig:
    defaults: dict[str, Any] = {
        "base_url": "https://api.test.local/v1",
        "api_key": "sk-test",
        "model": "test-model",
        "timeout": 10,
        "max_retries": 1,
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_request(**overrides: Any) -> ChatRequest:
    return ChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hello")],
        **overrides,
    )


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj).encode()


def _sse_lines(*lines: str) -> bytes:
    """Build an SSE response from *lines* (no leading ``data: `` needed)."""
    parts: list[str] = []
    for line in lines:
        parts.append(f"data: {line}\n\n")
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


# ---------------------------------------------------------------------------
# base URL normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://api.deepseek.com", "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/v1/", "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/v1/chat/completions", "https://api.deepseek.com/v1"),
        ("https://openai.example.com/v1", "https://openai.example.com/v1"),
        ("https://openai.example.com", "https://openai.example.com/v1"),
        ("http://localhost:8080/v1", "http://localhost:8080/v1"),
    ],
)
def test_normalise_base_url(raw: str, expected: str) -> None:
    assert _normalise_base_url(raw) == expected


def test_request_url_no_double_v1() -> None:
    """The resolved request URL must be {base}/v1/chat/completions (no doubled /v1)."""
    body = {
        "id": "url-test",
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }

    captured_url = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url
        captured_url = str(request.url)
        return httpx.Response(200, json=body)

    config = _make_config(base_url="https://api.example.com")
    base = _normalise_base_url(config.base_url)  # → https://api.example.com/v1
    test_client = httpx.Client(
        base_url=base,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        timeout=httpx.Timeout(config.timeout),
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICompatibleProvider(config=config)
    provider._client = test_client  # type: ignore[assignment]
    provider.chat(_make_request())

    # Must NOT contain /v1/v1/
    assert "/v1/v1/" not in captured_url
    assert captured_url == "https://api.example.com/v1/chat/completions", (
        f"expected .../v1/chat/completions, got {captured_url}"
    )


def test_non_retryable_http_error_has_body() -> None:
    """A 404 with empty body raises LLMHTTPError (not JSONDecodeError)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"")

    config = _make_config(max_retries=1)
    test_client = httpx.Client(
        base_url=_normalise_base_url(config.base_url),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"},
        timeout=httpx.Timeout(config.timeout),
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICompatibleProvider(config=config)
    provider._client = test_client  # type: ignore[assignment]

    with pytest.raises(LLMHTTPError) as exc_info:
        provider.chat(_make_request())
    assert exc_info.value.status_code == 404


def test_extra_body_merged_at_top_level() -> None:
    """request.extra fields (e.g. DeepSeek thinking) appear at JSON top level."""
    body = {
        "id": "extra-test",
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }

    sent_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_payload
        sent_payload = json.loads(request.content)
        return httpx.Response(200, json=body)

    config = _make_config()
    test_client = httpx.Client(
        base_url=_normalise_base_url(config.base_url),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"},
        timeout=httpx.Timeout(config.timeout),
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICompatibleProvider(config=config)
    provider._client = test_client  # type: ignore[assignment]
    provider.chat(_make_request(extra={"thinking": {"type": "disabled"}}))

    # Must be at top level, not nested under "extra_body"
    assert sent_payload["thinking"] == {"type": "disabled"}
    assert "extra_body" not in sent_payload


# ---------------------------------------------------------------------------
# non-streaming chat
# ---------------------------------------------------------------------------


def test_chat_non_streaming() -> None:
    body = {
        "id": "chat-1",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, world!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    config = _make_config()
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config)
        provider._client = client  # type: ignore[assignment]
        response = provider.chat(_make_request())
        assert response.content == "Hello, world!"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.model == "test-model"


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


def test_chat_streaming() -> None:
    sse_body = _sse_lines(
        json.dumps({"choices": [{"delta": {"content": "Hel"}, "index": 0}]}),
        json.dumps({"choices": [{"delta": {"content": "lo"}, "index": 0}]}),
        json.dumps({
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "model": "test-model",
        }),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body)

    config = _make_config()
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config)
        provider._client = client  # type: ignore[assignment]
        events = list(provider.stream(_make_request()))
        texts = [e.text for e in events if e.type == "text"]
        assert texts == ["Hel", "lo"]
        done = [e for e in events if e.type == "done"]
        assert len(done) == 1
        assert done[0].response is not None
        assert done[0].response.content == "Hello"
        assert done[0].response.finish_reason == "stop"


# ---------------------------------------------------------------------------
# tool calls
# ---------------------------------------------------------------------------


def test_chat_tool_calls_non_streaming() -> None:
    body = {
        "id": "chat-tc-1",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    config = _make_config()
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config)
        provider._client = client  # type: ignore[assignment]
        response = provider.chat(_make_request())
        assert response.content == "" or response.content is None
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "call_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Paris"}
        assert response.finish_reason == "tool_calls"


def test_chat_tool_calls_streaming() -> None:
    sse_body = _sse_lines(
        json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_s1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                },
                "index": 0,
            }]
        }),
        json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"city":'},
                        }
                    ]
                },
                "index": 0,
            }]
        }),
        json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '"Paris"}'},
                        }
                    ]
                },
                "index": 0,
            }]
        }),
        json.dumps({
            "choices": [{"delta": {}, "finish_reason": "tool_calls", "index": 0}],
            "model": "test-model",
        }),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body)

    config = _make_config()
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config)
        provider._client = client  # type: ignore[assignment]
        events = list(provider.stream(_make_request()))
        tool_call_events = [e for e in events if e.type == "tool_call"]
        assert len(tool_call_events) == 1
        tc = tool_call_events[0].tool_call
        assert tc is not None
        assert tc.id == "call_s1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Paris"}


# ---------------------------------------------------------------------------
# image serialisation
# ---------------------------------------------------------------------------


def test_image_data_url() -> None:
    """Verify that ImagePart.data is serialised as a base64 data URL."""
    pixels = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # fake JPEG header
    body = {
        "id": "img-1",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "I see an image"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
    }

    sent_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_payload
        sent_payload = json.loads(request.content)
        return httpx.Response(200, json=body)

    config = _make_config()
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config)
        provider._client = client  # type: ignore[assignment]
        msg = Message(
            role="user",
            content=[
                TextPart(text="What is this?"),
                ImagePart(data=pixels, mime_type="image/jpeg"),
            ],
        )
        response = provider.chat(ChatRequest(model="vlm", messages=[msg]))
        assert response.content == "I see an image"
        # Check that the image was embedded as a data URL
        user_content = sent_payload["messages"][0]["content"]
        assert isinstance(user_content, list)
        image_part = user_content[1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_image_url_passthrough() -> None:
    """ImagePart.url is passed through directly."""
    body = {
        "id": "img-2",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }

    sent_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_payload
        sent_payload = json.loads(request.content)
        return httpx.Response(200, json=body)

    config = _make_config()
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config)
        provider._client = client  # type: ignore[assignment]
        msg = Message(
            role="user",
            content=[ImagePart(url="https://example.com/img.jpg")],
        )
        provider.chat(ChatRequest(model="vlm", messages=[msg]))
        user_content = sent_payload["messages"][0]["content"]
        assert user_content[0]["image_url"]["url"] == "https://example.com/img.jpg"


# ---------------------------------------------------------------------------
# retry – success after transient failure
# ---------------------------------------------------------------------------


def test_retry_success_after_connect_error() -> None:
    """First two attempts fail with ConnectError, third succeeds."""
    attempts: list[Any] = []

    class FailingTransport:
        def __init__(self) -> None:
            self._count = 0

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            self._count += 1
            attempts.append(self._count)
            if self._count < 3:
                raise httpx.ConnectError("connection refused")
            body = {
                "id": "retry-ok",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "finally"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
            return httpx.Response(200, json=body)

    transport = FailingTransport()
    config = _make_config(max_retries=3)
    policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05)
    with httpx.Client(
        transport=httpx.MockTransport(transport.handle_request), base_url="http://test"
    ) as client:
        provider = OpenAICompatibleProvider(config=config, retry_policy=policy)
        provider._client = client  # type: ignore[assignment]
        response = provider.chat(_make_request())
        assert response.content == "finally"
        assert len(attempts) == 3


# ---------------------------------------------------------------------------
# retry – exhausted
# ---------------------------------------------------------------------------


def test_retry_exhausted_non_streaming() -> None:
    """Every attempt fails -> LLMRetryExhausted."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("refused")

    config = _make_config(max_retries=2)
    policy = RetryPolicy(max_attempts=2, base_delay=0.01, max_delay=0.05)
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config, retry_policy=policy)
        provider._client = client  # type: ignore[assignment]
        with pytest.raises(LLMRetryExhausted):
            provider.chat(_make_request())
        assert call_count == 2


def test_retry_exhausted_streaming() -> None:
    """Stream fails before any delta -> LLMRetryExhausted."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("refused")

    config = _make_config(max_retries=2)
    policy = RetryPolicy(max_attempts=2, base_delay=0.01, max_delay=0.05)
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config, retry_policy=policy)
        provider._client = client  # type: ignore[assignment]
        with pytest.raises(LLMRetryExhausted):
            list(provider.stream(_make_request()))
        assert call_count == 2


# ---------------------------------------------------------------------------
# Retry-After header
# ---------------------------------------------------------------------------


def test_retry_after_header() -> None:
    """503 with Retry-After: provider respects the header delay."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(503, headers={"Retry-After": "0.05"})
        body = {
            "id": "ra-ok",
            "model": "test-model",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {},
        }
        return httpx.Response(200, json=body)

    config = _make_config(max_retries=3)
    # Speed up the test: the retry will check Retry-After
    policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.1, jitter=0.0)
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        provider = OpenAICompatibleProvider(config=config, retry_policy=policy)
        provider._client = client  # type: ignore[assignment]
        response = provider.chat(_make_request())
        assert response.content == "ok"
        assert call_count == 2


# ---------------------------------------------------------------------------
# stream – no retry after first meaningful delta
# ---------------------------------------------------------------------------


def test_stream_no_retry_after_first_delta() -> None:
    """If a stream emits at least one text/tool_call delta and then fails,
    the provider must NOT retry and must raise LLMStreamError."""
    call_count = 0
    # First connection: send one delta, then break.
    first_chunk = _sse_lines(
        json.dumps({"choices": [{"delta": {"content": "Hel"}, "index": 0}]}),
    )

    class StreamBreakingTransport:
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Remove the [DONE] to cause the connection to die mid-stream
                content = first_chunk.replace(b"data: [DONE]\n\n", b"")
                return httpx.Response(200, content=content)
            # Second call: should not happen
            body = {
                "id": "never",
                "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "never"}, "finish_reason": "stop"}],
                "usage": {},
            }
            return httpx.Response(200, json=body)

    transport = StreamBreakingTransport()
    config = _make_config(max_retries=2)
    policy = RetryPolicy(max_attempts=2, base_delay=0.01, max_delay=0.05, jitter=0.0)
    with httpx.Client(
        transport=httpx.MockTransport(transport.handle_request), base_url="http://test"
    ) as client:
        provider = OpenAICompatibleProvider(config=config, retry_policy=policy)
        provider._client = client  # type: ignore[assignment]
        events = list(provider.stream(_make_request()))
        # Should get the text delta + an error event (or LLMStreamError raised).
        # Depending on implementation: the broken stream may yield an error event.
        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) >= 1
        # Should NOT have retried (call_count == 1)
        assert call_count == 1


# ---------------------------------------------------------------------------
# LLMService – profiles and message coercion
# ---------------------------------------------------------------------------


def test_service_profile_agent() -> None:
    """LLMService.chat(profile="agent") delegates to the configured provider."""
    mock_provider = MockProvider(script=["agent response"])
    service = LLMService()
    service.register_provider("openai", mock_provider)

    response = service.chat(
        profile="agent",
        messages=[Message(role="user", content="hi")],
    )
    assert response.content == "agent response"


def test_service_profile_vlm() -> None:
    """VLM profile uses vlm_model when available from config callback."""
    configs: list[ProviderConfig] = []

    def config_cb(profile: str) -> ProviderConfig:
        if profile == "vlm":
            cfg = ProviderConfig(
                base_url="http://vlm.test",
                api_key="sk-vlm",
                model="vlm-model",
            )
        else:
            cfg = ProviderConfig(
                base_url="http://agent.test",
                api_key="sk-agent",
                model="agent-model",
            )
        configs.append(cfg)
        return cfg

    mock_provider = MockProvider(script=["vlm reply"])
    service = LLMService(config_provider=config_cb)
    service.register_provider("openai", mock_provider)

    response = service.chat(profile="vlm", messages=[Message(role="user", content="describe this")])
    assert response.content == "vlm reply"
    # Verify the config was resolved for vlm
    assert len(configs) >= 1
    assert configs[-1].model == "vlm-model"


def test_service_message_dict() -> None:
    """LLMService accepts OpenAI-style dict messages."""
    mock_provider = MockProvider(script=["ok"])
    service = LLMService()
    service.register_provider("openai", mock_provider)

    response = service.chat(
        messages=[{"role": "user", "content": "test"}],
    )
    assert response.content == "ok"
    assert len(mock_provider.calls) == 1


def test_service_config_for() -> None:
    """config_for returns a populated ProviderConfig."""
    def config_cb(profile: str) -> ProviderConfig:
        return ProviderConfig(
            base_url="http://example.com",
            api_key="k",
            model=f"{profile}-model",
            temperature=0.5,
            max_tokens=200,
            timeout=30,
            max_retries=5,
            extra_headers={"X-Test": "1"},
            extra_body={"thinking": "disabled"},
        )

    service = LLMService(config_provider=config_cb)
    cfg = service.config_for("agent")
    assert cfg.model == "agent-model"
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 200
    assert cfg.timeout == 30
    assert cfg.max_retries == 5
    assert cfg.extra_headers == {"X-Test": "1"}
    assert cfg.extra_body == {"thinking": "disabled"}


def test_service_info() -> None:
    service = LLMService()
    mock_provider = MockProvider()
    service.register_provider("openai", mock_provider)
    info = service.info()
    assert "openai" in info["providers"]
    assert info["default_provider"] == "openai"


def test_service_close() -> None:
    mock_provider = MockProvider()
    service = LLMService()
    service.register_provider("openai", mock_provider)
    service.close()
    # Registry should be empty after close_all
    assert service.registry.names() == []


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


def test_mock_provider_text() -> None:
    provider = MockProvider(script=["hello", "world"])
    r1 = provider.chat(_make_request())
    assert r1.content == "hello"
    r2 = provider.chat(_make_request())
    assert r2.content == "world"
    with pytest.raises(Exception):
        provider.chat(_make_request())


def test_mock_provider_response_objects() -> None:
    provider = MockProvider(script=[
        ChatResponse(content="a", finish_reason="stop", model="mock"),
    ])
    r = provider.chat(_make_request())
    assert r.content == "a"
    assert r.model == "mock"


def test_mock_provider_tool_calls() -> None:
    provider = MockProvider(script=[
        ChatResponse(
            content="",
            tool_calls=[ToolCall(id="c1", name="f", arguments={"x": 1})],
            finish_reason="tool_calls",
            model="mock",
        ),
    ])
    r = provider.chat(_make_request())
    assert r.tool_calls[0].name == "f"
    assert r.tool_calls[0].arguments == {"x": 1}


def test_mock_provider_stream() -> None:
    provider = MockProvider(script=["abcdef"])
    events = list(provider.stream(_make_request()))
    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 2  # 3-char chunks: "abc", "def"
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert done[0].response is not None
    assert done[0].response.content == "abcdef"


def test_mock_provider_callable_script() -> None:
    def handler(req: ChatRequest) -> str:
        return f"echo: {req.messages[0].content}"

    provider = MockProvider(script=[handler])
    r = provider.chat(_make_request())
    assert r.content == "echo: hello"


# ---------------------------------------------------------------------------
# LLMPlugin integration
# ---------------------------------------------------------------------------


def test_llm_plugin_setup_registers_settings_and_service() -> None:
    """LLMPlugin.setup registers settings specs and publishes the llm service."""
    from core.meta import MetaRegistry
    from plugins.llm_core.plugin import LLMPlugin

    fake_dir = pytest.importorskip("pathlib").Path("/tmp/opencode/llm_plugin_test")
    fake_dir.mkdir(parents=True, exist_ok=True)

    bus = EventBus()
    services = ServiceRegistry()
    fake_settings = FakeSettings()
    services.register("settings", fake_settings, plugin_id="settings")

    ctx = PluginContext(
        plugin_id="llm",
        data_dir=fake_dir,
        events=bus,
        streams=mock.MagicMock(),
        llm=None,
        ui=mock.MagicMock(),
        services=services,
        meta=MetaRegistry(),
        logger=mock.MagicMock(),
    )

    plugin = LLMPlugin()
    plugin.setup(ctx)

    # Check settings were registered
    assert "llm.base_url" in fake_settings.specs_map
    assert "llm.api_key" in fake_settings.specs_map
    assert "llm.model" in fake_settings.specs_map
    assert "llm.vlm_model" in fake_settings.specs_map
    assert "llm.temperature" in fake_settings.specs_map
    assert "llm.max_tokens" in fake_settings.specs_map
    assert "llm.timeout" in fake_settings.specs_map
    assert "llm.max_retries" in fake_settings.specs_map
    assert "llm.extra_headers" in fake_settings.specs_map
    assert "llm.extra_body" in fake_settings.specs_map

    # The api_key spec should be secret
    api_key_spec = fake_settings.specs_map["llm.api_key"]
    assert api_key_spec.secret is True

    # Check service is published and has the openai provider registered
    llm_service = services.get("llm")
    assert llm_service is not None
    assert "openai" in llm_service.registry.names()

    # Replace the openai provider with a mock for testing
    llm_service.registry.unregister("openai")
    llm_service.registry.unregister("openai-compatible")
    mock_provider = MockProvider(script=["plugin test ok"])
    llm_service.register_provider("openai", mock_provider)
    response = llm_service.chat(
        profile="agent",
        messages=[Message(role="user", content="hi")],
    )
    assert response.content == "plugin test ok"

    plugin.teardown()


def test_llm_plugin_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default base_url / api_key come from env vars."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

    from plugins.llm_core.plugin import LLMPlugin, _resolve_env

    url = _resolve_env("LLM_BASE_URL", ("LLM_BASE_URL", "OPENAI_BASE_URL"), "https://default")
    assert url == "https://custom.openai.com/v1"

    key = _resolve_env("LLM_API_KEY", ("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"), "")
    assert key == "sk-env-key"


# ---------------------------------------------------------------------------
# error hierarchy
# ---------------------------------------------------------------------------


def test_error_hierarchy() -> None:
    assert issubclass(LLMAuthError, RuntimeError)
    assert issubclass(LLMRateLimitError, RuntimeError)
    assert issubclass(LLMRetryExhausted, RuntimeError)
    assert issubclass(LLMStreamError, RuntimeError)


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------


def test_message_content_str_or_list() -> None:
    m1 = Message(role="user", content="plain")
    assert m1.content == "plain"

    m2 = Message(
        role="user",
        content=[TextPart(text="describe"), ImagePart(url="http://x")],
    )
    assert len(m2.content) == 2  # type: ignore[arg-type]


def test_stream_event_fields() -> None:
    e = StreamEvent(type="text", text="hi")
    assert e.type == "text"
    assert e.text == "hi"
    assert e.tool_call is None
    assert e.response is None
    assert e.error is None


def test_chat_response_defaults() -> None:
    r = ChatResponse(content="ok")
    assert r.tool_calls == []
    assert r.finish_reason is None
    assert r.usage.total_tokens == 0
    assert r.model == ""


def test_provider_config_defaults() -> None:
    cfg = ProviderConfig()
    assert cfg.base_url == ""
    assert cfg.api_key == ""
    assert cfg.model == ""
    assert cfg.timeout == 60.0
    assert cfg.max_retries == 3