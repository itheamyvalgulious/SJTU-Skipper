"""Core framework: streams, events, plugins, services and UI."""

from __future__ import annotations

from .events import (
    CLOCK_TICK,
    EVENT_ERROR,
    SESSION_STARTED,
    SESSION_STOPPED,
    STREAM_APPENDED,
    Clock,
    Event,
    EventBus,
    EventType,
    Subscription,
)
from .meta import MetaRegistry
from .plugin import Plugin, PluginContext, PluginError, PluginManager
from .runtime import LLMProxy, Runtime, build_runtime
from .services import ServiceError, ServiceNotFound, ServiceRegistry
from .streams import (
    AudioChunk,
    AudioStream,
    SessionInfo,
    StreamHub,
    StreamSet,
    TextItem,
    TextStream,
    VideoFrame,
    VideoStream,
)
from .ui import ActionResult, PageHandle, TabHandle, UIManager, render_markdown

__all__ = [
    "ActionResult",
    "AudioChunk",
    "AudioStream",
    "CLOCK_TICK",
    "Clock",
    "EVENT_ERROR",
    "Event",
    "EventBus",
    "EventType",
    "LLMProxy",
    "MetaRegistry",
    "PageHandle",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginManager",
    "Runtime",
    "SESSION_STARTED",
    "SESSION_STOPPED",
    "STREAM_APPENDED",
    "ServiceError",
    "ServiceNotFound",
    "ServiceRegistry",
    "SessionInfo",
    "StreamHub",
    "StreamSet",
    "Subscription",
    "TabHandle",
    "TextItem",
    "TextStream",
    "UIManager",
    "VideoFrame",
    "VideoStream",
    "build_runtime",
    "render_markdown",
]
