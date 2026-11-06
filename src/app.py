"""Application factory helpers."""

from __future__ import annotations

import logging
from typing import Any

from config import AppConfig, SHUTDOWN_TIMEOUT, load_config
from core.runtime import Runtime, build_runtime
from plugins.builtin import builtin_plugins

logger = logging.getLogger("lecture_agent.app")


def build_default_runtime(config: AppConfig | None = None) -> Runtime:
    config = (config or load_config()).resolve()
    plugins = builtin_plugins(config)
    return build_runtime(config, plugins=plugins)


def create_app(runtime: Runtime | None = None, *, config: AppConfig | None = None):
    from ui.server import create_app as _create_app

    if runtime is None:
        runtime = build_default_runtime(config)
    return _create_app(runtime)


def run_server(
    *,
    config: AppConfig | None = None,
    runtime: Runtime | None = None,
) -> None:
    import uvicorn

    config = (config or load_config()).resolve()
    if runtime is None:
        runtime = build_default_runtime(config)
    app = create_app(runtime)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        timeout_graceful_shutdown=SHUTDOWN_TIMEOUT,
    )


__all__ = [
    "Runtime",
    "build_default_runtime",
    "create_app",
    "run_server",
]
