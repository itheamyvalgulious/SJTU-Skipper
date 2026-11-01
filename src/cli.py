"""Command line entry point."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Sequence

from config import load_config
from applog import setup_logging


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=None, help="runtime data directory")
    parser.add_argument("--host", default=None, help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default 8765)")
    parser.add_argument("--config", type=Path, default=None, help="agent.toml path")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")


def _config_from_args(args: argparse.Namespace):
    cfg = load_config(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        config_file=args.config,
    )
    if args.log_level:
        cfg.log_level = args.log_level
    return cfg


def _load_extra_plugins(specs: Sequence[str]) -> list[object]:
    plugins = []
    for spec in specs:
        module_name, _, attr = spec.partition(":")
        if not module_name or not attr:
            raise SystemExit(f"invalid plugin spec {spec!r}; expected module:Class")
        module = importlib.import_module(module_name)
        plugin_cls = getattr(module, attr)
        plugins.append(plugin_cls())
    return plugins


def cmd_serve(args: argparse.Namespace) -> int:
    from app import build_default_runtime, create_app
    import uvicorn

    cfg = _config_from_args(args)
    runtime = build_default_runtime(cfg)
    if args.plugin:
        runtime.add_plugins(_load_extra_plugins(args.plugin))
    app = create_app(runtime)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level.lower())
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from app import build_demo_runtime, create_app
    import uvicorn

    cfg = _config_from_args(args)
    cfg.demo_mode = True
    runtime = build_demo_runtime(cfg)
    app = create_app(runtime)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level.lower())
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from app import build_default_runtime

    cfg = _config_from_args(args)
    runtime = build_default_runtime(cfg)
    runtime.start()
    try:
        print("plugin order:")
        for index, plugin_id in enumerate(runtime.plugins.order, start=1):
            print(f"  {index:2d}. {plugin_id}")
        print("loaded:", ", ".join(runtime.plugins.loaded))
        print("services:", ", ".join(runtime.services.names()) or "(none)")
        print("data_dir:", cfg.data_dir)
        print("ok")
    finally:
        runtime.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lecture-agent", description="课堂总 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the web UI with all plugins")
    _add_common(serve)
    serve.add_argument("--plugin", action="append", default=[], help="extra plugin module:Class")
    serve.set_defaults(func=cmd_serve)

    demo = sub.add_parser("demo", help="start the web UI with synthetic classroom content")
    _add_common(demo)
    demo.set_defaults(func=cmd_demo)

    check = sub.add_parser("check", help="load plugins and print the topological order")
    _add_common(check)
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level or "INFO")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("lecture_agent.cli").exception("command failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1
