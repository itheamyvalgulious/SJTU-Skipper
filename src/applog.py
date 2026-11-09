"""Logging helpers."""

from __future__ import annotations

import faulthandler
import logging
import signal
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure a compact, idempotent console logger."""

    global _CONFIGURED
    # ``kill -USR1 <pid>`` dumps every thread's stack: the quickest way to see
    # what a hung STT/VLM worker is waiting on.
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:  # noqa: BLE001 - not supported on every platform
        pass
    root = logging.getLogger("lecture_agent")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "lecture_agent") -> logging.Logger:
    return logging.getLogger(name)
