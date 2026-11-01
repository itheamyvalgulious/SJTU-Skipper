from __future__ import annotations

import time
from pathlib import Path

from app import build_demo_runtime
from config import load_config


def _wait_until(predicate, timeout: float = 20.0, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_demo_end_to_end(tmp_path: Path) -> None:
    cfg = load_config(data_dir=tmp_path)
    runtime = build_demo_runtime(cfg)
    runtime.start()
    try:
        state = runtime.ui.state()
        assert any(tab["id"] == "demo" for tab in state["left_tabs"])
        assert any(page["id"] == "demo:main" for page in state["pages"])

        result = runtime.ui.dispatch_action("demo.start", {})
        assert result.ok

        session_id = runtime.hub.current.session_id
        notes = tmp_path / "sessions" / session_id / "notes"
        assert _wait_until(
            lambda: (notes / "knowledge.md").exists()
            and "留数定理" in (notes / "knowledge.md").read_text(encoding="utf-8")
            and (notes / "gossip.md").exists()
            and "AI" in (notes / "gossip.md").read_text(encoding="utf-8")
            and (notes / "notice.md").exists()
            and "作业" in (notes / "notice.md").read_text(encoding="utf-8")
        ), "demo did not produce the three note files"
    finally:
        runtime.ui.dispatch_action("demo.stop", {})
        runtime.stop()
