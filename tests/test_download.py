"""Tests for the recording downloader (local HTTP server, no network)."""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest

from plugins.input.download import MediaDownloader


class _Handler(http.server.BaseHTTPRequestHandler):
    payload = b"0123456789" * 1024
    fail = False

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.fail:
            self.send_response(500)
            self.end_headers()
            return
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            start_text, _, end_text = header[6:].partition("-")
            start = int(start_text or 0)
            end = int(end_text) if end_text else len(self.payload) - 1
            end = min(end, len(self.payload) - 1)
            body = self.payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(self.payload)}"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args: object) -> None:  # silence the test log
        return


@pytest.fixture()
def http_base():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_downloads_to_final_path(http_base: str, tmp_path: Path) -> None:
    progress: list[tuple[int, int]] = []
    downloader = MediaDownloader()
    result = downloader.download(
        f"{http_base}/lecture.mp4",
        tmp_path / "media" / "lecture.mp4",
        on_progress=lambda received, total: progress.append((received, total)),
    )
    assert result.ok is True
    assert result.path == tmp_path / "media" / "lecture.mp4"
    assert result.path.read_bytes() == _Handler.payload
    assert not (tmp_path / "media" / "lecture.mp4.part").exists()
    assert progress and progress[-1][0] == len(_Handler.payload)


def test_failure_keeps_no_partial_file(http_base: str, tmp_path: Path) -> None:
    _Handler.fail = True
    try:
        result = MediaDownloader().download(
            f"{http_base}/missing.mp4", tmp_path / "media" / "x.mp4"
        )
    finally:
        _Handler.fail = False
    assert result.ok is False
    assert not (tmp_path / "media" / "x.mp4").exists()
    assert not (tmp_path / "media" / "x.mp4.part").exists()


def test_cancel_before_download(tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()
    downloader = MediaDownloader()
    result = downloader.download(
        "http://127.0.0.1:9/never.mp4", tmp_path / "x.mp4", stop_event=stop
    )
    assert result.ok is False


def test_large_file_uses_parallel_ranges(http_base: str, tmp_path: Path) -> None:
    from plugins.input import download as download_mod

    payload = bytes(range(256)) * (200 * 1024)  # ~51 MB
    _Handler.payload = payload
    original_chunk = download_mod._CHUNK_SIZE
    download_mod._CHUNK_SIZE = 4 * 1024 * 1024
    try:
        result = MediaDownloader().download(
            f"{http_base}/big.mp4", tmp_path / "big.mp4"
        )
    finally:
        download_mod._CHUNK_SIZE = original_chunk
        _Handler.payload = b"0123456789" * 1024
    assert result.ok is True
    assert result.path.read_bytes() == payload
    assert not list(tmp_path.glob("big.mp4*chunk*"))
