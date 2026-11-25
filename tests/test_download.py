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
        if self.path == "/sequential":
            header = None
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
            if self.path == "/changed":
                self.send_header(
                    "ETag", '"first"' if header == "bytes=0-0" else '"second"'
                )
            self.end_headers()
            if self.path == "/truncated" and header != "bytes=0-0":
                self.wfile.write(body[: max(1, len(body) // 2)])
                self.close_connection = True
                return
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args: object) -> None:
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


def test_reuses_complete_file_without_network(tmp_path: Path) -> None:
    dest = tmp_path / "lecture.mp4"
    dest.write_bytes(b"cached")
    progress: list[tuple[int, int]] = []
    downloader = MediaDownloader()

    handle = downloader.start(
        "http://127.0.0.1:1/unreachable.mp4",
        dest,
        on_progress=lambda received, total: progress.append((received, total)),
    )
    try:
        result = handle.wait()
    finally:
        handle.close()

    assert result.ok
    assert result.path == dest
    assert result.total == 6
    assert progress == [(6, 6)]


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
    payload = bytes(range(256)) * (200 * 1024)  # ~51 MB
    _Handler.payload = payload
    try:
        result = MediaDownloader().download(
            f"{http_base}/big.mp4", tmp_path / "big.mp4"
        )
    finally:
        _Handler.payload = b"0123456789" * 1024
    assert result.ok is True
    assert result.path.read_bytes() == payload
    assert not list(tmp_path.glob("big.mp4*chunk*"))


def test_cache_ranges(http_base: str, tmp_path: Path) -> None:
    import httpx2

    handle = MediaDownloader().start(
        f"{http_base}/lecture.mp4", tmp_path / "lecture.mp4"
    )
    try:
        with httpx2.Client() as client:
            response = client.get(handle.input_url, headers={"Range": "bytes=-17"})
            assert response.status_code == 206
            assert response.content == _Handler.payload[-17:]
            assert (
                response.headers["content-range"]
                == f"bytes {len(_Handler.payload) - 17}-{len(_Handler.payload) - 1}/{len(_Handler.payload)}"
            )
            for value in ("bytes=0-1,4-5", "bytes=-0", "bytes=900000-", "bytes=4-2"):
                assert (
                    client.get(handle.input_url, headers={"Range": value}).status_code
                    == 416
                )
            response = client.head(handle.input_url)
            assert response.status_code == 200
            assert int(response.headers["content-length"]) == len(_Handler.payload)
        assert handle.wait().ok
        assert handle.wait().ok
    finally:
        handle.close()
        handle.close()


def test_sequential_source(http_base: str, tmp_path: Path) -> None:
    result = MediaDownloader().download(
        f"{http_base}/sequential", tmp_path / "media.mp4"
    )
    assert result.ok
    assert result.path.read_bytes() == _Handler.payload


def test_truncated_source_fails(http_base: str, tmp_path: Path) -> None:
    result = MediaDownloader().download(
        f"{http_base}/truncated", tmp_path / "media.mp4"
    )
    assert not result.ok
    assert not (tmp_path / "media.mp4").exists()
    assert not list(tmp_path.glob("*.part*"))


def test_changed_resource_fails(http_base: str, tmp_path: Path) -> None:
    result = MediaDownloader().download(f"{http_base}/changed", tmp_path / "media.mp4")
    assert not result.ok
    assert "changed" in result.error
    assert not list(tmp_path.iterdir())
