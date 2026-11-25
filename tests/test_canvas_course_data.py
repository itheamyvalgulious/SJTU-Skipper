from __future__ import annotations

from pathlib import Path

import httpx2

from integrations.sjtu_canvas_assistant.canvas_client import CanvasClient
from integrations.sjtu_canvas_assistant.models import (
    CanvasFile,
    Course,
)
from integrations.sjtu_canvas_assistant.path_utils import (
    canvas_file_path,
    course_dir_name,
    planned_file_path,
    safe_path_segment,
)


def _client(handler) -> CanvasClient:
    return CanvasClient(
        "https://oc.test",
        "token",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def test_course_files_keep_canvas_folder_layout(tmp_path: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["Authorization"] == "Bearer token"
        if request.url.path == "/api/v1/courses/1/folders":
            return httpx2.Response(
                200,
                json=[
                    {
                        "id": 7,
                        "name": "课件",
                        "full_name": "course files/课件",
                    }
                ],
            )
        if request.url.path == "/api/v1/courses/1/files":
            return httpx2.Response(
                200,
                json=[
                    {
                        "id": 11,
                        "folder_id": 7,
                        "display_name": "第 1 讲.pdf",
                        "url": "https://oc.test/files/11/download",
                        "size": 123,
                        "updated_at": "2026-09-01T00:00:00Z",
                    }
                ],
            )
        raise AssertionError(f"unexpected request: {request.url}")

    with _client(handler) as client:
        folders = client.list_course_folders(1)
        files = client.list_course_files(1)

    folder_map = {folder.id: folder for folder in folders}
    assert canvas_file_path(files[0], folder_map) == "课件/第 1 讲.pdf"
    course = Course(id=1, name="测试课程")
    assert planned_file_path(tmp_path, course, files[0], folder_map) == (
        tmp_path / "测试课程 [1]" / "课件" / "第 1 讲.pdf"
    )


def test_assignments_and_submission_parsing() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/v1/courses/1/assignments":
            return httpx2.Response(
                200,
                json=[
                    {
                        "id": 5,
                        "name": "作业一",
                        "due_at": "2026-10-08T15:59:59Z",
                        "points_possible": 100,
                        "submission_types": ["online_upload"],
                        "all_dates": [],
                    }
                ],
            )
        if request.url.path == "/api/v1/courses/1/assignments/5/submissions/self":
            return httpx2.Response(
                200,
                json={
                    "id": 9,
                    "assignment_id": 5,
                    "submitted_at": "2026-10-01T00:00:00Z",
                    "workflow_state": "graded",
                    "score": 95,
                    "submission_history": [],
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    with _client(handler) as client:
        assignments = client.list_assignments(1)
        records = client.list_my_submissions(1)

    assert assignments[0].due_at == "2026-10-08T15:59:59Z"
    assert records[0][0].name == "作业一"
    assert records[0][1].score == 95


def test_paginated_files_follow_link_header() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        page = request.url.params.get("page", "1")
        if page == "1":
            return httpx2.Response(
                200,
                json=[{"id": 1, "display_name": "a.txt"}],
                headers={
                    "Link": '<https://oc.test/api/v1/courses/1/files?page=2>; rel="next"'
                },
            )
        return httpx2.Response(200, json=[{"id": 2, "display_name": "b.txt"}])

    with _client(handler) as client:
        files = client.list_course_files(1)
    assert [file.effective_name for file in files] == ["a.txt", "b.txt"]


def test_download_file_streams_to_disk(tmp_path: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/files/11/download"
        return httpx2.Response(200, content=b"hello canvas")

    file = CanvasFile(
        id=11,
        display_name="讲义.txt",
        url="https://oc.test/files/11/download",
        size=len(b"hello canvas"),
    )
    destination = tmp_path / "讲义.txt"
    with _client(handler) as client:
        client.download_file(file, destination)
    assert destination.read_bytes() == b"hello canvas"
    assert not destination.with_name("讲义.txt.part").exists()


def test_safe_path_segment_strips_illegal_characters() -> None:
    assert safe_path_segment('a/b:c*?"<>|') == "a_b_c______"
    assert safe_path_segment("CON") == "CON_"
    assert course_dir_name(Course(id=1, name="课程/一")) == "课程_一 [1]"


def test_catalog_download_skips_and_dry_runs(tmp_path: Path) -> None:
    from integrations.sjtu_canvas import SJTUCatalog
    from integrations.sjtu_canvas_assistant.models import CanvasFile, Folder

    payload = b"0123456789"
    file = CanvasFile(
        id=11,
        folder_id=1,
        display_name="讲义.pdf",
        url="https://oc.test/files/11/download",
        size=len(payload),
    )

    class FakeClient:
        def get_course(self, course_id: int) -> Course:
            return Course(id=course_id, name="测试课程")

        def list_course_folders(self, course_id: int) -> list[Folder]:
            return [Folder(id=1, full_name="course files/课件")]

        def list_course_files(self, course_id: int) -> list[CanvasFile]:
            return [file]

        def download_file(self, item: CanvasFile, destination: Path) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)

    catalog = SJTUCatalog(token="token", base_url="https://oc.test")
    catalog._canvas = FakeClient()
    try:
        dry = catalog.download_course_files("1", tmp_path, match="讲义", dry_run=True)
        assert dry["summary"]["would_download"] == 1
        assert not (tmp_path / "测试课程 [1]" / "课件" / "讲义.pdf").exists()

        first = catalog.download_course_files("1", tmp_path)
        assert first["summary"]["downloaded"] == 1
        path = tmp_path / "测试课程 [1]" / "课件" / "讲义.pdf"
        assert path.read_bytes() == payload

        second = catalog.download_course_files("1", tmp_path)
        assert second["summary"]["skipped"] == 1
    finally:
        catalog.close()
