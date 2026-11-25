"""SJTU Canvas & video portal integration.

Wraps the vendored ``sjtu_canvas_assistant`` package into a catalog that the
input plugin consumes. Each Canvas course gets its own ``SJTUVideoClient`` to
avoid auth crossover.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .sjtu_canvas_assistant.canvas_client import CanvasClient
from .sjtu_canvas_assistant.config import _clean_base_url
from .sjtu_canvas_assistant.exceptions import (
    AssistantError,
    CanvasAPIError,
    CanvasAuthError,
    CanvasNetworkError,
    CanvasNotFoundError,
    CanvasPermissionError,
    VideoAuthError,
    VideoIntegrationError,
    VideoNetworkError,
)
from .sjtu_canvas_assistant.path_utils import (
    canvas_file_path,
    course_dir_name,
    planned_file_path,
    unique_path_for_conflict,
)
from .sjtu_canvas_assistant.video_client import NEW_VIDEO_SERVICE_ROOT, SJTUVideoClient

_log = logging.getLogger("lecture_agent.integrations.sjtu_canvas")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InputError(RuntimeError):
    """User-facing input/catalog error (token-safe message)."""


def load_token_from_env() -> str:
    """Read the Canvas token from the project .env file."""

    from dotenv import load_dotenv

    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return (os.getenv("CANVAS_TOKEN") or "").strip()


def _load_base_url(default: str = "https://oc.sjtu.edu.cn") -> str:
    raw = (os.getenv("CANVAS_BASE_URL") or default).strip()
    return _clean_base_url(raw)


def _file_download_record(
    file: Any,
    folder_map: dict[int, Any],
    target: Path,
    status: str,
    message: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": file.id,
        "name": file.effective_name,
        "canvas_path": canvas_file_path(file, folder_map),
        "path": str(target),
        "status": status,
        "message": message,
        "size": file.size,
        **extra,
    }


class SJTUCatalog:
    """Thin catalog that combines Canvas courses and SJTU video portal data.

    Each course gets its own per-course ``SJTUVideoClient`` to avoid auth
    crossover (the video client caches a per-course token).
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = "",
        timeout_seconds: float = 30.0,
    ) -> None:
        # token=None means "load from env"; token="" means "intentionally empty"
        if token is None:
            self._token = load_token_from_env()
        else:
            self._token = str(token)
        self._base_url = base_url or _load_base_url()
        self._timeout = float(timeout_seconds)
        self._canvas: CanvasClient | None = None
        self._video_clients: dict[str, SJTUVideoClient] = {}
        self._closed = False
        # The course tree fetches every course concurrently, so client
        # creation/iteration must be serialised (never held across I/O).
        self._lock = threading.RLock()

    def _require_token(self) -> str:
        token = self._token
        if not token:
            raise InputError(
                "Canvas token 未配置。请在设置中填写 input.canvas_token，"
                "或将 CANVAS_TOKEN 写入 .env 文件。"
            )
        return token

    def _canvas_client(self) -> CanvasClient:
        with self._lock:
            if self._closed:
                raise InputError("catalog 已关闭")
            if self._canvas is None:
                self._canvas = CanvasClient(
                    base_url=self._base_url,
                    token=self._require_token(),
                    timeout_seconds=self._timeout,
                )
            return self._canvas

    @contextmanager
    def _canvas_errors(self, label: str) -> Iterator[None]:
        """Translate Canvas client exceptions into user-facing errors."""

        try:
            yield
        except CanvasAuthError:
            raise InputError("Canvas token 无效，请在设置中更新 input.canvas_token。")
        except CanvasNetworkError as exc:
            raise InputError(f"Canvas 网络错误：{exc}")
        except CanvasAPIError as exc:
            raise InputError(f"{label}：{exc}")

    def _video_client_for(self, course_id: str) -> SJTUVideoClient:
        cid = str(course_id)
        with self._lock:
            if self._closed:
                raise InputError("catalog 已关闭")
            if cid not in self._video_clients:
                self._video_clients[cid] = SJTUVideoClient(
                    canvas_base_url=self._base_url,
                    canvas_token=self._require_token(),
                    timeout_seconds=self._timeout,
                )
            return self._video_clients[cid]

    def _reset_video_client(self, course_id: str) -> SJTUVideoClient:
        """Force a fresh video client (e.g. after auth error)."""
        cid = str(course_id)
        with self._lock:
            old = self._video_clients.pop(cid, None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        return self._video_client_for(cid)

    def list_courses(self, active_only: bool = True) -> list[dict[str, Any]]:
        """Return Canvas courses: id, name, term, teacher, course_code.

        Only active courses by default (excludes historical semesters).
        """
        client = self._canvas_client()
        try:
            courses = client.list_courses(active_only=active_only)
        except CanvasAuthError:
            raise InputError("Canvas token 无效，请在设置中更新 input.canvas_token。")
        except CanvasNetworkError as exc:
            raise InputError(f"Canvas 网络错误：{exc}")
        except (CanvasAPIError, CanvasPermissionError, CanvasNotFoundError) as exc:
            raise InputError(f"Canvas API 错误：{exc}")

        result: list[dict[str, Any]] = []
        for c in courses:
            teacher_names = [
                t.display_name for t in (c.teachers or []) if t.display_name
            ]
            term_name = c.term.name if c.term and c.term.name else ""
            result.append(
                {
                    "id": str(c.id),
                    "name": c.name or c.course_code or f"course-{c.id}",
                    "course_code": c.course_code,
                    "term": term_name,
                    "teacher": ", ".join(teacher_names) if teacher_names else "",
                    "workflow_state": c.workflow_state,
                }
            )
        return result

    def get_current_user(self) -> dict[str, Any]:
        """Return the Canvas account behind the configured token."""

        with self._canvas_errors("获取用户信息失败"):
            user = self._canvas_client().get_current_user()
        return {
            "id": user.id,
            "name": user.name,
            "login_id": user.login_id,
            "email": user.email or "",
        }

    def get_course(self, course_id: str) -> dict[str, Any]:
        """Return one Canvas course with teachers and term."""

        with self._canvas_errors("获取课程信息失败"):
            course = self._canvas_client().get_course(int(course_id))
        teacher_names = [
            t.display_name for t in (course.teachers or []) if t.display_name
        ]
        return {
            "id": str(course.id),
            "name": course.display_name,
            "course_code": course.course_code,
            "term": course.term.name if course.term and course.term.name else "",
            "teacher": ", ".join(teacher_names),
            "workflow_state": course.workflow_state,
        }

    def list_course_folders(self, course_id: str) -> list[dict[str, Any]]:
        """Return Canvas file folders for a course."""

        with self._canvas_errors("获取课程文件夹失败"):
            folders = self._canvas_client().list_course_folders(int(course_id))
        return [
            {
                "id": folder.id,
                "name": folder.name,
                "full_name": folder.full_name,
                "parent_folder_id": folder.parent_folder_id,
                "files_count": folder.files_count,
                "folders_count": folder.folders_count,
                "locked": folder.locked,
            }
            for folder in folders
        ]

    def list_course_files(self, course_id: str) -> list[dict[str, Any]]:
        """Return course material files with their Canvas folder path."""

        client = self._canvas_client()
        cid = int(course_id)
        with self._canvas_errors("获取课程文件失败"):
            folders = client.list_course_folders(cid)
            files = client.list_course_files(cid)
        folder_map = {folder.id: folder for folder in folders}
        return [
            {
                "id": file.id,
                "name": file.effective_name,
                "folder_id": file.folder_id,
                "path": canvas_file_path(file, folder_map),
                "size": file.size,
                "updated_at": file.updated_at or file.modified_at or "",
                "mime_class": file.mime_class,
                "locked": file.locked,
                "downloadable": bool(file.url),
            }
            for file in files
        ]

    def list_course_modules(self, course_id: str) -> list[dict[str, Any]]:
        """Return course modules with their items (files, pages, links...)."""

        with self._canvas_errors("获取课程模块失败"):
            modules = self._canvas_client().list_modules(
                int(course_id), include_items=True
            )
        result: list[dict[str, Any]] = []
        for module in modules:
            items = []
            for item in module.items:
                items.append(
                    {
                        "id": item.id,
                        "position": item.position,
                        "title": item.title,
                        "type": item.type,
                        "content_id": item.content_id,
                        "due_at": (
                            item.content_details.due_at
                            if item.content_details and item.content_details.due_at
                            else ""
                        ),
                        "url": item.html_url or item.external_url or item.url or "",
                    }
                )
            result.append(
                {
                    "id": module.id,
                    "name": module.name,
                    "position": module.position,
                    "items": items,
                }
            )
        return result

    def list_course_assignments(self, course_id: str) -> list[dict[str, Any]]:
        """Return course assignments with due dates and submission types."""

        with self._canvas_errors("获取课程作业失败"):
            assignments = self._canvas_client().list_assignments(int(course_id))
        result: list[dict[str, Any]] = []
        for assignment in assignments:
            due_at = assignment.due_at or ""
            if not due_at:
                for date in assignment.all_dates:
                    if date.base and date.due_at:
                        due_at = date.due_at
                        break
            result.append(
                {
                    "id": assignment.id,
                    "name": assignment.name,
                    "due_at": due_at,
                    "points_possible": assignment.points_possible,
                    "description": assignment.description or "",
                    "submission_types": list(assignment.submission_types),
                    "html_url": assignment.html_url,
                    "published": assignment.published,
                    "has_submitted_submissions": assignment.has_submitted_submissions,
                    "grading_type": assignment.grading_type,
                }
            )
        return result

    def list_my_submissions(self, course_id: str) -> list[dict[str, Any]]:
        """Return the current user's submission state for every assignment.

        Canvas has no bulk "self submissions" endpoint, so this costs one
        request per assignment.
        """

        client = self._canvas_client()
        cid = int(course_id)
        with self._canvas_errors("获取作业提交记录失败"):
            assignments = client.list_assignments(cid)
            submissions = [
                (assignment, client.get_my_submission(cid, assignment.id))
                for assignment in assignments
            ]
        result: list[dict[str, Any]] = []
        for assignment, submission in submissions:
            due_at = assignment.due_at or ""
            if not due_at:
                for date in assignment.all_dates:
                    if date.base and date.due_at:
                        due_at = date.due_at
                        break
            result.append(
                {
                    "assignment_id": assignment.id,
                    "assignment_name": assignment.name,
                    "due_at": due_at,
                    "submitted_at": submission.submitted_at or "",
                    "workflow_state": submission.workflow_state,
                    "attempt": submission.attempt,
                    "score": submission.score,
                    "grade": submission.grade,
                    "late": submission.late,
                    "missing": submission.missing,
                    "excused": submission.excused,
                    "submission_types": list(assignment.submission_types),
                }
            )
        return result

    def _assignment_due_at(self, assignment: Any) -> str:
        due_at = assignment.due_at or ""
        if not due_at:
            for date in assignment.all_dates:
                if date.base and date.due_at:
                    due_at = date.due_at
                    break
        return due_at

    def get_assignment_detail(
        self, course_id: str, assignment_id: int
    ) -> dict[str, Any]:
        """Return one assignment with submission rules and current state."""

        cid = int(course_id)
        client = self._canvas_client()
        with self._canvas_errors("获取作业信息失败"):
            assignment = client.get_assignment(cid, int(assignment_id))
            submission = client.get_my_submission(cid, int(assignment_id))
        return {
            "id": assignment.id,
            "name": assignment.name,
            "due_at": self._assignment_due_at(assignment),
            "unlock_at": assignment.unlock_at or "",
            "lock_at": assignment.lock_at or "",
            "points_possible": assignment.points_possible,
            "submission_types": list(assignment.submission_types),
            "allowed_extensions": list(assignment.allowed_extensions),
            "published": assignment.published,
            "html_url": assignment.html_url,
            "submission": {
                "submitted_at": submission.submitted_at or "",
                "attempt": submission.attempt,
                "workflow_state": submission.workflow_state,
                "late": submission.late,
                "missing": submission.missing,
                "excused": submission.excused,
            },
        }

    def submit_assignment(
        self,
        course_id: str,
        assignment_id: int,
        *,
        files: Sequence[Path | str] = (),
        text: str = "",
        url: str = "",
        comment: str = "",
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Submit an assignment (or return a plan when ``confirm`` is False).

        Exactly one payload kind must be given: ``files``, ``text`` or ``url``.
        Validation happens before any upload so a bad request never leaves a
        half-uploaded submission behind.
        """

        kinds = sum([bool(files), bool(text), bool(url)])
        if kinds != 1:
            raise InputError(
                "必须且只能提供一种提交内容：--file（可重复）、--text 或 --url。"
            )
        cid = int(course_id)
        aid = int(assignment_id)
        client = self._canvas_client()
        with self._canvas_errors("获取作业信息失败"):
            assignment = client.get_assignment(cid, aid)
            previous = client.get_my_submission(cid, aid)

        warnings: list[str] = []
        if assignment.published is False:
            raise InputError("该作业未发布，无法提交。")

        now = datetime.now(UTC)

        def _parse(ts: str | None) -> datetime | None:
            if not ts:
                return None
            try:
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

        unlock_at = _parse(assignment.unlock_at)
        lock_at = _parse(assignment.lock_at)
        if unlock_at and now < unlock_at:
            raise InputError(f"作业尚未开放提交（解锁于 {assignment.unlock_at}）。")
        if lock_at and now > lock_at:
            warnings.append(f"已过最终锁定时间 {assignment.lock_at}，Canvas 可能拒绝提交。")

        due_at = self._assignment_due_at(assignment)
        due = _parse(due_at)
        if due and now > due:
            warnings.append(f"已过截止时间 {due_at}，本次提交会被标记为 late。")

        submission_type = ""
        file_paths: list[Path] = []
        if files:
            submission_type = "online_upload"
            for raw in files:
                p = Path(raw).expanduser()
                if not p.is_file():
                    raise InputError(f"提交文件不存在或不是文件：{raw}")
                file_paths.append(p)
            allowed = [ext.lower().lstrip(".") for ext in assignment.allowed_extensions]
            if allowed:
                bad = [
                    p.name
                    for p in file_paths
                    if p.suffix.lower().lstrip(".") not in allowed
                ]
                if bad:
                    raise InputError(
                        f"文件类型不被允许：{', '.join(bad)}；"
                        f"允许的扩展名：{', '.join(allowed)}"
                    )
        elif text:
            submission_type = "online_text_entry"
        else:
            submission_type = "online_url"

        allowed_types = list(assignment.submission_types)
        if submission_type not in allowed_types:
            raise InputError(
                f"该作业不支持 {submission_type} 提交；"
                f"支持的类型：{', '.join(allowed_types) or '无'}"
            )

        if previous.submitted_at:
            warnings.append(
                f"已有提交记录（{previous.submitted_at}，第 {previous.attempt or '?'} 次），"
                "本次将作为新的提交版本覆盖当前版本。"
            )

        plan = {
            "course_id": cid,
            "assignment": {
                "id": assignment.id,
                "name": assignment.name,
                "due_at": due_at,
                "lock_at": assignment.lock_at or "",
                "submission_types": allowed_types,
            },
            "submission_type": submission_type,
            "files": [
                {"path": str(p), "name": p.name, "size": p.stat().st_size}
                for p in file_paths
            ],
            "text_length": len(text),
            "url": url,
            "comment": comment,
            "previous_submitted_at": previous.submitted_at or "",
            "warnings": warnings,
        }
        if not confirm:
            plan["status"] = "dry_run"
            return plan

        file_ids: list[int] = []
        uploaded: list[dict[str, Any]] = []
        try:
            for p in file_paths:
                with self._canvas_errors(f"上传文件 {p.name} 失败"):
                    file_id = client.upload_submission_file(cid, aid, p)
                file_ids.append(file_id)
                uploaded.append({"name": p.name, "file_id": file_id})
            with self._canvas_errors("提交作业失败"):
                submission = client.submit_assignment(
                    cid,
                    aid,
                    submission_type=submission_type,
                    file_ids=file_ids,
                    body=text,
                    url=url,
                    comment=comment,
                )
        except InputError:
            if uploaded:
                # Uploaded files stay attached to the user's drafts; report them.
                raise InputError(
                    "提交过程中断，已上传但未提交的附件："
                    + ", ".join(f"{u['name']}#{u['file_id']}" for u in uploaded)
                )
            raise

        return {
            "status": "submitted",
            "course_id": cid,
            "assignment_id": aid,
            "assignment_name": assignment.name,
            "submission_type": submission_type,
            "submitted_at": submission.submitted_at or "",
            "attempt": submission.attempt,
            "workflow_state": submission.workflow_state,
            "late": submission.late,
            "files": uploaded,
            "preview_url": submission.preview_url or "",
            "warnings": warnings,
        }

    def download_course_files(
        self,
        course_id: str,
        output_dir: Path | str,
        *,
        force: bool = False,
        file_ids: Sequence[int] | None = None,
        match: str = "",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Download course material files, keeping the Canvas folder layout.

        Existing files with the same size are skipped unless ``force`` is set.
        Returns a JSON-friendly summary; every file gets its own record.
        """

        root = Path(output_dir).expanduser()
        cid = int(course_id)
        client = self._canvas_client()
        with self._canvas_errors("获取课程文件失败"):
            course = client.get_course(cid)
            folders = client.list_course_folders(cid)
            files = client.list_course_files(cid)

        folder_map = {folder.id: folder for folder in folders}
        wanted = {int(value) for value in file_ids} if file_ids else None
        needle = match.strip().lower()
        selected = []
        for file in files:
            if wanted is not None and file.id not in wanted:
                continue
            canvas_path = canvas_file_path(file, folder_map)
            if needle and needle not in canvas_path.lower():
                continue
            selected.append(file)

        used_targets: list[Path] = []
        records: list[dict[str, Any]] = []

        def add_record(
            file: Any, target: Path, status: str, message: str = "", **extra: Any
        ) -> None:
            records.append(
                _file_download_record(file, folder_map, target, status, message, extra)
            )

        for file in selected:
            target = planned_file_path(root, course, file, folder_map)
            if target in used_targets:
                target = unique_path_for_conflict(target, used_targets, file.id)
            used_targets.append(target)

            if (
                not force
                and target.exists()
                and (not file.size or target.stat().st_size == file.size)
            ):
                add_record(file, target, "skipped", "already exists")
                continue
            if file.locked:
                add_record(file, target, "failed", "file is locked in Canvas")
                continue
            if not file.url:
                add_record(file, target, "failed", "missing download URL")
                continue
            if target.exists() and file.size and target.stat().st_size != file.size:
                target = unique_path_for_conflict(target, used_targets, file.id)
                used_targets.append(target)

            if dry_run:
                add_record(file, target, "would_download", "dry run")
                continue

            try:
                client.download_file(file, target)
            except (AssistantError, OSError) as exc:
                add_record(file, target, "failed", str(exc))
            else:
                add_record(file, target, "downloaded", size=target.stat().st_size)

        summary = {"downloaded": 0, "skipped": 0, "would_download": 0, "failed": 0}
        for entry in records:
            if entry["status"] in summary:
                summary[entry["status"]] += 1
        return {
            "course": {
                "id": str(course.id),
                "name": course.display_name,
            },
            "output_dir": str(root),
            "course_dir": str(root / course_dir_name(course)),
            "summary": summary,
            "files": records,
        }

    def list_recordings(
        self,
        course_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return VOD recordings for a Canvas course."""
        vc = self._video_client_for(course_id)
        cid_int = int(course_id)
        try:
            vc.resolve_course_video_auth(cid_int)
        except VideoAuthError as exc:
            _log.warning("Video auth failed for course %s: %s", course_id, exc)
            raise InputError(
                f"无法获取课程 {course_id} 的视频授权，可能是该课程无视频或授权已过期。"
            )
        except (VideoIntegrationError, VideoNetworkError) as exc:
            raise InputError(f"视频服务错误：{exc}")

        try:
            videos = vc.list_course_videos(cid_int, limit=max(1, int(limit)))
        except (VideoAuthError, VideoIntegrationError, VideoNetworkError) as exc:
            raise InputError(f"获取录播列表失败：{exc}")

        result: list[dict[str, Any]] = []
        for v in videos:
            result.append(
                {
                    "video_id": v.video_id,
                    "name": v.display_name,
                    "begin": v.course_begin_time,
                    "end": v.course_end_time,
                    "classroom": v.classroom_name,
                    "teacher": v.user_name,
                }
            )
        return result

    def get_recording_url(self, course_id: str, video_id: str) -> str:
        """Resolve a VOD playback stream URL."""
        vc = self._video_client_for(course_id)
        cid_int = int(course_id)
        try:
            vc.resolve_course_video_auth(cid_int)
            info = vc.get_video_info(cid_int, video_id)
        except VideoAuthError as exc:
            raise InputError(f"视频授权失败：{exc}")
        except (VideoIntegrationError, VideoNetworkError) as exc:
            raise InputError(f"获取视频信息失败：{exc}")

        url = info.get_stream_url() or ""
        if not url:
            _log.info("No stream URL in first attempt, retrying auth for %s", video_id)
            self._reset_video_client(course_id)
            vc2 = self._video_client_for(course_id)
            try:
                vc2.resolve_course_video_auth(cid_int)
                info2 = vc2.get_video_info(cid_int, video_id)
                url = info2.get_stream_url() or ""
            except Exception as exc:
                raise InputError(f"重试获取视频 URL 失败：{exc}")
        if not url:
            raise InputError(
                f"未找到视频 {video_id} 的播放地址。可能该视频无可播放的流。"
            )
        return url

    def get_recording_streams(self, course_id: str, video_id: str) -> dict[str, str]:
        """Return per-view stream URLs for one recording.

        The platform publishes several views per class; ``viewNum`` 1 is the
        classroom camera/blackboard and ``viewNum`` 2 is the shared screen.
        Keys are stringified view numbers.
        """

        vc = self._video_client_for(course_id)
        cid_int = int(course_id)
        try:
            vc.resolve_course_video_auth(cid_int)
            info = vc.get_video_info(cid_int, video_id)
        except VideoAuthError as exc:
            raise InputError(f"视频授权失败：{exc}")
        except (VideoIntegrationError, VideoNetworkError) as exc:
            raise InputError(f"获取视频信息失败：{exc}")

        views: dict[int, str] = {}
        for item in getattr(info, "play_items", None) or []:
            if not isinstance(item, dict):
                continue
            url = (
                item.get("url")
                or item.get("playUrl")
                or item.get("videoUrl")
                or item.get("hlsUrl")
            )
            if not url:
                continue
            try:
                view = int(item.get("viewNum") or item.get("view_num") or 0)
            except (TypeError, ValueError):
                view = 0
            views.setdefault(view, str(url))
        if not views:
            url = info.get_stream_url() or ""
            if url:
                views[0] = url
        return {str(key): value for key, value in views.items()}

    def _ensure_live_token(self) -> tuple[str, dict[str, str]]:
        """Acquire a video token for the live/device API.

        Iterates through active Canvas courses, skipping those that fail
        auth (e.g. no video tab), until one yields a valid token.

        Returns ``(token, headers)``.
        """
        # Try existing cached clients first (snapshot: workers may add more).
        with self._lock:
            cached_clients = list(self._video_clients.values())
        for vc in cached_clients:
            if vc.video_token:
                token = vc.video_token
                headers = self._live_headers(token)
                return token, headers

        client = self._canvas_client()
        try:
            courses = client.list_courses(active_only=True)
        except Exception as exc:
            raise InputError(f"无法获取课程列表以获取直播令牌：{exc}")
        if not courses:
            raise InputError("没有活跃的 Canvas 课程，无法获取视频访问令牌。")

        errors: list[str] = []
        for course in courses:
            cid = str(course.id)
            try:
                vc = self._video_client_for(cid)
                vc.resolve_course_video_auth(int(cid))
                token = vc.video_token
                if token:
                    return token, self._live_headers(token)
            except Exception as exc:
                _log.debug(
                    "Live token: course %s (%s) auth failed: %s",
                    cid,
                    course.name or cid,
                    exc,
                )
                errors.append(f"  - {course.name or cid}: {exc}")
                continue

        raise InputError(
            "无法获取视频访问令牌。已尝试所有活跃课程，均未成功：\n" + "\n".join(errors)
        )

    def _live_headers(self, token: str) -> dict[str, str]:
        """Build HTTP headers for the live API given a token."""
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://v.sjtu.edu.cn/jy-application-resourcemanage-ui/",
        }
        if token.startswith("eyJ"):
            headers["jwt-token"] = token
        else:
            headers["token"] = token
        return headers

    def list_lives(
        self,
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        """Query vod_live for live streams in [start, end).

        ``start`` / ``end`` format: ``YYYY-MM-DD HH:MM:SS``.
        """
        self._require_token()
        import httpx2

        _token, headers = self._ensure_live_token()

        params: dict[str, Any] = {
            "courBeginTime": str(start),
            "courEndTime": str(end),
            "page.pageIndex": 1,
            "page.pageSize": 100,
            "page.orders[0].asc": "true",
            "page.orders[0].field": "courBeginTime",
        }

        with httpx2.Client(timeout=httpx2.Timeout(15.0), follow_redirects=True) as cl:
            try:
                resp = cl.get(
                    f"{NEW_VIDEO_SERVICE_ROOT}/v1/vod_live",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
            except httpx2.HTTPStatusError as exc:
                raise InputError(f"直播列表查询失败 (HTTP {exc.response.status_code})")
            except httpx2.TimeoutException:
                raise InputError("直播列表查询超时")
            except httpx2.TransportError as exc:
                raise InputError(f"直播列表网络错误：{exc}")

        try:
            payload = resp.json()
        except Exception:
            raise InputError("直播列表返回了非 JSON 响应。")

        data = payload.get("data") if isinstance(payload, dict) else payload
        records = (data.get("records") if isinstance(data, dict) else None) or []

        result: list[dict[str, Any]] = []
        for item in records:
            result.append(
                {
                    "id": str(item.get("id") or item.get("courseId", "")),
                    "name": str(
                        item.get("subjName")
                        or item.get("courName")
                        or item.get("name", "")
                    ),
                    "classroom": str(
                        item.get("clroName") or item.get("classroomName", "")
                    ),
                    "begin": str(
                        item.get("courBeginTime") or item.get("beginTime", "")
                    ),
                    "end": str(item.get("courEndTime") or item.get("endTime", "")),
                    "status": str(item.get("status") or item.get("liveStatus", "")),
                }
            )
        return result

    def get_live_devices(self, live_id: str) -> list[dict[str, Any]]:
        """Fetch camera/device list for a live stream."""
        import httpx2

        _token, headers = self._ensure_live_token()

        with httpx2.Client(timeout=httpx2.Timeout(15.0), follow_redirects=True) as cl:
            try:
                resp = cl.get(
                    f"{NEW_VIDEO_SERVICE_ROOT}/v1/course_vod_videoinfos",
                    params={"courseId": str(live_id), "playType": "2"},
                    headers=headers,
                )
                resp.raise_for_status()
            except httpx2.HTTPStatusError as exc:
                raise InputError(f"直播设备查询失败 (HTTP {exc.response.status_code})")
            except httpx2.TimeoutException:
                raise InputError("直播设备查询超时")
            except httpx2.TransportError as exc:
                raise InputError(f"直播设备网络错误：{exc}")

        try:
            payload = resp.json()
        except Exception:
            raise InputError("直播设备列表返回了非 JSON 响应。")

        data = payload.get("data") if isinstance(payload, dict) else payload
        devices = (
            data.get("courseDeviceViewDtoList") if isinstance(data, dict) else []
        ) or []

        result: list[dict[str, Any]] = []
        for d in devices:
            result.append(
                {
                    "name": str(d.get("chanNameMain") or d.get("deviceName", "")),
                    "url": str(d.get("chanNameMainPlayUrl") or d.get("playUrl", "")),
                    "view_num": d.get("deviViewNum") or d.get("viewNum", 0),
                    "device_id": str(
                        d.get("deviId") or d.get("deviceId") or d.get("chanId", "")
                    ),
                }
            )
        return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            canvas, self._canvas = self._canvas, None
            clients = list(self._video_clients.values())
            self._video_clients.clear()
        if canvas is not None:
            try:
                canvas.close()
            except Exception:
                pass
        for vc in clients:
            try:
                vc.close()
            except Exception:
                pass

    def __enter__(self) -> "SJTUCatalog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
