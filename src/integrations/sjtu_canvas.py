"""SJTU Canvas & video portal integration.

Wraps the vendored ``sjtu_canvas_assistant`` package into a catalog that the
input plugin consumes.  Each Canvas course gets its own ``SJTUVideoClient`` to
avoid auth crossover.  The old submodule is kept only in local ``legacy/`` and
is not required at runtime.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .sjtu_canvas_assistant.canvas_client import CanvasClient
from .sjtu_canvas_assistant.config import _clean_base_url
from .sjtu_canvas_assistant.exceptions import (
    CanvasAPIError,
    CanvasAuthError,
    CanvasNetworkError,
    CanvasNotFoundError,
    CanvasPermissionError,
    VideoAuthError,
    VideoIntegrationError,
    VideoNetworkError,
)
from .sjtu_canvas_assistant.models import CanvasVideo as _CanvasVideo
from .sjtu_canvas_assistant.models import Course as _Course
from .sjtu_canvas_assistant.video_client import NEW_VIDEO_SERVICE_ROOT
from .sjtu_canvas_assistant.video_client import SJTUVideoClient

_log = logging.getLogger("lecture_agent.integrations.sjtu_canvas")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InputError(RuntimeError):
    """User-facing input/catalog error (token-safe message)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_token_from_env() -> str:
    """Read the Canvas token from the project or legacy .env file."""

    from dotenv import load_dotenv

    for env_path in (
        _PROJECT_ROOT / ".env",
        _PROJECT_ROOT / "legacy" / "sjtu-canvas-assistant" / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    return (os.getenv("CANVAS_TOKEN") or "").strip()


def _load_base_url(default: str = "https://oc.sjtu.edu.cn") -> str:
    raw = (os.getenv("CANVAS_BASE_URL") or default).strip()
    return _clean_base_url(raw)


def _extract_stream_url(info: Any) -> str:
    """Support both the user's enhanced VideoInfo and the stock submodule."""

    getter = getattr(info, "get_stream_url", None)
    if callable(getter):
        try:
            url = getter()
            if url:
                return str(url)
        except Exception:  # noqa: BLE001 - fall through to play_items
            pass
    items = getattr(info, "play_items", None) or []
    for item in items:
        if isinstance(item, dict):
            url = (
                item.get("url")
                or item.get("playUrl")
                or item.get("videoUrl")
                or item.get("hlsUrl")
            )
            if url:
                return str(url)
    return ""


# ---------------------------------------------------------------------------
# SJTUCatalog
# ---------------------------------------------------------------------------

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
        # course_id (str) -> SJTUVideoClient
        self._video_clients: dict[str, SJTUVideoClient] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_token(self) -> str:
        token = self._token
        if not token:
            raise InputError(
                "Canvas token 未配置。请在设置中填写 input.canvas_token，"
                "或将 CANVAS_TOKEN 写入 legacy/sjtu-canvas-assistant/.env 文件。"
            )
        return token

    def _canvas_client(self) -> CanvasClient:
        if self._closed:
            raise InputError("catalog 已关闭")
        if self._canvas is None:
            self._canvas = CanvasClient(
                base_url=self._base_url,
                token=self._require_token(),
                timeout_seconds=self._timeout,
            )
        return self._canvas

    def _video_client_for(self, course_id: str) -> SJTUVideoClient:
        if self._closed:
            raise InputError("catalog 已关闭")
        cid = str(course_id)
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
        old = self._video_clients.pop(cid, None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        return self._video_client_for(cid)

    # ------------------------------------------------------------------
    # Canvas courses
    # ------------------------------------------------------------------

    def list_courses(self) -> list[dict[str, Any]]:
        """Return Canvas courses: id, name, term, teacher, course_code.

        Only active courses by default (excludes historical semesters).
        """
        client = self._canvas_client()
        try:
            courses = client.list_courses(active_only=True)
        except CanvasAuthError:
            raise InputError(
                "Canvas token 无效，请在设置中更新 input.canvas_token。"
            )
        except CanvasNetworkError as exc:
            raise InputError(f"Canvas 网络错误：{exc}")
        except (CanvasAPIError, CanvasPermissionError, CanvasNotFoundError) as exc:
            raise InputError(f"Canvas API 错误：{exc}")

        result: list[dict[str, Any]] = []
        for c in courses:
            teacher_names = [t.display_name for t in (c.teachers or []) if t.display_name]
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

    # ------------------------------------------------------------------
    # Recordings (VOD)
    # ------------------------------------------------------------------

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

        url = _extract_stream_url(info)
        if not url:
            # Try to re-auth and retry once
            _log.info("No stream URL in first attempt, retrying auth for %s", video_id)
            self._reset_video_client(course_id)
            vc2 = self._video_client_for(course_id)
            try:
                vc2.resolve_course_video_auth(cid_int)
                info2 = vc2.get_video_info(cid_int, video_id)
                url = _extract_stream_url(info2)
            except Exception as exc:
                raise InputError(f"重试获取视频 URL 失败：{exc}")
        if not url:
            raise InputError(
                f"未找到视频 {video_id} 的播放地址。可能该视频无可播放的流。"
            )
        return url

    # ------------------------------------------------------------------
    # Live token (shared by list_lives and get_live_devices)
    # ------------------------------------------------------------------

    def _ensure_live_token(self) -> tuple[str, dict[str, str]]:
        """Acquire a video token for the live/device API.

        Iterates through active Canvas courses, skipping those that fail
        auth (e.g. no video tab), until one yields a valid token.

        Returns ``(token, headers)``.
        """
        # Try existing cached clients first
        for vc in self._video_clients.values():
            if vc.video_token:
                token = vc.video_token
                headers = self._live_headers(token)
                return token, headers

        # No cached token — iterate active courses
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
                _log.debug("Live token: course %s (%s) auth failed: %s",
                           cid, course.name or cid, exc)
                errors.append(f"  - {course.name or cid}: {exc}")
                continue

        raise InputError(
            "无法获取视频访问令牌。已尝试所有活跃课程，均未成功：\n"
            + "\n".join(errors)
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

    # ------------------------------------------------------------------
    # Live streams (新版 SJTU 门户)
    # ------------------------------------------------------------------

    def list_lives(
        self,
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        """Query vod_live for live streams in [start, end).

        ``start`` / ``end`` format: ``YYYY-MM-DD HH:MM:SS``.
        """
        self._require_token()
        try:
            import httpx
        except ImportError:
            raise InputError("httpx 未安装，无法查询直播列表。")

        _token, headers = self._ensure_live_token()

        params: dict[str, Any] = {
            "courBeginTime": str(start),
            "courEndTime": str(end),
            "page.pageIndex": 1,
            "page.pageSize": 100,
            "page.orders[0].asc": "true",
            "page.orders[0].field": "courBeginTime",
        }

        with httpx.Client(timeout=httpx.Timeout(15.0), follow_redirects=True) as cl:
            try:
                resp = cl.get(
                    f"{NEW_VIDEO_SERVICE_ROOT}/v1/vod_live",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise InputError(f"直播列表查询失败 (HTTP {exc.response.status_code})")
            except httpx.TimeoutException:
                raise InputError("直播列表查询超时")
            except httpx.TransportError as exc:
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
                    "classroom": str(item.get("clroName") or item.get("classroomName", "")),
                    "begin": str(item.get("courBeginTime") or item.get("beginTime", "")),
                    "end": str(item.get("courEndTime") or item.get("endTime", "")),
                    "status": str(item.get("status") or item.get("liveStatus", "")),
                }
            )
        return result

    def get_live_devices(self, live_id: str) -> list[dict[str, Any]]:
        """Fetch camera/device list for a live stream."""
        try:
            import httpx
        except ImportError:
            raise InputError("httpx 未安装。")

        _token, headers = self._ensure_live_token()

        with httpx.Client(timeout=httpx.Timeout(15.0), follow_redirects=True) as cl:
            try:
                resp = cl.get(
                    f"{NEW_VIDEO_SERVICE_ROOT}/v1/course_vod_videoinfos",
                    params={"courseId": str(live_id), "playType": "2"},
                    headers=headers,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise InputError(f"直播设备查询失败 (HTTP {exc.response.status_code})")
            except httpx.TimeoutException:
                raise InputError("直播设备查询超时")
            except httpx.TransportError as exc:
                raise InputError(f"直播设备网络错误：{exc}")

        try:
            payload = resp.json()
        except Exception:
            raise InputError("直播设备列表返回了非 JSON 响应。")

        data = payload.get("data") if isinstance(payload, dict) else payload
        devices = (
            data.get("courseDeviceViewDtoList")
            if isinstance(data, dict)
            else []
        ) or []

        result: list[dict[str, Any]] = []
        for d in devices:
            result.append(
                {
                    "name": str(d.get("chanNameMain") or d.get("deviceName", "")),
                    "url": str(d.get("chanNameMainPlayUrl") or d.get("playUrl", "")),
                    "view_num": d.get("deviViewNum") or d.get("viewNum", 0),
                    "device_id": str(d.get("deviId") or d.get("deviceId") or d.get("chanId", "")),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._closed = True
        if self._canvas is not None:
            try:
                self._canvas.close()
            except Exception:
                pass
            self._canvas = None
        for vc in list(self._video_clients.values()):
            try:
                vc.close()
            except Exception:
                pass
        self._video_clients.clear()

    def __enter__(self) -> "SJTUCatalog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()