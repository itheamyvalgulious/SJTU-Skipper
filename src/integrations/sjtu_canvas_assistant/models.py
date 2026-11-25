from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CanvasModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Teacher(CanvasModel):
    id: Optional[int] = None
    display_name: str = ""
    avatar_image_url: str = ""
    html_url: str = ""


class Term(CanvasModel):
    id: Optional[int] = None
    name: str = ""
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    workflow_state: str = ""


class Enrollment(CanvasModel):
    type: str = ""
    role: str = ""
    enrollment_state: str = ""


class Course(CanvasModel):
    id: int
    uuid: str = ""
    name: str = ""
    course_code: str = ""
    enrollments: List[Enrollment] = Field(default_factory=list)
    access_restricted_by_date: Optional[bool] = None
    teachers: List[Teacher] = Field(default_factory=list)
    term: Optional[Term] = None
    workflow_state: str = ""

    @property
    def display_name(self) -> str:
        return self.name or self.course_code or f"course-{self.id}"


class User(CanvasModel):
    id: int
    name: str = ""
    sortable_name: str = ""
    short_name: str = ""
    login_id: str = ""
    email: Optional[str] = None


class Folder(CanvasModel):
    id: int
    name: str = ""
    full_name: str = ""
    parent_folder_id: Optional[int] = None
    locked: bool = False
    files_count: int = 0
    folders_count: int = 0


class CanvasFile(CanvasModel):
    id: int
    uuid: str = ""
    folder_id: Optional[int] = None
    display_name: str = ""
    filename: str = ""
    url: str = ""
    size: int = 0
    locked: bool = False
    hidden: Optional[bool] = None
    mime_class: str = ""
    content_type: str = Field(default="", alias="content-type")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    modified_at: Optional[str] = None

    @property
    def effective_name(self) -> str:
        return self.display_name or self.filename or f"file-{self.id}"


class CompletionRequirement(CanvasModel):
    type: str = ""
    min_score: Optional[float] = None
    completed: Optional[bool] = None


class ContentDetails(CanvasModel):
    points_possible: Optional[float] = None
    due_at: Optional[str] = None
    unlock_at: Optional[str] = None
    lock_at: Optional[str] = None


class ModuleItem(CanvasModel):
    id: int
    module_id: Optional[int] = None
    position: int = 0
    title: str = ""
    indent: int = 0
    type: str = ""
    content_id: Optional[int] = None
    html_url: str = ""
    url: Optional[str] = None
    page_url: Optional[str] = None
    external_url: Optional[str] = None
    new_tab: Optional[bool] = None
    completion_requirement: Optional[CompletionRequirement] = None
    content_details: Optional[ContentDetails] = None
    published: Optional[bool] = None


class Module(CanvasModel):
    id: int
    position: int = 0
    name: str = ""
    unlock_at: Optional[str] = None
    require_sequential_progress: bool = False
    prerequisite_module_ids: List[int] = Field(default_factory=list)
    items_count: int = 0
    items_url: str = ""
    items: List[ModuleItem] = Field(default_factory=list)
    state: Optional[str] = None
    completed_at: Optional[str] = None
    published: Optional[bool] = None


class AssignmentDate(CanvasModel):
    id: Optional[int] = None
    base: bool = False
    title: str = ""
    due_at: Optional[str] = None
    unlock_at: Optional[str] = None
    lock_at: Optional[str] = None


class Assignment(CanvasModel):
    id: int
    course_id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    due_at: Optional[str] = None
    unlock_at: Optional[str] = None
    lock_at: Optional[str] = None
    points_possible: Optional[float] = None
    html_url: str = ""
    submission_types: List[str] = Field(default_factory=list)
    allowed_extensions: List[str] = Field(default_factory=list)
    has_submitted_submissions: Optional[bool] = None
    published: Optional[bool] = None
    submissions_download_url: str = ""
    all_dates: List[AssignmentDate] = Field(default_factory=list)
    grading_type: str = ""


class SubmissionAttachment(CanvasModel):
    id: int
    uuid: str = ""
    display_name: str = ""
    filename: str = ""
    url: str = ""
    size: int = 0
    content_type: str = Field(default="", alias="content-type")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    modified_at: Optional[str] = None

    @property
    def effective_name(self) -> str:
        return self.display_name or self.filename or f"submission-file-{self.id}"

    @field_validator("display_name", "filename", "url", "content_type", mode="before")
    @classmethod
    def _stringify(cls, value: object) -> str:
        return "" if value is None else str(value)


class SubmissionAttempt(CanvasModel):
    id: Optional[int] = None
    attempt: Optional[int] = None
    submitted_at: Optional[str] = None
    submission_type: str = ""
    workflow_state: str = ""
    attachments: List[SubmissionAttachment] = Field(default_factory=list)
    body: Optional[str] = None
    url: Optional[str] = None

    @field_validator("submission_type", "workflow_state", mode="before")
    @classmethod
    def _stringify(cls, value: object) -> str:
        return "" if value is None else str(value)


class Submission(CanvasModel):
    id: Optional[int] = None
    assignment_id: Optional[int] = None
    user_id: Optional[int] = None
    submitted_at: Optional[str] = None
    attempt: Optional[int] = None
    workflow_state: str = ""
    submission_type: str = ""
    grade: Optional[str] = None
    score: Optional[float] = None
    late: Optional[bool] = None
    missing: Optional[bool] = None
    excused: Optional[bool] = None
    attachments: List[SubmissionAttachment] = Field(default_factory=list)
    submission_history: List[SubmissionAttempt] = Field(default_factory=list)
    body: Optional[str] = None
    url: Optional[str] = None
    preview_url: str = ""

    @field_validator("workflow_state", "submission_type", "preview_url", mode="before")
    @classmethod
    def _stringify(cls, value: object) -> str:
        return "" if value is None else str(value)


class MySubmission(CanvasModel):
    assignment: Assignment
    submission: Submission


class VideoAuthSource(StrEnum):
    NONE = ""
    ENVIRONMENT = "environment"
    CANVAS_SESSIONLESS_LTI = "canvas_sessionless_lti"


class VideoAuth(CanvasModel):
    """Resolved SJTU video service auth for one Canvas course.

    The token is intentionally not included in normal CLI/MCP outputs.
    """

    video_token: str = ""
    video_course_id: str = ""
    source: VideoAuthSource = VideoAuthSource.NONE


class CanvasVideo(CanvasModel):
    video_id: str = Field(default="", alias="videoId")
    video_name: str = Field(default="", alias="videoName")
    user_name: str = Field(default="", alias="userName")
    classroom_name: str = Field(default="", alias="classroomName")
    course_begin_time: str = Field(default="", alias="courseBeginTime")
    course_end_time: str = Field(default="", alias="courseEndTime")
    canvas_course_id: str = Field(default="", alias="canvasCourseId")
    cour_id: str = Field(default="", alias="courId")

    @property
    def display_name(self) -> str:
        return self.video_name or f"video-{self.video_id}"

    @field_validator(
        "video_id",
        "video_name",
        "user_name",
        "classroom_name",
        "course_begin_time",
        "course_end_time",
        "canvas_course_id",
        "cour_id",
        mode="before",
    )
    @classmethod
    def _stringify(cls, value: object) -> str:
        return "" if value is None else str(value)


class VideoInfo(CanvasModel):
    id: str = ""
    cour_id: str = Field(default="", alias="courId")
    video_id: str = Field(default="", alias="videoId")
    video_name: str = Field(default="", alias="videoName")
    user_name: str = Field(default="", alias="userName")
    classroom_name: str = Field(default="", alias="classroomName")
    course_begin_time: str = Field(default="", alias="courseBeginTime")
    course_end_time: str = Field(default="", alias="courseEndTime")
    video_play_response_vo_list: List[Dict[str, Any]] = Field(
        default_factory=list, alias="videoPlayResponseVoList"
    )
    video_play_response_vo_list_snake: List[Dict[str, Any]] = Field(
        default_factory=list, alias="video_play_response_vo_list"
    )
    course_vod_view_list: List[Dict[str, Any]] = Field(
        default_factory=list, alias="courseVodViewList"
    )

    @property
    def play_items(self) -> List[Dict[str, Any]]:
        return (
            self.video_play_response_vo_list
            or self.video_play_response_vo_list_snake
            or self.course_vod_view_list
        )

    def get_stream_url(self, view_num: int = 1) -> Optional[str]:
        for item in self.play_items:
            if item.get("viewNum") == view_num and item.get("url"):
                return str(item["url"])
        for item in self.play_items:
            url = (
                item.get("url")
                or item.get("playUrl")
                or item.get("videoUrl")
                or item.get("hlsUrl")
            )
            if url:
                return str(url)
        return None

    @field_validator(
        "id",
        "cour_id",
        "video_id",
        "video_name",
        "user_name",
        "classroom_name",
        "course_begin_time",
        "course_end_time",
        mode="before",
    )
    @classmethod
    def _stringify(cls, value: object) -> str:
        return "" if value is None else str(value)
