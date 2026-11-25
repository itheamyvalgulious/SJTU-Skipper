"""Canvas CLI used by the `sjtu-canvas` opencode skill.

Lists courses, course material files/folders/modules, assignments and
submission states, downloads course material files, and can submit
assignments. Submission is the only write operation; it requires an explicit
``--yes`` flag and otherwise only prints a dry-run plan.

Run as::

    python -m integrations.sjtu_canvas_cli [options] <command> ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .sjtu_canvas import InputError, SJTUCatalog

DEFAULT_OUTPUT = "~/sjtu-canvas"


def _fmt_size(size: object) -> str:
    try:
        value = float(size or 0)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _parse_time(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _fmt_time(value: object, *, local: bool = True) -> str:
    parsed = _parse_time(value)
    if parsed is None:
        return str(value or "")
    if local:
        parsed = parsed.astimezone()
    return parsed.strftime("%Y-%m-%d %H:%M")


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _course_label(course: dict[str, Any]) -> str:
    code = course.get("course_code") or ""
    term = course.get("term") or ""
    teacher = course.get("teacher") or ""
    details = " · ".join(part for part in (code, term, teacher) if part)
    return f"[{course['id']}] {course['name']}" + (f" ({details})" if details else "")


def _resolve_assignment(
    catalog: SJTUCatalog, course_id: str, identifier: str
) -> int:
    """Resolve a numeric assignment ID or a unique name keyword."""

    text = identifier.strip()
    if not text:
        raise InputError("作业 ID 或名称不能为空。")
    if text.isdigit():
        return int(text)
    needle = text.lower()
    assignments = catalog.list_course_assignments(course_id)
    matches = [a for a in assignments if needle in (a["name"] or "").lower()]
    if not matches:
        raise InputError(f"没有找到匹配「{identifier}」的作业。")
    if len(matches) > 1:
        listing = "\n".join(f"  [{a['id']}] {a['name']}" for a in matches)
        raise InputError(f"「{identifier}」匹配到多个作业，请改用作业 ID：\n{listing}")
    return int(matches[0]["id"])


def _resolve_course(catalog: SJTUCatalog, identifier: str) -> dict[str, Any]:
    """Resolve a numeric course ID or a unique name/code keyword."""

    text = identifier.strip()
    if not text:
        raise InputError("课程 ID 或名称不能为空。")
    if text.isdigit():
        return catalog.get_course(text)

    needle = text.lower()
    matches = [
        course
        for course in catalog.list_courses()
        if needle in course["name"].lower()
        or needle in (course.get("course_code") or "").lower()
    ]
    if not matches:
        raise InputError(f"没有找到匹配「{identifier}」的课程。")
    if len(matches) > 1:
        listing = "\n".join(f"  {_course_label(course)}" for course in matches)
        raise InputError(f"「{identifier}」匹配到多门课程，请改用课程 ID：\n{listing}")
    return matches[0]


def _assignment_due_key(assignment: dict[str, Any]) -> tuple[int, float]:
    due = _parse_time(assignment.get("due_at"))
    # Undated assignments go last, otherwise sort chronologically.
    return (0, due.timestamp()) if due else (1, 0.0)


def _submission_state(record: dict[str, Any]) -> str:
    if record.get("submitted_at"):
        return "submitted"
    if record.get("missing"):
        return "missing"
    if record.get("excused"):
        return "excused"
    return record.get("workflow_state") or "unsubmitted"


def _output_root(args: argparse.Namespace) -> Path:
    if getattr(args, "output", None):
        return Path(args.output).expanduser()
    env_dir = os.getenv("CANVAS_DATA_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return Path(DEFAULT_OUTPUT).expanduser()


def _cmd_whoami(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    user = catalog.get_current_user()
    if args.json:
        _print_json(user)
        return
    identity = user["login_id"] or user["email"]
    print(f"{user['name']} ({identity})" if identity else user["name"])


def _cmd_courses(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    courses = catalog.list_courses(active_only=not args.all)
    if args.json:
        _print_json(courses)
        return
    if not courses:
        print("没有可访问的课程。")
        return
    for course in courses:
        print(_course_label(course))


def _cmd_files(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    course = _resolve_course(catalog, args.course)
    files = catalog.list_course_files(course["id"])
    needle = (args.match or "").strip().lower()
    if needle:
        files = [file for file in files if needle in file["path"].lower()]
    if args.json:
        _print_json({"course": course, "files": files})
        return
    print(_course_label(course))
    if not files:
        print("没有课程资料文件。")
        return
    for file in files:
        flags = []
        if file["locked"]:
            flags.append("locked")
        if not file["downloadable"]:
            flags.append("no-url")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        print(
            f"[{file['id']}] {file['path']}  "
            f"{_fmt_size(file['size'])}  {_fmt_time(file['updated_at'])}{suffix}"
        )


def _cmd_folders(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    course = _resolve_course(catalog, args.course)
    folders = catalog.list_course_folders(course["id"])
    if args.json:
        _print_json({"course": course, "folders": folders})
        return
    print(_course_label(course))
    if not folders:
        print("没有文件夹。")
        return
    for folder in folders:
        print(
            f"[{folder['id']}] {folder['full_name'] or folder['name']}  "
            f"(files={folder['files_count']}, folders={folder['folders_count']})"
        )


def _cmd_modules(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    course = _resolve_course(catalog, args.course)
    modules = catalog.list_course_modules(course["id"])
    if args.json:
        _print_json({"course": course, "modules": modules})
        return
    print(_course_label(course))
    if not modules:
        print("没有课程模块。")
        return
    for module in sorted(modules, key=lambda item: item["position"]):
        print(f"\n## {module['name']}")
        for item in module["items"]:
            due = f"  due={_fmt_time(item['due_at'])}" if item["due_at"] else ""
            print(f"  [{item['type']}] {item['title']}{due}")


def _cmd_assignments(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    course = _resolve_course(catalog, args.course)
    assignments = catalog.list_course_assignments(course["id"])
    if args.days is not None:
        now = datetime.now(UTC)
        end = now + timedelta(days=args.days)
        assignments = [
            assignment
            for assignment in assignments
            if (due := _parse_time(assignment["due_at"])) is not None
            and now <= due <= end
        ]
    assignments.sort(key=_assignment_due_key)
    if args.json:
        _print_json({"course": course, "assignments": assignments})
        return
    print(_course_label(course))
    if not assignments:
        print("没有作业。")
        return
    for assignment in assignments:
        due = assignment["due_at"]
        when = _fmt_time(due) if due else "无截止时间"
        points = (
            f"  {assignment['points_possible']:g} 分"
            if assignment["points_possible"] is not None
            else ""
        )
        types = ", ".join(assignment["submission_types"]) or "-"
        print(f"[{assignment['id']}] {assignment['name']}  due={when}{points}  {types}")
        if assignment["html_url"]:
            print(f"    {assignment['html_url']}")


def _cmd_submissions(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    course = _resolve_course(catalog, args.course)
    records = catalog.list_my_submissions(course["id"])
    if args.assignment is not None:
        records = [
            record for record in records if record["assignment_id"] == args.assignment
        ]
    if args.submitted_only:
        records = [record for record in records if record["submitted_at"]]
    records.sort(key=_assignment_due_key)
    if args.json:
        _print_json({"course": course, "submissions": records})
        return
    print(_course_label(course))
    if not records:
        print("没有匹配的提交记录。")
        return
    for record in records:
        state = _submission_state(record)
        submitted = (
            f"  submitted={_fmt_time(record['submitted_at'])}"
            if record["submitted_at"]
            else ""
        )
        score = f"  score={record['score']:g}" if record["score"] is not None else ""
        due = f"  due={_fmt_time(record['due_at'])}" if record["due_at"] else ""
        print(f"[{state}] {record['assignment_name']}{due}{submitted}{score}")


def _cmd_submit(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    course = _resolve_course(catalog, args.course)
    assignment_id = _resolve_assignment(catalog, course["id"], args.assignment)

    text = args.text or ""
    if args.text_file:
        text = Path(args.text_file).expanduser().read_text(encoding="utf-8")

    result = catalog.submit_assignment(
        course["id"],
        assignment_id,
        files=args.file or (),
        text=text,
        url=args.url or "",
        comment=args.comment or "",
        confirm=args.yes,
    )
    if args.json:
        _print_json(result)
        return

    print(_course_label(course))
    if result["status"] == "dry_run":
        print(
            f"[预检] 作业 [{result['assignment']['id']}] "
            f"{result['assignment']['name']}"
        )
        print(f"    提交类型：{result['submission_type']}")
        for f in result["files"]:
            print(f"    文件：{f['path']}  {_fmt_size(f['size'])}")
        if result["text_length"]:
            print(f"    文本：{result['text_length']} 字符")
        if result["url"]:
            print(f"    URL：{result['url']}")
        if result["comment"]:
            print(f"    留言：{result['comment']}")
        if result["previous_submitted_at"]:
            print(f"    上次提交：{_fmt_time(result['previous_submitted_at'])}")
        for warning in result["warnings"]:
            print(f"    ⚠ {warning}")
        print("未实际提交。确认无误后加 --yes 正式提交。")
        return

    print(
        f"[已提交] 作业 [{result['assignment_id']}] "
        f"{result.get('assignment_name', '')}"
    )
    print(
        f"    submitted={_fmt_time(result['submitted_at'])}"
        f"  attempt={result['attempt']}"
        + ("  late" if result["late"] else "")
    )
    for f in result["files"]:
        print(f"    附件：{f['name']} (file_id={f['file_id']})")
    for warning in result["warnings"]:
        print(f"    ⚠ {warning}")


def _cmd_download(catalog: SJTUCatalog, args: argparse.Namespace) -> None:
    course = _resolve_course(catalog, args.course)
    result = catalog.download_course_files(
        course["id"],
        _output_root(args),
        force=args.force,
        file_ids=args.file_id,
        match=args.match,
        dry_run=args.dry_run,
    )
    if args.json:
        _print_json(result)
        return
    summary = result["summary"]
    print(_course_label(course))
    print(f"输出目录：{result['course_dir']}")
    for record in result["files"]:
        print(
            f"[{record['status']}] {record['path']}"
            + (f"  ({record['message']})" if record["message"] else "")
        )
    labels = [
        ("downloaded", "已下载"),
        ("skipped", "已存在"),
        ("would_download", "待下载"),
        ("failed", "失败"),
    ]
    counts = "，".join(
        f"{label} {summary[key]}" for key, label in labels if summary.get(key)
    )
    print(f"完成：{counts or '没有文件'}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sjtu-canvas",
        description="读取上海交通大学 Canvas 课程资料与作业，并可提交作业（唯一写操作）。",
    )
    parser.add_argument(
        "--token", help="Canvas access token（默认读取 CANVAS_TOKEN / .env）。"
    )
    parser.add_argument("--base-url", help="Canvas 站点，默认 https://oc.sjtu.edu.cn。")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text)

    whoami = add("whoami", "显示当前 Canvas 账号。")
    whoami.add_argument("--json", action="store_true")
    whoami.set_defaults(func=_cmd_whoami)

    courses = add("courses", "列出当前活跃课程。")
    courses.add_argument("--all", action="store_true", help="包含历史学期课程。")
    courses.add_argument("--json", action="store_true")
    courses.set_defaults(func=_cmd_courses)

    files = add("files", "列出课程资料文件。")
    files.add_argument("course", help="课程 ID 或课程名关键字。")
    files.add_argument("--match", help="只显示路径中包含该关键字的文件。")
    files.add_argument("--json", action="store_true")
    files.set_defaults(func=_cmd_files)

    folders = add("folders", "列出课程资料文件夹。")
    folders.add_argument("course", help="课程 ID 或课程名关键字。")
    folders.add_argument("--json", action="store_true")
    folders.set_defaults(func=_cmd_folders)

    modules = add("modules", "列出课程模块及其条目。")
    modules.add_argument("course", help="课程 ID 或课程名关键字。")
    modules.add_argument("--json", action="store_true")
    modules.set_defaults(func=_cmd_modules)

    assignments = add("assignments", "列出课程作业与截止时间。")
    assignments.add_argument("course", help="课程 ID 或课程名关键字。")
    assignments.add_argument(
        "--days", type=int, help="只显示未来 N 天内截止的作业（含即将截止）。"
    )
    assignments.add_argument("--json", action="store_true")
    assignments.set_defaults(func=_cmd_assignments)

    submissions = add("submissions", "查看自己的作业提交状态（只读）。")
    submissions.add_argument("course", help="课程 ID 或课程名关键字。")
    submissions.add_argument("--assignment", type=int, help="只看指定作业 ID。")
    submissions.add_argument(
        "--submitted-only", action="store_true", help="只显示已提交作业。"
    )
    submissions.add_argument("--json", action="store_true")
    submissions.set_defaults(func=_cmd_submissions)

    submit = add(
        "submit",
        "提交作业（默认只预检；加 --yes 才真正提交）。",
    )
    submit.add_argument("course", help="课程 ID 或课程名关键字。")
    submit.add_argument("assignment", help="作业 ID 或作业名关键字。")
    submit.add_argument(
        "--file",
        action="append",
        help="要提交的本地文件，可重复（online_upload 类型）。",
    )
    submit.add_argument("--text", help="在线文本内容（online_text_entry 类型）。")
    submit.add_argument(
        "--text-file", help="从本地文件读取在线文本内容（UTF-8）。"
    )
    submit.add_argument("--url", help="要提交的 URL（online_url 类型）。")
    submit.add_argument("--comment", help="随提交附上的留言。")
    submit.add_argument(
        "--yes",
        action="store_true",
        help="确认执行真实提交；不加时只输出预检结果。",
    )
    submit.add_argument("--json", action="store_true")
    submit.set_defaults(func=_cmd_submit)

    download = add("download", "下载课程资料文件（保持 Canvas 文件夹结构）。")
    download.add_argument("course", help="课程 ID 或课程名关键字。")
    download.add_argument(
        "-o", "--output", help="输出根目录（默认 CANVAS_DATA_DIR 或 ~/sjtu-canvas）。"
    )
    download.add_argument(
        "--file-id", type=int, action="append", help="只下载指定文件 ID，可重复。"
    )
    download.add_argument("--match", help="只下载路径中包含该关键字的文件。")
    download.add_argument("--force", action="store_true", help="重新下载已存在的文件。")
    download.add_argument(
        "--dry-run", action="store_true", help="只列出将要下载的文件。"
    )
    download.add_argument("--json", action="store_true")
    download.set_defaults(func=_cmd_download)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    token = args.token
    base_url = args.base_url or ""
    try:
        with SJTUCatalog(token=token, base_url=base_url) as catalog:
            args.func(catalog, args)
    except InputError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
