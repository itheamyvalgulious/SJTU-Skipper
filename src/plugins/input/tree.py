"""Course tree building: one unified list of courses → class sessions.

Live streams and recordings are presented the same way: a course node with
child class nodes.  Consecutive recording periods (2–3 back-to-back periods of
the same course) are merged into a single class node that keeps every part.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable

_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
)

# Two periods belong to the same class when the break between them is short.
DEFAULT_MAX_GAP_MINUTES = 25
DEFAULT_MAX_PARTS = 3


def parse_time(value: Any) -> datetime | None:
    """Parse a Canvas/video-portal timestamp (campus local time)."""

    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def local_time(value: Any, *, with_year: bool = False) -> str:
    """Human readable local time, e.g. ``10-28 08:00``."""

    dt = parse_time(value)
    if dt is None:
        return str(value or "")
    return dt.strftime("%Y-%m-%d %H:%M" if with_year else "%m-%d %H:%M")


def live_state(begin: Any, end: Any, *, now: datetime | None = None) -> str:
    """Return 直播中 / 未开始 / 已结束 for a time range."""

    begin_dt = parse_time(begin)
    end_dt = parse_time(end)
    if begin_dt is None or end_dt is None:
        return ""
    current = now or datetime.now()
    if begin_dt <= current <= end_dt:
        return "直播中"
    if current < begin_dt:
        return "未开始"
    return "已结束"


def next_boundary(
    lives: Iterable[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> float:
    """Epoch seconds of the nearest upcoming live begin/end, or ``0.0``.

    The sidebar derives each class label (直播中 / 未开始 / 已结束) from its
    begin/end times, so those timestamps are exactly the moments a label can
    change.  Returning the nearest one lets the caller refresh the tree on
    schedule instead of waiting for the user to press 刷新.
    """

    current = now or datetime.now()
    upcoming: list[float] = []
    for live in lives:
        for key in ("begin", "end"):
            dt = parse_time(live.get(key))
            if dt is not None and dt > current:
                upcoming.append(dt.timestamp())
    return min(upcoming) if upcoming else 0.0


# ---------------------------------------------------------------------------
# Merging consecutive periods
# ---------------------------------------------------------------------------
def merge_periods(
    recordings: Iterable[dict[str, Any]],
    *,
    max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES,
    max_parts: int = DEFAULT_MAX_PARTS,
) -> list[dict[str, Any]]:
    """Group back-to-back recording periods into one class.

    Each returned item is ``{"parts": [...], "begin": datetime, "end": datetime}``
    where ``parts`` keeps the original recording dicts in chronological order.
    """

    items = [r for r in recordings if parse_time(r.get("begin")) is not None]
    items.sort(key=lambda r: parse_time(r.get("begin")) or datetime.min)
    max_gap = timedelta(minutes=max(0, int(max_gap_minutes)))

    groups: list[dict[str, Any]] = []
    for item in items:
        begin = parse_time(item.get("begin"))
        end = parse_time(item.get("end")) or begin
        assert begin is not None and end is not None
        if groups:
            last = groups[-1]
            gap = begin - last["end"]
            same_day = begin.date() == last["begin"].date()
            if (
                same_day
                and timedelta(0) <= gap <= max_gap
                and len(last["parts"]) < max(1, int(max_parts))
            ):
                last["parts"].append(item)
                last["end"] = max(last["end"], end)
                continue
        groups.append({"parts": [item], "begin": begin, "end": end})
    return groups


# ---------------------------------------------------------------------------
# Course / class nodes
# ---------------------------------------------------------------------------
def _norm_name(text: Any) -> str:
    return "".join(str(text or "").split()).lower()


def _match_course(name: str, courses: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = _norm_name(name)
    if not target:
        return None
    for course in courses:
        if _norm_name(course.get("name")) == target:
            return course
    for course in courses:
        other = _norm_name(course.get("name"))
        if other and (target in other or other in target):
            return course
    return None


def _recording_overlaps(live_begin: datetime | None, groups: list[dict[str, Any]]) -> bool:
    if live_begin is None:
        return False
    window = timedelta(minutes=40)
    for group in groups:
        if abs(group["begin"] - live_begin) <= window:
            return True
    return False


def _class_label(begin: datetime, end: datetime, parts: int) -> str:
    return f"{begin:%m-%d} {begin:%H:%M}–{end:%H:%M}"


def _recording_child(
    course: dict[str, Any],
    group: dict[str, Any],
) -> dict[str, Any]:
    course_id = str(course.get("id", ""))
    parts = group["parts"]
    classrooms = [str(p.get("classroom", "") or "") for p in parts]
    classroom = next((c for c in classrooms if c), "")
    video_ids = [str(p.get("video_id", "")) for p in parts if p.get("video_id")]
    label = _class_label(group["begin"], group["end"], len(parts))
    subtitle = " · ".join(
        part
        for part in (
            f"{len(parts)} 节" if len(parts) > 1 else "",
            classroom,
        )
        if part
    )
    return {
        "id": f"class:{course_id}:rec:{video_ids[0] if video_ids else ''}",
        "label": label,
        "subtitle": subtitle,
        "badge": "录播",
        "badge_class": "done",
        "action": {
            "action": "input.open_class",
            "value": {
                "id": f"class:{course_id}:rec:{video_ids[0] if video_ids else ''}",
                "kind": "recording",
                "course": course_id,
                "name": str(course.get("name", "")),
                "label": label,
                "videos": video_ids,
            },
        },
    }


def merge_live_periods(
    lives: Iterable[dict[str, Any]],
    *,
    max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES,
    max_parts: int = DEFAULT_MAX_PARTS,
) -> list[dict[str, Any]]:
    """Group back-to-back live periods of one course into one class.

    The video portal publishes a 2- or 3-period lecture as separate live rows;
    merging them keeps the sidebar from showing "two classes" that are really
    one.  Mirrors :func:`merge_periods` for recordings.
    """

    items = [item for item in lives if parse_time(item.get("begin")) is not None]
    items.sort(key=lambda item: parse_time(item.get("begin")) or datetime.min)
    max_gap = timedelta(minutes=max(0, int(max_gap_minutes)))

    groups: list[dict[str, Any]] = []
    for item in items:
        begin = parse_time(item.get("begin"))
        end = parse_time(item.get("end")) or begin
        assert begin is not None and end is not None
        if groups:
            last = groups[-1]
            gap = begin - last["end"]
            same_day = begin.date() == last["begin"].date()
            if (
                same_day
                and timedelta(0) <= gap <= max_gap
                and len(last["parts"]) < max(1, int(max_parts))
            ):
                last["parts"].append(item)
                last["end"] = max(last["end"], end)
                continue
        groups.append({"parts": [item], "begin": begin, "end": end})
    return groups


def _live_group_child(
    course_id: str,
    course_name: str,
    group: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any] | None:
    """One class node for a merged run of live periods."""

    parts: list[dict[str, Any]] = group["parts"]
    states = [
        live_state(part.get("begin"), part.get("end"), now=now) for part in parts
    ]
    if not any(states):
        return None
    if "直播中" in states:
        state = "直播中"
    elif "未开始" in states:
        state = "未开始"
    else:
        state = "已结束"

    # Open the part that is live now, else the next upcoming one, else the last.
    active = None
    for part, part_state in zip(parts, states):
        if part_state == "直播中":
            active = part
            break
    if active is None:
        for part, part_state in zip(parts, states):
            if part_state == "未开始":
                active = part
                break
    if active is None:
        active = parts[-1]

    begin = group["begin"]
    end = group["end"]
    label = f"{begin:%m-%d %H:%M}–{end:%H:%M}"
    classrooms = [str(p.get("classroom", "") or "") for p in parts]
    subtitle = " · ".join(
        part
        for part in (
            state,
            next((c for c in classrooms if c), ""),
            f"{len(parts)} 节" if len(parts) > 1 else "",
        )
        if part
    )
    live_id = str(active.get("id", ""))
    child_id = f"class:{course_id}:live:{live_id}"
    return {
        "id": child_id,
        "label": label,
        "subtitle": subtitle,
        "badge": state,
        "badge_class": "live" if state == "直播中" else "upcoming",
        "live": state == "直播中",
        "action": {
            "action": "input.open_class",
            "value": {
                "id": child_id,
                "kind": "live",
                "course": course_id,
                "name": course_name,
                "label": label,
                "live_id": live_id,
                "parts": len(parts),
            },
        },
    }


def _live_child(
    course_id: str,
    course_name: str,
    live: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any] | None:
    """Backwards-compatible single-period wrapper."""

    begin = parse_time(live.get("begin"))
    end = parse_time(live.get("end")) or begin
    if begin is None or end is None:
        return None
    return _live_group_child(
        course_id,
        course_name,
        {"parts": [live], "begin": begin, "end": end},
        now=now,
    )


def build_tree(
    courses: list[dict[str, Any]],
    recordings_by_course: dict[str, list[dict[str, Any]]],
    lives: list[dict[str, Any]] | None = None,
    *,
    now: datetime | None = None,
    expanded: Iterable[str] = (),
    current_class_id: str = "",
    max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES,
    max_parts: int = DEFAULT_MAX_PARTS,
) -> list[dict[str, Any]]:
    """Build the unified course tree (course → class) for the sidebar."""

    now = now or datetime.now()
    expanded_ids = {str(item) for item in expanded}
    lives = list(lives or [])

    groups_by_course: dict[str, list[dict[str, Any]]] = {}
    for course in courses:
        cid = str(course.get("id", ""))
        groups_by_course[cid] = merge_periods(
            recordings_by_course.get(cid, []),
            max_gap_minutes=max_gap_minutes,
            max_parts=max_parts,
        )

    unmatched: list[dict[str, Any]] = []
    lives_by_course: dict[str, list[dict[str, Any]]] = {}
    names_by_course: dict[str, str] = {}
    for live in lives:
        matched = _match_course(str(live.get("name", "")), courses)
        if matched is None:
            unmatched.append(live)
            continue
        cid = str(matched.get("id", ""))
        begin = parse_time(live.get("begin"))
        if _recording_overlaps(begin, groups_by_course.get(cid, [])):
            continue  # recorded already — do not duplicate the class
        lives_by_course.setdefault(cid, []).append(live)
        names_by_course[cid] = str(matched.get("name", ""))

    for cid, items in lives_by_course.items():
        for group in merge_live_periods(
            items, max_gap_minutes=max_gap_minutes, max_parts=max_parts
        ):
            child = _live_group_child(
                cid, names_by_course.get(cid, ""), group, now=now
            )
            if child is not None:
                groups_by_course.setdefault(cid, [])
                groups_by_course[cid].append(
                    {"__live__": child, "begin": group["begin"]}
                )

    nodes: list[dict[str, Any]] = []
    for course in courses:
        cid = str(course.get("id", ""))
        children: list[tuple[datetime, dict[str, Any]]] = []
        for group in groups_by_course.get(cid, []):
            if "__live__" in group:
                children.append((group["begin"] or now, group["__live__"]))
            else:
                children.append((group["begin"], _recording_child(course, group)))

        children.sort(key=lambda item: item[0], reverse=True)
        child_nodes = [node for _begin, node in children]
        is_live = any(node.get("live") for node in child_nodes)
        if current_class_id:
            for node in child_nodes:
                node["active"] = node.get("id") == current_class_id

        nodes.append(
            {
                "id": f"course:{cid}",
                "label": str(course.get("name", "")) or f"课程 {cid}",
                "subtitle": " · ".join(
                    part
                    for part in (
                        str(course.get("term", "") or ""),
                        str(course.get("teacher", "") or ""),
                    )
                    if part
                ),
                "expanded": (cid in expanded_ids) or is_live,
                "live": is_live,
                "children": child_nodes,
                "toggle": {
                    "action": "input.toggle_course",
                    "value": {"course": cid},
                },
            }
        )

    if unmatched:
        children = []
        by_name: dict[str, list[dict[str, Any]]] = {}
        for live in unmatched:
            by_name.setdefault(_norm_name(live.get("name", "")), []).append(live)
        for items in by_name.values():
            for group in merge_live_periods(
                items, max_gap_minutes=max_gap_minutes, max_parts=max_parts
            ):
                child = _live_group_child("other", "其他直播", group, now=now)
                if child is not None:
                    if current_class_id:
                        child["active"] = child.get("id") == current_class_id
                    children.append(child)
        if children:
            nodes.append(
                {
                    "id": "course:other",
                    "label": "其他直播",
                    "subtitle": "未匹配到 Canvas 课程",
                    "expanded": "other" in expanded_ids,
                    "live": any(c.get("live") for c in children),
                    "children": children,
                    "toggle": {
                        "action": "input.toggle_course",
                        "value": {"course": "other"},
                    },
                }
            )

    return nodes
