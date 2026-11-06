"""Tests for the unified course tree (recordings + live, merged periods)."""

from __future__ import annotations

from datetime import datetime

from plugins.input.tree import build_tree, live_state, merge_periods, parse_time

COURSES = [
    {
        "id": "1",
        "name": "高等数学（A）",
        "term": "2026 秋",
        "teacher": "张三",
    },
    {
        "id": "2",
        "name": "线性代数",
        "term": "2026 秋",
        "teacher": "李四",
    },
]


def _rec(video_id: str, begin: str, end: str, classroom: str = "东中院 101") -> dict:
    return {
        "video_id": video_id,
        "name": f"第 {video_id} 节",
        "begin": begin,
        "end": end,
        "classroom": classroom,
    }


# ---------------------------------------------------------------------------
# Period merging
# ---------------------------------------------------------------------------


def test_parse_time_formats() -> None:
    assert parse_time("2026-11-05 08:00:00") == datetime(2026, 11, 5, 8, 0)
    assert parse_time("2026-11-05T08:00:00") == datetime(2026, 11, 5, 8, 0)
    assert parse_time("") is None
    assert parse_time("nonsense") is None


def test_merge_adjacent_periods() -> None:
    recordings = [
        _rec("a", "2026-11-05 08:00:00", "2026-11-05 08:45:00"),
        _rec("b", "2026-11-05 08:50:00", "2026-11-05 09:35:00"),
        _rec("c", "2026-11-05 10:20:00", "2026-11-05 11:05:00"),
    ]
    groups = merge_periods(recordings, max_gap_minutes=25, max_parts=3)
    assert [len(group["parts"]) for group in groups] == [2, 1]
    assert [p["video_id"] for p in groups[0]["parts"]] == ["a", "b"]
    assert groups[0]["end"] == datetime(2026, 11, 5, 9, 35)


def test_merge_caps_at_three_periods() -> None:
    recordings = [
        _rec("a", "2026-11-05 08:00:00", "2026-11-05 08:45:00"),
        _rec("b", "2026-11-05 08:50:00", "2026-11-05 09:35:00"),
        _rec("c", "2026-11-05 09:40:00", "2026-11-05 10:25:00"),
        _rec("d", "2026-11-05 10:30:00", "2026-11-05 11:15:00"),
    ]
    groups = merge_periods(recordings, max_gap_minutes=30, max_parts=3)
    assert [len(group["parts"]) for group in groups] == [3, 1]


def test_merge_does_not_cross_days() -> None:
    recordings = [
        _rec("a", "2026-11-05 23:00:00", "2026-11-05 23:40:00"),
        _rec("b", "2026-11-06 00:00:00", "2026-11-06 00:40:00"),
    ]
    assert len(merge_periods(recordings, max_gap_minutes=60)) == 2


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------


def test_build_tree_course_and_class_nodes() -> None:
    recordings = {
        "1": [
            _rec("a", "2026-11-05 08:00:00", "2026-11-05 08:45:00"),
            _rec("b", "2026-11-05 08:50:00", "2026-11-05 09:35:00"),
        ],
        "2": [],
    }
    nodes = build_tree(COURSES, recordings, [], now=datetime(2026, 11, 6, 12, 0))
    assert [node["label"] for node in nodes] == ["高等数学（A）", "线性代数"]
    maths = nodes[0]
    assert maths["expanded"] is False
    assert len(maths["children"]) == 1
    child = maths["children"][0]
    assert child["label"] == "11-05 08:00–09:35"
    assert "2 节" in child["subtitle"]
    value = child["action"]["value"]
    assert value["kind"] == "recording"
    assert value["videos"] == ["a", "b"]
    assert nodes[1]["children"] == []


def test_build_tree_live_session_is_marked() -> None:
    recordings = {"1": [_rec("a", "2026-11-04 08:00:00", "2026-11-04 09:35:00")], "2": []}
    lives = [
        {
            "id": "L1",
            "name": "高等数学（A）",
            "begin": "2026-11-05 08:00:00",
            "end": "2026-11-05 09:35:00",
            "classroom": "东中院 101",
        }
    ]
    now = datetime(2026, 11, 5, 8, 30)
    nodes = build_tree(COURSES, recordings, lives, now=now)
    maths = nodes[0]
    assert maths["live"] is True
    assert maths["expanded"] is True  # courses with a live class open by default
    live_child = next(child for child in maths["children"] if child.get("live"))
    assert live_child["badge"] == "直播中"
    assert live_child["action"]["value"]["kind"] == "live"
    assert live_child["action"]["value"]["live_id"] == "L1"


def test_build_tree_does_not_duplicate_recorded_live() -> None:
    recordings = {
        "1": [_rec("a", "2026-11-05 08:00:00", "2026-11-05 09:35:00")],
        "2": [],
    }
    lives = [
        {
            "id": "L1",
            "name": "高等数学（A）",
            "begin": "2026-11-05 08:00:00",
            "end": "2026-11-05 09:35:00",
        }
    ]
    nodes = build_tree(COURSES, recordings, lives, now=datetime(2026, 11, 5, 10, 0))
    # A near-identical recording already exists, so the live entry is skipped.
    assert len(nodes[0]["children"]) == 1
    assert nodes[0]["children"][0]["badge"] == "录播"


def test_build_tree_unmatched_live_goes_to_other_node() -> None:
    nodes = build_tree(
        COURSES,
        {"1": [], "2": []},
        [
            {
                "id": "L9",
                "name": "人工智能导论",
                "begin": "2026-11-05 08:00:00",
                "end": "2026-11-05 09:00:00",
            }
        ],
        now=datetime(2026, 11, 5, 8, 30),
    )
    assert nodes[-1]["label"] == "其他直播"
    assert nodes[-1]["children"][0]["action"]["value"]["live_id"] == "L9"


def test_build_tree_current_class_is_active_and_expanded() -> None:
    recordings = {"1": [_rec("a", "2026-11-05 08:00:00", "2026-11-05 09:35:00")], "2": []}
    nodes = build_tree(
        COURSES,
        recordings,
        [],
        now=datetime(2026, 11, 6, 12, 0),
        expanded={"1"},
        current_class_id="class:1:rec:a",
    )
    assert nodes[0]["expanded"] is True
    assert nodes[0]["children"][0]["active"] is True


def test_live_state_labels() -> None:
    now = datetime(2026, 11, 5, 8, 30)
    assert live_state("2026-11-05 08:00:00", "2026-11-05 09:00:00", now=now) == "直播中"
    assert live_state("2026-11-05 10:00:00", "2026-11-05 11:00:00", now=now) == "未开始"
    assert live_state("2026-11-05 06:00:00", "2026-11-05 07:00:00", now=now) == "已结束"
    assert live_state("", "") == ""
