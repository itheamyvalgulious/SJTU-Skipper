"""Safe local path helpers for Canvas names."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import CanvasFile, Course, Folder

INVALID_SEGMENT_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_path_segment(value: object, *, fallback: str = "_", max_length: int = 120) -> str:
    """Return a cross-platform safe single path segment."""

    text = str(value or "").strip()
    text = INVALID_SEGMENT_CHARS.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        text = fallback

    stem_upper = text.split(".", 1)[0].upper()
    if stem_upper in RESERVED_WINDOWS_NAMES:
        text = f"{text}_"

    if len(text) > max_length:
        suffix = ""
        if "." in text:
            possible_suffix = "." + text.rsplit(".", 1)[1]
            if len(possible_suffix) <= 16:
                suffix = possible_suffix
        text = text[: max_length - len(suffix)].rstrip(" .") + suffix
    return text or fallback


def course_dir_name(course: Course) -> str:
    return safe_path_segment(f"{course.display_name} [{course.id}]", fallback=f"course-{course.id}")


def folder_relative_parts(folder: Optional[Folder]) -> List[str]:
    if folder is None:
        return []
    full_name = folder.full_name or folder.name
    parts = [part for part in full_name.replace("\\", "/").split("/") if part]
    if parts and parts[0].lower() == "course files":
        parts = parts[1:]
    return [safe_path_segment(part) for part in parts]


def canvas_file_path(file: CanvasFile, folder_map: Dict[int, Folder]) -> str:
    folder = folder_map.get(file.folder_id or -1)
    parts = folder_relative_parts(folder) + [file.effective_name]
    return "/".join(parts)


def planned_file_path(
    root_dir: Path, course: Course, file: CanvasFile, folder_map: Dict[int, Folder]
) -> Path:
    parts = [root_dir, Path(course_dir_name(course))]
    folder = folder_map.get(file.folder_id or -1)
    for part in folder_relative_parts(folder):
        parts.append(Path(part))
    parts.append(Path(safe_path_segment(file.effective_name, fallback=f"file-{file.id}")))
    return Path(*parts)


def unique_path_for_conflict(path: Path, used_paths: Iterable[Path], identifier: object) -> Path:
    """Return a deterministic non-conflicting path for a file name collision."""

    used = {p.resolve() for p in used_paths if p.exists() or p == path}
    candidate = path
    if candidate.resolve() not in used:
        return candidate

    suffix = f"__canvas_{identifier}"
    if path.suffix:
        candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    else:
        candidate = path.with_name(f"{path.name}{suffix}")
    counter = 1
    while candidate.exists() or candidate.resolve() in used:
        if path.suffix:
            candidate = path.with_name(f"{path.stem}{suffix}_{counter}{path.suffix}")
        else:
            candidate = path.with_name(f"{path.name}{suffix}_{counter}")
        counter += 1
    return candidate
