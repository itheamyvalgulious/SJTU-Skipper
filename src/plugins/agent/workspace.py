"""FileWorkspace: logical name → absolute path mapping with safe line-oriented I/O.

Safety invariants enforced at registration and on every access:

- ``..`` is rejected (path traversal).
- Absolute paths are rejected (must be relative logical names).
- Empty / non-string names are rejected.
- Atomic writes via ``mkstemp`` + ``os.replace``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class FileWorkspace:
    """A sandbox that maps logical names to absolute file paths.

    Plugins (for example summarise) create a workspace per session and pass it to
    ``AgentService.create_session(files=ws)``.  The agent tools only see the
    logical names; the real paths are never exposed to the LLM.
    """

    def __init__(self, files: Mapping[str, str | Path] | None = None) -> None:
        self._files: dict[str, Path] = {}
        if files:
            for name, path in files.items():
                self.register(name, path)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, name: str, path: str | Path) -> None:
        """Map *name* to *path*, validating the name first."""
        self._validate_name(name)
        resolved = Path(path).resolve()
        self._files[name] = resolved

    def ensure(self, name: str) -> Path:
        """Return the absolute path for *name*, creating an empty file if missing."""
        self._validate_name(name)
        if name not in self._files:
            raise KeyError(f"workspace file {name!r} not registered")
        path = self._files[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return path

    @property
    def names(self) -> tuple[str, ...]:
        """Registered logical names in insertion order."""
        return tuple(self._files)

    def path(self, name: str) -> Path:
        """Return the absolute path for *name* (never creates)."""
        self._validate_name(name)
        if name not in self._files:
            raise KeyError(f"workspace file {name!r} not registered")
        return self._files[name]

    def __contains__(self, name: str) -> bool:
        return name in self._files

    def to_dict(self) -> dict[str, Path]:
        """Shallow copy of the mapping (for subsession inheritance)."""
        return dict(self._files)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("file name must be a non-empty string")
        parts = name.replace("\\", "/").split("/")
        if any(p == ".." for p in parts):
            raise ValueError(f"file name must not contain '..': {name!r}")
        if os.path.isabs(name):
            raise ValueError(f"file name must not be an absolute path: {name!r}")

    # ------------------------------------------------------------------
    # Atomic write
    # ------------------------------------------------------------------
    def write(self, name: str, content: str) -> None:
        """Atomically replace the file content (ensures parent dirs)."""
        path = self.ensure(name)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Line-oriented I/O (1-based)
    # ------------------------------------------------------------------
    def read_lines(self, name: str, l: int, r: int) -> str:
        """Read lines *l* through *r* (1-based, inclusive).

        ``-1`` for *l* or *r* means "unbounded on that side".
        Returns a header line ``"# name (lines lo-hi of N)\\n"`` followed by
        the raw lines **without** line-number prefixes (so the output is
        directly usable as ``old_str`` for ``replace_unique``).
        """
        path = self._files.get(name)
        if path is None or not path.exists():
            return f"# {name} (lines ?-? of 0)\n"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        N = len(lines)
        if N == 0:
            return f"# {name} (lines ?-? of 0)\n"
        # Resolve bounds
        lo = max(1, l) if l != -1 else 1
        hi = min(N, r) if r != -1 else N
        if lo > hi:
            lo = max(1, hi)
            hi = min(N, l if l != -1 else N)
        lo = max(1, lo)
        hi = min(N, hi)
        if lo > hi:
            lo, hi = 1, 1
        selected = lines[lo - 1 : hi]
        header = f"# {name} (lines {lo}-{hi} of {N})\n"
        return header + "".join(selected)

    def replace_unique(self, name: str, old: str, new: str) -> str:
        """Replace *old* with *new* exactly once.

        Returns a success message.  Raises :class:`ValueError` when *old*
        appears zero times or more than once.  The error message includes
        the count and, for multiple matches, the line numbers.
        """
        path = self.ensure(name)
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ValueError(f"未找到要替换的文本（出现 0 次）")
        if count > 1:
            line_nos = _find_line_numbers(text, old)
            line_str = ", ".join(str(ln) for ln in line_nos)
            raise ValueError(
                f"替换文本出现 {count} 次（行 {line_str}），"
                f"请提供更具体的上下文以确保唯一"
            )
        new_text = text.replace(old, new, 1)
        self.write(name, new_text)
        return "已替换 1 处"

    def append_line(self, name: str, text: str) -> str:
        """Append *text* as a new line at the end, ensuring a trailing newline."""
        path = self.ensure(name)
        content = path.read_text(encoding="utf-8")
        if content and not content.endswith("\n"):
            content += "\n"
        content += text + "\n"
        self.write(name, content)
        return f"已追加 1 行到 {name}"

    def text(self, name: str) -> str:
        """Return full file content, or ``""`` if missing."""
        path = self._files.get(name)
        if path is None or not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Pickle helpers (for copying)
    # ------------------------------------------------------------------
    def copy(self) -> "FileWorkspace":
        """Return a shallow clone with the same logical→path mapping."""
        return FileWorkspace(dict(self._files))


def _find_line_numbers(text: str, needle: str) -> list[int]:
    """Return 1-based line numbers where *needle* appears in *text*."""
    line_nos: list[int] = []
    remaining = text
    offset = 0
    while True:
        idx = remaining.find(needle)
        if idx == -1:
            break
        line_no = text[: offset + idx].count("\n") + 1
        line_nos.append(line_no)
        offset += idx + len(needle)
        remaining = text[offset:]
    return line_nos