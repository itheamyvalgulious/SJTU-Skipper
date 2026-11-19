"""Use browser navigation only within the request that supplies it."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Generator


@dataclass(slots=True)
class Navigation:
    page: str = ""
    class_id: str = ""
    doc: str = ""
    pages: list[str] = field(default_factory=list)
    class_meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "class": self.class_id,
            "doc": self.doc,
            "pages": list(self.pages),
        }


_navigation: ContextVar[Navigation | None] = ContextVar(
    "request_navigation", default=None
)


def current_navigation() -> Navigation:
    navigation = _navigation.get()
    return navigation if navigation is not None else Navigation()


@contextmanager
def navigation_scope(navigation: Navigation) -> Generator[Navigation]:
    token = _navigation.set(navigation)
    try:
        yield navigation
    finally:
        _navigation.reset(token)
