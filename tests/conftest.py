import pytest

from core.navigation import Navigation, navigation_scope


@pytest.fixture(autouse=True)
def navigation():
    with navigation_scope(Navigation()) as request:
        yield request
