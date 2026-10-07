import pytest

from tests.fake_server import FakeServer


@pytest.fixture
def fake_server():
    with FakeServer() as s:
        yield s
