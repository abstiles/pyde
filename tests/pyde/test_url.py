"""Tests of pyde.url"""

import pytest

from pyde.path import UrlPath
from ..test import parametrize, Fixture


EXAMPLE_URL = "https://www.example.com/path/to/file.html"


@pytest.fixture(name='url_path')
def fixture_url_path(request: pytest.FixtureRequest) -> UrlPath:
    return UrlPath(request.param)


def test_simple_equality() -> None:
    first = UrlPath(EXAMPLE_URL)
    second = UrlPath(EXAMPLE_URL)
    assert first == second


@parametrize(
    "https://www.example.com/path/to/different_file.html",
    "https://www.example.com/different/path/to/file.html",
    "https://example.com/path/to/file.html",
    "https://www.different.com/path/to/file.html",
    "http://www.example.com/path/to/file.html",
    "https://www.example.com:9001/path/to/file.html",
)
def test_inequality(url_path: Fixture[UrlPath]) -> None:
    example = UrlPath(EXAMPLE_URL)
    assert url_path != example


@parametrize(
    ("https://www.example.com/path/to/file.html", "/path/to/file.html",),
    ("https://www.example.com/dir/", "/dir/",),
    ("https://www.example.com/", "/",),
    ("https://www.example.com", "/",),
    ("/path/to/file.html", "/path/to/file.html",),
    ("", "/",),
)
def test_path(url_path: Fixture[UrlPath], path: str) -> None:
    """Test basic construction of path objects"""
    assert url_path.path == path


@parametrize(
    ("https://www.example.com/path/to/file.html", "www.example.com",),
    ("/path/to/file.html", None,),
    ("", None,),
)
def test_host(url_path: Fixture[UrlPath], host: str | None) -> None:
    assert url_path.host == host


@parametrize(
    ("/path/to", "file.html", "/path/to/file.html"),
    ("/path/to/", "file.html", "/path/to/file.html"),
    ("/path/to/", "/file.html", "/file.html"),
    ("path/to/", "file.html", "path/to/file.html"),
    ("/path", "to/", "/path/to/"),
)
def test_slash_join(url_path: Fixture[UrlPath], path: str, expected: str) -> None:
    assert str(url_path / path) == expected


@parametrize(
    ("/path/to", "file.html", "/path/to/file.html"),
    ("path/to", "file.html", "path/to/file.html"),
    ("/path/to", "dir/", "/path/to/dir/"),
    ("/path/to/dir/", "file.html", "/path/to/dir/file.html"),
    ("/path/to/dir/", "/file.html", "/file.html"),
    ("/path/to/dir/", "../style.css", "/path/to/style.css"),
    ("/path/to/dir", "../style.css", "/path/to/style.css"),
    ("/path/to/dir/", "..", "/path/to/"),
)
def test_child(url_path: Fixture[UrlPath], path: str, expected: str) -> None:
    assert str(url_path.child(path)) == expected


@parametrize(
    ("/path/to/file.html", "style.css", "/path/to/style.css"),
    ("/path/to/file.html", "./style.css", "/path/to/style.css"),
    ("/path/to/file.html", "/style.css", "/style.css"),
    ("/path/to/dir/", "style.css", "/path/to/dir/style.css"),
    ("/path/to/dir/", "./style.css", "/path/to/dir/style.css"),
    ("/path/to/dir/", "/style.css", "/style.css"),
    ("/path/to/file.html", "../style.css", "/path/style.css"),
    ("/path/to/dir/", "../style.css", "/path/to/style.css"),
    ("/path/to/dir/", "..", "/path/to/"),
)
def test_navigate(url_path: Fixture[UrlPath], path: str, expected: str) -> None:
    assert str(url_path.navigate(path)) == expected


@parametrize(
    ("/path/to/file.html", "/path/to/"),
    ("/path/to/dir/", "/path/to/"),
)
def test_parent(url_path: Fixture[UrlPath], expected: str) -> None:
    assert str(url_path.parent) == expected


def test_query() -> None:
    url_path = UrlPath(f'{EXAMPLE_URL}?x=1&y=2')
    assert url_path.query.to_dict() == {'x': '1', 'y': '2'}


def d(**kwargs: object) -> dict[str, object]:
    return kwargs

@parametrize(
    (f'{EXAMPLE_URL}?x=1&y=2', d(z=3), {'x': '1', 'y': '2', 'z': '3'}),
    (f'{EXAMPLE_URL}?x=1&y=2', d(x=2), {'x': ['1', '2'], 'y': '2'}),
    (f'{EXAMPLE_URL}?x=1&y=2', ('x', 2), {'x': ['1', '2'], 'y': '2'}),
    (f'{EXAMPLE_URL}?x=1&y=2', 'x=3&y=4&z=5', {
        'x': ['1', '3'], 'y': ['2', '4'], 'z': '5',
    }),
    (f'{EXAMPLE_URL}?x=1&y=2', [('x', 3), ('y', 4), ('z', 5)], {
        'x': ['1', '3'], 'y': ['2', '4'], 'z': '5',
    }),
)
def test_query_adds(url_path: Fixture[UrlPath], params, expected) -> None:
    url_path &= params
    assert url_path.query.to_dict() == expected

@parametrize(
    ('/path/with spaces.html', '/path/with%20spaces.html'),
    ('/path/page.html#with fragment', '/path/page.html#with%20fragment'),
    ('/path/page.html?query=with spaces', '/path/page.html?query=with+spaces'),
)
def test_quoting(url_path: Fixture[UrlPath], expected: str) -> None:
    assert str(url_path) == expected
