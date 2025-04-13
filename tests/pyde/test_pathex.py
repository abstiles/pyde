from pathlib import Path

from pyde.pathex import match, compile

from ..test import parametrize


@parametrize(
    ('**', 'file.ext'),
    ('**', './file.ext'),
    ('**', 'path/to/file.ext'),
    ('path/**/file.ext', 'path/to/some/deep/file.ext'),
    ('path/**/*.ext', 'path/to/some/deep/file.ext'),
    ('path/*/file.ext', 'path/to/file.ext'),
    ('.*', '.dotfile'),
    ('**/.*', 'some/.dotfile/and/contained/files.ext'),
    ('file+with+plusses.ext', 'file+with+plusses.ext'),
    ('fi?e.ext', 'file.ext'),
    ('[!p]ile.ext', 'file.ext'),
)
def test_match(glob: str, path: str) -> None:
    pathex = compile(glob)
    assert pathex.match(Path(path))


@parametrize(
    ('path/**/file.ext', 'some/deep/file.ext'),
    ('path/**/*.ext', 'path/to/some/deep/file.md'),
    ('path/**/*.ext', 'path/to/some/deep/file.extra'),
    ('path/*/file.ext', 'path/to/some/deep/file.ext'),
    ('path/*/file.ext', 'another/path/to/file.ext'),
    ('[!f]ile.ext', 'file.ext'),
    ('**/.*', 'path/to/file.ext'),
    ('**/.*', '.dotfile'),
)
def test_not_match(glob: str, path: str) -> None:
    pathex = compile(glob)
    assert not pathex.match(Path(path))
    assert not match(glob, Path(path))
