import io
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Self

from pyde.path.filepath import FilePath

from . import ReadableBuffer, WriteablePath


class LocalPath(WriteablePath):
    """
    Model a path that exists on the local filesystem
    """

    def __init__(self, path: str | Path | 'LocalPath'):
        self._path = Path(path)

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({str(self._path)!r})'

    def __eq__(self, other: object) -> bool:
        return self._path == other

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def stem(self) -> str:
        return self._path.stem

    @property
    def suffix(self) -> str:
        return self._path.suffix

    def with_suffix(self, suffix: str) -> Self:
        return self.__class__(self._path.with_suffix(suffix))

    def match(self, path_pattern: str) -> bool:
        return self._path.match(path_pattern)

    def relative_to(self, *other: StrPath) -> Self:
        return self.__class__(self._path.relative_to(*other))

    @property
    def parent(self) -> Path:
        return self._path.parent

    @property
    def parents(self) -> Sequence[Path]:
        return self._path.parents

    def __truediv__(self, key: str | FilePath) -> Self:
        if isinstance(key, (str, LocalPath, Path)):
            return self.__class__(self._path / key)
        return NotImplemented

    def __rtruediv__(self, key: str | FilePath) -> Self:
        return self.__class__(key / self._path)

    def read_bytes(self) -> bytes:
        return self._content

    def read_text(self) -> str:
        return self._content.decode('utf8')

    def write_bytes(self, data: ReadableBuffer) -> int:
        self._content = bytes(data)
        return len(self._content)

    def write_text(
        self, data: str, encoding: str | None=None,
        errors: str | None=None, newline: str | None=None,
    ) -> int:
        # Use StringIO to handle the newline translation the same way as
        # real file writes would.
        with closing(io.StringIO(newline=newline)) as f:
            f.write(data)
            text = f.getvalue()
        self._content = text.encode(encoding or 'utf8', errors or 'strict')
        return len(self._content)

    def __fspath__(self) -> str:
        return str(self._path)
