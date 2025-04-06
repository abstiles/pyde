from datetime import datetime
import io
import textwrap
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Self

from . import LocalPath, PydePath, ReadableBuffer, WriteablePath


class VirtualPath(WriteablePath):
    """
    Model a file that exists only in memory

    A "virtual path" represents a file that is not on the filesystem, but is
    associated with a particular path on the filesystem anyway. This allows
    a pyde.transformer.Transformer to produce dynamically generated files
    without a pre-existing source.
    """

    def __init__(self, path: Path | PydePath, content: str | bytes=b''):
        self._path = LocalPath(path)
        self._content: bytes = (
            content if isinstance(content, bytes)
            else content.encode('utf8')
        )
        self._timestamp = datetime.now()

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        as_str = self._content.decode('utf8', errors='backslashreplace')
        abbreviated = textwrap.shorten(as_str, width=100)
        return (
            f'{type(self).__name__}('
            f'{str(self._path)!r}'
            f', content={abbreviated!r}'
            ')'
        )

    def _with_path(self: Self, path: PydePath) -> Self:
        return self.__class__(path, self._content)

    def __hash__(self) -> int:
        return hash(self._path) ^ hash(self._content)

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

    def absolute(self) -> Self:
        return self._with_path(self._path.absolute())

    def with_suffix(self, suffix: str) -> Self:
        return self._with_path(self._path.with_suffix(suffix))

    def match(self, path_pattern: str) -> bool:
        return self._path.match(path_pattern)

    def relative_to(self, *other: PydePath) -> Self:
        return self._with_path(self._path.relative_to(*other))

    @property
    def parent(self) -> LocalPath:
        return self._path.parent

    @property
    def parents(self) -> Sequence[LocalPath]:
        return self._path.parents

    def __truediv__(self, key: Path | PydePath) -> Self:
        return self._with_path(self._path / key)

    def __rtruediv__(self, key: Path | PydePath) -> Self:
        return self._with_path(str(key)) / self._path

    def read_bytes(self) -> bytes:
        return self._content

    def read_text(
        self, encoding: str | None=None, errors: str | None = None
    ) -> str:
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

    def timestamp(self) -> datetime:
        return self._timestamp
