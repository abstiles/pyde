from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, TypeAlias, overload, runtime_checkable

from typing_extensions import Any, Buffer

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer
else:
    ReadableBuffer = Buffer

class FilePath(Protocol):
    def with_suffix(self, suffix: str) -> FilePath: ...
    def match(self, path_pattern: str) -> bool: ...
    def relative_to(self, *other: PydePath) -> FilePath: ...
    def absolute(self) -> FilePath: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __truediv__(self, key: Path | PydePath) -> FilePath: ...
    def __rtruediv__(self, key: Path | PydePath) -> FilePath: ...

    @property
    def name(self) -> str: ...
    @property
    def stem(self) -> str: ...
    @property
    def suffix(self) -> str: ...
    @property
    def parent(self) -> FilePath: ...
    @property
    def parents(self) -> Sequence[FilePath]: ...


PydePath: TypeAlias = FilePath | str


@runtime_checkable
class ReadablePath(FilePath, Protocol):
    def read_bytes(self) -> bytes: ...
    def read_text(
        self, encoding: str | None=None, errors: str | None = None
    ) -> str: ...
    @property
    def parent(self) -> 'LocalPath': ...
    @property
    def parents(self) -> Sequence['LocalPath']: ...
    def __truediv__(self, key: Path | PydePath) -> ReadablePath: ...
    def __rtruediv__(self, key: Path | PydePath) -> ReadablePath: ...
    def timestamp(self) -> datetime: ...


@runtime_checkable
class WriteablePath(ReadablePath, Protocol):
    def write_bytes(self, data: ReadableBuffer) -> int: ...
    def write_text(
        self,
        data: str,
        encoding: str | None=None,
        errors: str | None=None,
        newline: str | None=None,
    ) -> int: ...
    def __truediv__(self, key: Path | PydePath) -> WriteablePath: ...
    def __rtruediv__(self, key: Path | PydePath) -> WriteablePath: ...


class LocalPath(WriteablePath, os.PathLike[str]):
    """
    Model a path that exists on the local filesystem
    """

    def __init__(self, path: Path | PydePath):
        self._path = Path(str(path))

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({str(self._path)!r})'

    def __hash__(self) -> int:
        return hash(self._path)

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
        return self.__class__(self._path.absolute())

    def with_suffix(self, suffix: str) -> Self:
        return self.__class__(self._path.with_suffix(suffix))

    def match(self, path_pattern: str) -> bool:
        return self._path.match(path_pattern)

    def relative_to(self, *other: PydePath) -> Self:
        return self.__class__(self._path.relative_to(*map(str, other)))

    @property
    def parent(self) -> LocalPath:
        return LocalPath(self._path.parent)

    @property
    def parents(self) -> Sequence[LocalPath]:
        return [LocalPath(parent) for parent in self._path.parents]

    @overload
    def __truediv__(self, key: str | Path | LocalPath) -> LocalPath: ...
    @overload
    def __truediv__(self, key: Path | PydePath) -> WriteablePath: ...
    def __truediv__(self, key: Path | PydePath) -> WriteablePath:
        if isinstance(key, (str, Path, LocalPath)):
            return self.__class__(self._path / str(key))
        # If not one of the above, the other object should be a FilePath, and
        # probably one that implements some special behavior (like VirtualPath).
        # This method returns NotImplemented so that the __rtruediv__
        # implmentation on the other object can handle its own special behavior.
        return NotImplemented

    def __rtruediv__(self, key: Path | PydePath) -> WriteablePath:
        return self.__class__(str(key) / self._path)

    def read_bytes(self) -> bytes:
        return self._path.read_bytes()

    def read_text(
        self, encoding: str | None=None, errors: str | None = None
    ) -> str:
        return self._path.read_text(encoding, errors)

    def write_bytes(self, data: ReadableBuffer) -> int:
        return self._path.write_bytes(data)

    def write_text(
        self, data: str, encoding: str | None=None,
        errors: str | None=None, newline: str | None=None,
    ) -> int:
        return self._path.write_text(data, encoding, errors, newline)

    def __fspath__(self) -> str:
        return str(self._path)

    def mkdir(self, parents: bool=False, exist_ok: bool=False) -> None:
        self._path.mkdir(parents=parents, exist_ok=exist_ok)

    def is_dir(self) -> bool:
        return self._path.is_dir()

    def is_file(self) -> bool:
        return self._path.is_file()

    def __getattr__(self, attr: str) -> Any:
        path_attr = getattr(self._path, attr)
        # Wrap pathlib Paths in LocalPath instances.
        if isinstance(path_attr, Path):
            return self.__class__(path_attr)
        # Anything else that's not a function/method is fine.
        if not callable(path_attr):
            return path_attr
        # Decorate methods to make them also wrap pathlib Paths.
        @wraps(path_attr)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = path_attr(*args, **kwargs)
            if isinstance(result, Path):
                return self.__class__(result)
            return result
        return wrapper

    def timestamp(self) -> datetime:
        stat = self._path.stat()
        try:
            timestamp = stat.st_birthtime
        except AttributeError:
            timestamp = (
                stat.st_ctime if platform.system() == 'Windows'
                else stat.st_mtime
            )
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except OverflowError:
            # For some reason, st_birthtime on FreeBSD (at least my reference
            # system) reports this as the improbably large value of
            # 1.8446744073709552e+19 and I have no idea why. At least ctime
            # seems fine?
            return datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)


if TYPE_CHECKING:
    # Because the type checkers don't provide a nice way to declare that
    # LocalPath proxies its Path instance with getattr, let's create the
    # polite fiction that it inherits from Path.
    class _LocalPath(LocalPath, Path):  # type: ignore # pylint: disable=all
        def __new__(cls, path: Path | PydePath) -> Self: ...
        def __init__(self, path: Path | PydePath): ...
    LocalPath = _LocalPath  # type: ignore
