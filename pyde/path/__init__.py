from pathlib import Path
from typing import TypeAlias, TypeVar, overload
from functools import lru_cache

from .filepath import (  # type: ignore
    FilePath,
    LocalPath,
    PydePath,
    ReadableBuffer,
    ReadablePath,
    WriteablePath,
)
from .url import UrlPath
from .virtual import VirtualPath

AnyRealPath: TypeAlias = Path | LocalPath
AnySource: TypeAlias = str | Path | ReadablePath
AnyDest: TypeAlias = str | Path | WriteablePath
_R = TypeVar('_R', bound=ReadablePath)
_RW = TypeVar('_RW', bound=WriteablePath)


@overload
def source(path: _R) -> _R: ...
@overload
def source(path: str | Path) -> LocalPath: ...
@lru_cache
def source(path: AnySource) -> ReadablePath:
    """Converts source paths to their correct type"""
    if isinstance(path, Path):
        return LocalPath(path)
    if isinstance(path, ReadablePath):
        return path
    return LocalPath(str(path))


@overload
def dest(path: _RW) -> _RW: ...
@overload
def dest(path: str | Path) -> LocalPath: ...
@lru_cache
def dest(path: AnyDest) -> WriteablePath:
    """Converts destination paths to their correct type"""
    if isinstance(path, Path):
        return LocalPath(path)
    if isinstance(path, WriteablePath):
        return path
    return LocalPath(str(path))


__all__ = [
    'AnyRealPath',
    'AnySource',
    'AnyDest',
    'FilePath',
    'LocalPath',
    'ReadableBuffer',
    'ReadablePath',
    'PydePath',
    'UrlPath',
    'VirtualPath',
    'WriteablePath',
    'source',
    'dest',
]
