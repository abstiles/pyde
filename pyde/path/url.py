"""URL manipulation utilities"""

from __future__ import annotations

import collections.abc
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain
from os import PathLike
from pathlib import PurePosixPath
from typing import Any, Literal, NamedTuple, NewType, TypeAlias, Union, cast, overload
from urllib.parse import parse_qs, urlencode, urlparse

from ..utils import dictfilter
from .filepath import FilePath, PydePath

PathType: TypeAlias = Union[PydePath, PathLike[str], 'UrlPath']
UrlStr = NewType('UrlStr', str)
UrlQuoted = UrlStr | Literal['', '/']
QueryStr = NewType('QueryStr', str)
QueryEncoded = QueryStr | Literal['']
Unquoted = NewType('Unquoted', str)


class ParsedUrl(NamedTuple):
    """A 6-tuple containing the URL-quoted components of a parsed URL"""
    scheme: str = ''
    netloc: str = ''
    path: UrlQuoted = '/'
    params: UrlQuoted = ''
    query: QueryEncoded = ''
    fragment: UrlQuoted = ''

    @classmethod
    def parse(cls, url: PathType, quote: bool=True) -> ParsedUrl:
        scheme, netloc, path, params, query, fragment = urlparse(str(url))
        if quote:
            path, params, fragment = map(urlquote, (path, params, fragment))
            query = queryquote(query)
        else:
            path, params, fragment = map(UrlStr, (path, params, fragment))
            query = QueryStr(query)
        return cls(scheme, netloc, path, params, query, fragment)

    def __str__(self) -> str:
        return urllib.parse.ParseResult(*self).geturl()

    def geturl(self) -> UrlQuoted:
        return UrlStr(str(self))


class UrlPath(FilePath):  # pylint: disable=too-many-public-methods
    """Represents a URL"""

    @classmethod
    def _parse_url(
        cls,
        url: PathType | ParsedUrl='',
        quote: bool = True,
    ) -> ParsedUrl:
        match url:
            case ParsedUrl(): return url
            case UrlPath(): return url.url_tuple
            case PathLike(): return ParsedUrl.parse(url, quote=quote)
            case _: return ParsedUrl.parse(str(url), quote=quote)

    def __init__(
        self,
        url: PathType | ParsedUrl='',
        *,
        quote: bool=True,
        abbreviate: bool=True
    ):
        self.url_tuple = self._parse_url(url, quote)
        self._path = PurePosixPath(self.url_tuple.path)
        self._abbreviate = abbreviate

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UrlPath):
            return self.url_tuple == other.url_tuple
        return False

    def __hash__(self) -> int:
        return hash(self.url_tuple)

    def __str__(self) -> str:
        if self._abbreviate and self.stem == 'index':
            return str(self.parent)
        return self.encoded()

    def encoded(self) -> UrlQuoted:
        return self.url_tuple.geturl()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self}')"

    @overload
    def _updated(self, *, scheme: str) -> UrlPath: ...
    @overload
    def _updated(self, *, netloc: str) -> UrlPath: ...
    @overload
    def _updated(self, *, path: UrlQuoted) -> UrlPath: ...
    @overload
    def _updated(self, *, params: UrlQuoted) -> UrlPath: ...
    @overload
    def _updated(self, *, query: QueryEncoded) -> UrlPath: ...
    @overload
    def _updated(self, *, fragment: UrlQuoted) -> UrlPath: ...
    @overload
    def _updated(
        self, *,
        scheme: str | None = None,
        netloc: str | None = None,
        path: UrlQuoted | None = None,
        params: UrlQuoted | None = None,
        query: QueryEncoded | None = None,
        fragment: UrlQuoted | None = None,
    ) -> UrlPath: ...

    def _updated(self, **kwargs: Any) -> UrlPath:
        updates = dictfilter(kwargs)
        if updates and self.url_tuple != (
            new_tuple := self.url_tuple._replace(**updates)
        ):
            return UrlPath(new_tuple)
        return self

    @property
    def query(self) -> QueryMap:
        query_params = parse_qs(self.url_tuple.query)
        return QueryMap(self, query_params)

    @property
    def proto(self) -> str | None:
        return self.url_tuple.scheme or None

    @property
    def host(self) -> str | None:
        return self.url_tuple.netloc or None

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def stem(self) -> str:
        return self._path.stem

    @property
    def suffix(self) -> str:
        return self._path.suffix

    @property
    def path(self) -> UrlQuoted:
        return self.url_tuple.path or '/'

    @property
    def dir(self) -> UrlPath:
        return self if self.is_dir else self.parent

    @property
    def parent(self) -> UrlPath:
        parent_path = str(self._path.parent)
        return self._updated(path=UrlStr(parent_path)).as_dir()

    @property
    def parents(self) -> Sequence[UrlPath]:
        parents = str(self._path.parent)
        return [
            self._updated(path=UrlStr(parent)).as_dir()
            for parent in parents
        ]

    def __rtruediv__(self, parent: PathType) -> UrlPath:
        return UrlPath(parent) / self

    def __truediv__(self, child: PathType) -> UrlPath:
        return self.child(child)

    def __rfloordiv__(self, parent: PathType) -> UrlPath:
        return UrlPath(parent) // self

    def __floordiv__(self, new_path: PathType) -> UrlPath:
        if isinstance(new_path, UrlPath):
            url_path = new_path
        else:
            url_path = UrlPath(str(new_path))
        return self.navigate(url_path.absolute())

    def match(self, path_pattern: str) -> bool:
        return self._path.match(path_pattern)

    def child(self, key: PathType) -> UrlPath:
        return self.as_dir().navigate(key)

    def absolute(self, from_path: PathType='/') -> UrlPath:
        root = urlquote(from_path)
        path = urljoin(root, self.path)
        if path == self.path:
            return self
        return self._updated(path=path)

    @property
    def is_dir(self) -> bool:
        path = self._dir_str(self.path)
        return path == self.path

    def as_dir(self) -> UrlPath:
        path = self._dir_str(self.path)
        return (
            self if path == self.path
            else self._updated(path=path, query='')
        )

    @staticmethod
    def _dir_str(path: UrlQuoted) -> UrlQuoted:
        if path.endswith('/'):
            return path
        return UrlStr(path + '/')

    def __matmul__(self, fragment: str) -> UrlPath:
        return self._updated(fragment=urlquote(fragment))

    def __rshift__(self, path: PathType) -> UrlPath:
        return self.navigate(path)

    def navigate(self, path: PathType) -> UrlPath:
        url_path = UrlPath(path)
        new_proto = url_path.proto or self.proto
        new_host = url_path.host or self.host
        if new_proto != self.proto or new_host != self.host:
            return url_path._updated(scheme=new_proto, netloc=new_host)
        new_path = urljoin(self.path, url_path.path)
        return self._updated(path=new_path, query='', fragment='')

    def with_suffix(self, suffix: str) -> UrlPath:
        return self._updated(path=UrlStr(str(self._path.with_suffix(suffix))))

    def with_query(self, query: str) -> UrlPath:
        return self._updated(query=queryquote(query))

    @staticmethod
    def _unquote(path_part: UrlQuoted) -> str:
        return urllib.parse.unquote(path_part)

    def __and__(self, param: str | tuple[str, Any] | Mapping[str, Any]) -> UrlPath:
        def split_assign(assign: str) -> tuple[str, str]:
            name, value = assign.partition('=')[::2]
            return name, value
        if isinstance(param, Mapping):
            return self.query.with_added(param)
        if isinstance(param, str):
            params = param.split('&')
            pairs = (split_assign(param) for param in params)
            return self.query.with_added(pairs)
        if len(param) > 1 and isinstance(param[0], str):
            return self.query.with_added([param])
        return self.query.with_added(param)

    def __xor__(self, param: str) -> UrlPath:
        return self.query.with_removed(param)

    def relative_to(self, *other: PydePath) -> UrlPath:
        try:
            path = UrlPath(other[0])
        except KeyError:
            raise TypeError('need at least one argument') from None
        for segment in other[1:]:
            path = path / segment
        path_dir = UrlPath(path).dir
        new_path = str(self._path.relative_to(path_dir.path))
        return self._updated(path=UrlStr(new_path))


class QueryMap(Mapping[str, Sequence[str]]):
    """Immutable map of query parameters"""
    Pair: TypeAlias = tuple[str, Any]

    @overload
    def __init__(self, url: UrlPath, data: Mapping[str, Any], /): ...
    @overload
    def __init__(self, url: UrlPath, data: Iterable[Pair], /): ...
    @overload
    def __init__(self, url: UrlPath, **kwargs: Any): ...
    def __init__(
        self, url: UrlPath, *data: Mapping[str, Any] | Iterable[Pair],
        **kwargs: Any
    ):
        self.full_url = url
        temp = self.arg_dict(*data, **kwargs)
        self._keys = frozenset(temp.keys())
        self._values = tuple(tuple(value) for value in temp.values())
        self._idx_map = {key: idx for idx, key in enumerate(temp.keys())}

    @staticmethod
    def iteritems(data: Mapping[str, Any] | Iterable[Pair]) -> Iterable[Pair]:
        if isinstance(data, Mapping):
            data = cast(Mapping[str, Any], data)
            return ((k, data[k]) for k in data)
        return ((key, val) for (key, val, *_) in data)

    @staticmethod
    def arg_dict(
        *data: Mapping[str, Any] | Iterable[Pair], **kwargs: Any
    ) -> dict[str, list[str]]:
        kv_pairs = chain.from_iterable(map(QueryMap.iteritems, data))
        d: dict[str, list[str]] = {}
        for key, val in chain(kv_pairs, kwargs.items()):
            if isinstance(val, str) or not isinstance(val, Iterable):
                val = [val]
            d.setdefault(key, []).extend(map(str, val))
        return d

    def items(self) -> collections.abc.ItemsView[str, Sequence[str]]:
        return collections.abc.ItemsView(self)

    def values(self) -> collections.abc.ValuesView[Sequence[str]]:
        return collections.abc.ValuesView(self)

    def keys(self) -> collections.abc.KeysView[str]:
        return collections.abc.KeysView(self)

    def __getitem__(self, key: str) -> Sequence[str]:
        return self._values[self._idx_map[key]]

    def __str__(self) -> str:
        return str(query_encode(self))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.to_dict()})'

    def encoded(self) -> QueryEncoded:
        return QueryStr(str(self))

    def with_removed(self, *keys: str) -> UrlPath:
        removed = set(self._idx_map[key] for key in keys)
        return self._construct_url([
            pair for idx, pair in enumerate(self.items())
            if idx not in removed
        ])

    def with_update(
        self, *data: Mapping[str, Any] | Iterable[Pair], **kwargs: Any
    ) -> UrlPath:
        updates = self.arg_dict(*data, **kwargs)
        return self._construct_url([
            (k, v) if k not in updates else (k, updates[k])
            for (k, v) in self.items()
        ])

    def with_set(self, key: str, new_value: str | Iterable[str]) -> UrlPath:
        return self.with_update([(key, new_value)])

    def with_added(
        self, *data: Mapping[str, Any] | Iterable[Pair], **kwargs: Any
    ) -> UrlPath:
        return self._construct_url(
            chain(self.items(), self.arg_dict(*data, **kwargs).items())
        )
    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def to_dict(self) -> dict[str, str | list[str]]:
        return dict(
            (k, list(v)) if len(v) > 1 else (k, v[0])
            for k, v in self.items() if len(v) > 0
        )

    def _construct_url(
        self, *data: Mapping[str, Any] | Iterable[Pair], **kwargs: Any
    ) -> UrlPath:
        new_query = QueryMap(self.full_url, *data, **kwargs)
        return self.full_url.with_query(str(new_query))


def urlquote(part: PathType) -> UrlQuoted:
    if isinstance(part, UrlPath):
        return part.encoded()
    part_str = part if isinstance(part, str) else str(part)
    return UrlStr(urllib.parse.quote(part_str))


def urlunquote(path_part: UrlQuoted) -> str:
    return urllib.parse.unquote(path_part)


def urljoin(base: UrlQuoted, part: PathType) -> UrlQuoted:
    quoted = urlquote(part)
    return UrlStr(urllib.parse.urljoin(base, quoted, allow_fragments=True))


def queryquote(param_str: str | Unquoted) -> QueryEncoded:
    # Requoting an already quoted query string is idempotent.
    return query_encode(parse_qs(param_str))


def query_encode(query: Mapping[str, Any]) -> QueryEncoded:
    return QueryStr(urlencode(query, doseq=True))
