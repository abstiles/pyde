"""Immutable Map"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, KeysView, Mapping
from typing import Any, Callable, Hashable, Self, TypeVar, overload

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')
K2 = TypeVar('K2', bound=Hashable)
V2 = TypeVar('V2')


class ImmutableMap(Mapping[K, V]):
    """Like a regular map, but immutable"""

    __slots__ = ('_keys', '__getitem')

    @overload
    def __init__(self, data: Mapping[K, V] | Iterable[tuple[K, V]], /): ...
    @overload
    def __init__(self: ImmutableMap[str, V], **kwargs: V): ...
    def __init__(self, *args: Any, **kwargs: V):
        data = dict(*args, **kwargs)
        self._keys = data.keys()
        self.__getitem: Callable[[K], V] = data.__getitem__

    def __getitem__(self, key: K) -> V:
        return self.__getitem(key)

    def keys(self) -> KeysView[K]:
        return self._keys

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.to_dict()})'

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __sub__(self, other: Iterable[Any]) -> 'ImmutableMap[K, V]':
        return ImmutableMap({
            k: self[k] for k in self.keys() - set(other)
        })

    @overload
    def __or__(self, other: Mapping[K, V]) -> Self: ...
    @overload
    def __or__(self, other: Mapping[K2, V2]) -> 'ImmutableMap[K | K2, V | V2]': ...
    def __or__(self, other: Mapping[K2, V2]) -> 'ImmutableMap[K | K2, V | V2]':
        combined: dict[K|K2, V|V2] = {**self, **other}  # type: ignore
        return ImmutableMap(combined)

    def to_dict(self) -> dict[K, V]:
        return dict(self.items())
