"""Immutable Map"""

from __future__ import annotations

from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from typing import Any, Generic, Hashable, Literal, Self, TypeVar, cast, overload

K = TypeVar('K', bound=Hashable)
V = TypeVar('V', bound=Hashable)


class MapItem(Generic[K, V]):
    """Hack for fetching by key"""
    __slots__ = ('key', 'value')

    key: K
    value: V

    def __init__(self, key: K, value: V):
        self.key = key
        self.value = value

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __str__(self) -> str:
        return f'{self.key!r}: {self.value!r}'

    def __repr__(self) -> str:
        return f'MapItem(key={self.key!r}, value={self.value!r})'

    @overload
    def __getitem__(self, idx: Literal[0, -2]) -> K: ...
    @overload
    def __getitem__(self, idx: Literal[1, -1]) -> V: ...
    def __getitem__(self, idx: int) -> K | V:
        return (self.key, self.value)[idx]

    @property
    def tuple(self) -> tuple[K, V]:
        return self.key, self.value

    def __iter__(self) -> Iterator[K | V]:
        return iter(self.tuple)

    @classmethod
    def keyquery(cls, key: K) -> MapItem[K, V]:
        return cls(key, cast(V, ...))


class ImmutableMap(Mapping[K, V]):
    """Immutable map of query parameters"""
    _map: frozenset[MapItem[K, V]]

    @overload
    def __init__(self, data: Mapping[K, V] | tuple[K, V], /): ...
    @overload
    def __init__(self: ImmutableMap[str, V], **kwargs: V): ...
    def __init__(self, *args: Any, **kwargs: V):
        d = dict(*args, **kwargs)
        self._map = frozenset(MapItem(key, val) for key, val in d.items())

    def items(self) -> ItemsView[K, V]:
        return ItemsView(self)

    def values(self) -> ValuesView[V]:
        return ValuesView(self)

    def keys(self) -> KeysView[K]:
        return KeysView(self)

    def __getitem__(self, key: K) -> V:
        if key not in self._map:
            raise KeyError(key)
        # Frozensets have no way of getting a value out, so we have to do a
        # shenanigan here.
        query = MapItem[K, V].keyquery(key)
        items_without_target = self._map - frozenset({query})
        set_with_only_target = self._map - items_without_target
        item = next(iter(set_with_only_target))
        return item.value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.to_dict()})'

    def __sub__(self, other: Iterable[K]) -> Self:
        return ImmutableMap(self._map - set(other))  # type: ignore

    def __add__(self, other: Mapping[K, V]) -> Self:
        overrides = [*other]
        updates = other if isinstance(other, ImmutableMap) else ImmutableMap(other)
        return ImmutableMap((self._map - overrides) | updates._map)  # type: ignore

    def __iter__(self) -> Iterator[K]:
        return (param.key for param in self._map)

    def __len__(self) -> int:
        return len(self._map)

    def to_dict(self) -> dict[K, V]:
        return dict(self.items())
