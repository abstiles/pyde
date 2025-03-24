"""
Common utility functions
"""
from collections import deque
from collections.abc import Mapping, Reversible, Sequence
from dataclasses import fields, is_dataclass
from itertools import chain, count
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Never,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


def flatmap(f: Callable[[T], Iterable[U]], it: Iterable[T]) -> Iterable[U]:
    """Map, then flatten the contents"""
    return chain.from_iterable(map(f, it))


# Avoid constructing a deque each time, reduces fixed overhead enough
# that this beats the sum solution for all but length 0-1 inputs
consumeall: Callable[[Iterable[Any]], None] = deque(maxlen=0).extend

def ilen(it: Iterable[Any]) -> int:
    # Make a stateful counting iterator
    cnt = count()
    # zip it with the input iterator, then drain until input exhausted at C level
    consumeall(zip(it, cnt)) # cnt must be second zip arg to avoid advancing too far
    # Since count 0 based, the next value is the count
    return next(cnt)


def prepend(value: Any, it: Iterable[T]) -> Iterable[T]:
    return chain([value], it)


def not_none(it: U | None) -> TypeGuard[U]:
    return it is not None


@overload
def dictfilter(d: dict[T, U | None]) -> dict[T, U]: ...

@overload
def dictfilter(
    d: dict[T, U | None], *,
    keys: Callable[[T], bool] | None,
) -> dict[T, U]: ...

@overload
def dictfilter(
    d: dict[T, U | None], *,
    keys: Callable[[T], bool] | None,
    vals: Callable[[U | None], bool] | None,
) -> dict[T, U | None]: ...

def dictfilter(  # type: ignore
    d: dict[T, U | None], *,
    keys: Callable[[T], bool] | None=None,
    vals: Callable[[U | None], bool] | None=not_none,
) -> dict[T, U | None]:
    if keys is not None and vals is not None:
        return {k: v for (k, v) in d.items() if keys(k) and vals(v)}
    if keys is not None:
        return {k: v for (k, v) in d.items() if keys(k)}
    if vals is not None:
        return {k: v for (k, v) in d.items() if vals(v)}
    return d


class Unset:
    __slots__ = ()


class UnsetMeta(type):
    def __new__(mcs, repr_str: str='<Unset>') -> type:
        return super().__new__(  # pylint: disable=unused-variable
            mcs, 'UnsetType', (Unset,), {
                '__slots__': ('_instance', 'repr_str'),
                '__repr__': lambda s: s.repr_str,
                '__iter__': lambda _: iter(()),
                '__bool__': lambda _: False,
            }
        )
    def __init__(cls, repr_str: str='<Unset>'):
        cls.repr_str = repr_str
        cls._instance = type.__call__(cls)
    def __call__(cls: type[T]) -> T:
        return cls._instance  # type: ignore


class Maybe(Generic[T]):
    @overload
    def __init__(self: 'Maybe[Never]', value: Unset): ...
    @overload
    def __init__(self, value: T): ...
    def __init__(self, value: T | Unset):
        self.__it = () if isinstance(value, Unset) else (value,)

    def __iter__(self) -> Iterator[Never] | Iterator[T]:
        return iter(self.__it)

    def __bool__(self) -> bool:
        return bool(self.__it)

    @overload
    def map(self: 'Maybe[Never]', f: Callable[..., Any]) -> 'Maybe[Never]': ...
    @overload
    def map(self, f: Callable[[T], U]) -> 'Maybe[U]': ...
    def map(self, f: Callable[[T], U]) -> 'Maybe[U] | Maybe[Never]':
        if self.__it:
            return Maybe(f(self.__it[0]))
        return cast(Maybe[Never], self)

    @overload
    def flatmap(self: 'Maybe[Never]', f: Callable[..., Any]) -> 'Maybe[Never]': ...
    @overload
    def flatmap(self, f: Callable[[T], 'Maybe[U]']) -> 'Maybe[U]': ...
    def flatmap(self, f: Callable[[T], 'Maybe[U]']) -> 'Maybe[U] | Maybe[Never]':
        if self.__it:
            return f(self.__it[0])
        return cast(Maybe[Never], self)


if TYPE_CHECKING:
    class RaiseValueError(Unset): pass
else:
    RaiseValueError = UnsetMeta('<raise ValueError>')

@overload
def first(it: Iterable[T], /) -> T: ...
@overload
def first(it: Iterable[T], default: RaiseValueError, /) -> T: ...
@overload
def first(it: Iterable[T], default: U, /) -> T | U: ...
def first(
    it: Iterable[T],
    default: T | U | RaiseValueError = RaiseValueError(),
) -> T | U:
    try:
        return next(iter(it), *Maybe(default))
    except StopIteration:
        raise ValueError('Empty iterable has no first element') from None


@overload
def last(it: Iterable[T], /) -> T: ...
@overload
def last(it: Iterable[T], default: RaiseValueError, /) -> T: ...
@overload
def last(it: Iterable[T], default: U, /) -> T | U: ...
def last(
    it: Iterable[T],
    default: T | U | RaiseValueError = RaiseValueError(),
) -> T | U:
    try:
        if isinstance(it, Sequence):
            it = it[-1:]
        if isinstance(it, Reversible):
            it = reversed(it)
        else:
            container: deque[T] = deque(maxlen=1)
            container.extend(it)
            it = container
        return first(it, default)
    except (IndexError, ValueError):
        raise ValueError('Empty iterable has no last element') from None


if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    DC_T = TypeVar('DC_T', bound=DataclassInstance)

def dict_to_dataclass(cls: type['DC_T'], data: Mapping[str, Any]) -> 'DC_T':
    types = {
        field.name: field.type for field in fields(cls)
        if isinstance(field.type, (type, GenericAlias))
    }

    def coerce(cls: type['DC_T'], val: Any) -> 'DC_T':
        if isinstance(val, Mapping):
            if is_dataclass(cls):
                return dict_to_dataclass(cls, val)
            return val
        if isinstance(val, Iterable) and getattr(cls, '__args__', None):
            contained_type = cls.__args__[0]  # type: ignore
            if is_dataclass(contained_type):
                return [coerce(contained_type, it) for it in val]  # type: ignore
            return [  # type: ignore
                it if isinstance(it, contained_type) else contained_type(it)
                for it in val
            ]
        if isinstance(val, cls):
            return val
        return cls(val)  # type: ignore

    coerced_data: dict[str, Any] = {
        key: coerce(types.get(key, object), val)
        for key, val in data.items()
    }
    return cls(**coerced_data)
