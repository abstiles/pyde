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
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
T_co = TypeVar('T_co', covariant=True)
U_co = TypeVar('U_co', covariant=True)
V_co = TypeVar('V_co', covariant=True)


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
    @classmethod
    def is_not(cls, value: T | 'Unset') -> TypeGuard[T]:
        return value is not cls()


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


if TYPE_CHECKING:
    class RaiseValueError(Unset): pass
else:
    RaiseValueError = UnsetMeta('<raise ValueError>')


class Maybe(Generic[T_co]):
    """A generic, optional value"""

    NOT: 'Maybe[T_co]'
    __it: T_co

    @classmethod
    def yes(cls, value: T) -> 'Maybe[T]':
        if value is not None:
            return Maybe(value)
        self = Maybe(object())
        self.__it = None
        return cast(Maybe[T], self)

    @classmethod
    def no(cls: type['Maybe[T]']) -> 'Maybe[T]':
        return Maybe.NOT  # type: ignore

    def __new__(cls, value: T_co | None) -> 'Maybe[T_co]':
        if value is None:
            if getattr(cls, 'NOT', None) is None:
                cls.NOT = object.__new__(cls)
            return cls.NOT
        return object.__new__(cls)

    def __init__(self, value: T_co):
        self.__it = value

    def __repr__(self) -> str:
        if self is Maybe.no():
            return 'Maybe.NOT'
        return f'Maybe({self.__it!r})'

    def __iter__(self) -> Iterator[T_co]:
        if self is Maybe.no():
            return iter(())
        return iter((self.__it,))

    def __bool__(self) -> bool:
        return self is not Maybe.no()

    def get(self, default: U=cast(Any, RaiseValueError())) -> T_co | U:
        if self is Maybe.no():
            return self.__it
        if RaiseValueError.is_not(default):
            return default
        raise ValueError(f'{self} has no value')

    def or_maybe(self, other: 'Maybe[U]') -> 'Maybe[T_co | U]':
        if self is Maybe.no():
            return other
        return self

    def map(self, f: Callable[[T_co], U]) -> 'Maybe[U]':
        if self is Maybe.no():
            return Maybe(f(self.__it))
        return cast(Maybe[U], self)

    def flatmap(self, f: Callable[[T_co], 'Maybe[U]']) -> 'Maybe[U]':
        if self is Maybe.no():
            return f(self.__it)
        return cast(Maybe[U], self)

Maybe(None)  # Initialize the singleton NOT instance.

@overload
def first(it: Iterable[T], /) -> T: ...
@overload
def first(it: Iterable[T], default: U, /) -> T | U: ...
def first(
    it: Iterable[T],
    default: T | U = cast(Any, RaiseValueError()),
) -> T | U:
    try:
        args = () if isinstance(default, RaiseValueError) else (default,)
        return next(iter(it), *(args))
    except StopIteration:
        raise ValueError('Empty iterable has no first element') from None


@overload
def last(it: Iterable[T], /) -> T: ...
@overload
def last(it: Iterable[T], default: U, /) -> T | U: ...
def last(
    it: Iterable[T],
    default: T | U = cast(Any, RaiseValueError()),
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
else:
    DC_T = TypeVar('DC_T')


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
        key: coerce(types.get(key, object), val)  # type: ignore
        for key, val in data.items()
    }
    return cls(**coerced_data)


K1 = TypeVar('K1')
K2 = TypeVar('K2')
V1 = TypeVar('V1')
V2 = TypeVar('V2')

def _both_mapping(
    pair: tuple[Any, Any]
) -> TypeGuard[tuple[Mapping[Any, Any], Mapping[Any, Any]]]:
    d1, d2 = pair
    return isinstance(d1, Mapping) and isinstance(d2, Mapping)

def merge_dicts(
    orig: Mapping[K1, V1],
    update: Mapping[K2, V2]
) -> Mapping[K1 | K2, V1 | V2]:
    d1 = cast(Mapping[K1 | K2, V1], orig)
    d2 = cast(Mapping[K1 | K2, V2], update)
    result: Mapping[K1 | K2, V1 | V2] = {
        k: merge_dicts(*dicts) if _both_mapping(dicts := (d1.get(k), d2.get(k)))
        else d2.get(k, d1.get(k))  # type: ignore
        for k in set(d1) | set(d2)
    }
    return result


@overload
def seq_pivot(
    seq: Sequence[Mapping[T, U]], index: T, /,
) -> Mapping[U, Sequence[Mapping[T, U]]]: ...
@overload
def seq_pivot(
    seq: Iterable[U], /, *, attr: str,
) -> Mapping[Any, Sequence[U]]: ...
def seq_pivot(  # type: ignore [misc]
    seq: Iterable[Mapping[T, U]] | Iterable[U],
    index: T | None=None,
    /, *, attr: str='',
) -> Mapping[U, Sequence[Mapping[T, U]]] | Mapping[Any, Sequence[U]]:
    if attr:
        seq = cast(Iterable[U], seq)
        return seq_pivot_object(seq, attr)
    seq = cast(Iterable[Mapping[T, U]], seq)
    return seq_pivot_mapping(seq, cast(T, index))


def seq_pivot_mapping(
    seq: Iterable[Mapping[T, U]], index: T,
) -> Mapping[U, Sequence[Mapping[T, U]]]:
    result: dict[U, list[Mapping[T, U]]]  = {}
    for item in seq:
        if index in item:
            result.setdefault(item[index], []).append(item)
    return result


def seq_pivot_object(
    seq: Iterable[U], index: str,
) -> Mapping[Any, Sequence[U]]:
    result: dict[Any, list[U]]  = {}
    for item in seq:
        if hasattr(item, index):
            result.setdefault(getattr(item, index), []).append(item)
    return result
