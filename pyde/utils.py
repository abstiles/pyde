"""
Common utility functions
"""
import re
from collections import deque
from collections.abc import Generator, Mapping, Reversible, Sequence
from dataclasses import fields, is_dataclass
from itertools import chain, count, islice
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import ParamSpec

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
T_co = TypeVar('T_co', covariant=True)
U_co = TypeVar('U_co', covariant=True)
V_co = TypeVar('V_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)
P = ParamSpec('P')
F = TypeVar('F', bound=Callable[..., Any])


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
    __it: T_co | None

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

    def __init__(self, value: T_co | None):
        self.__it = value

    def __repr__(self) -> str:
        if self is Maybe.no():
            return 'Maybe.NOT'
        return f'Maybe({self.__it!r})'

    def __iter__(self) -> Iterator[T_co]:
        if self.__it is None:
            return iter(())
        return iter((self.__it,))

    def __bool__(self) -> bool:
        return self is not Maybe.no()

    def get(self, default: U=cast(Any, RaiseValueError())) -> T_co | U:
        if self.__it is not None:
            return self.__it
        if RaiseValueError.is_not(default):
            return default
        raise ValueError(f'{self} has no value')

    def or_maybe(self, other: 'Maybe[U]') -> 'Maybe[T_co | U]':
        if self.__it is None:
            return other
        return self

    def map(self, f: Callable[[T_co], U]) -> 'Maybe[U]':
        if self.__it is not None:
            return Maybe(f(self.__it))
        return cast(Maybe[U], self)

    def flatmap(self, f: Callable[[T_co], 'Maybe[U]']) -> 'Maybe[U]':
        if self.__it is not None:
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
) -> dict[K1 | K2, V1 | V2]:
    d1 = cast(Mapping[K1 | K2, V1], orig)
    d2 = cast(Mapping[K1 | K2, V2], update)
    return {
        k: merge_dicts(*dicts) if _both_mapping(dicts := (d1.get(k), d2.get(k)))
        else d2.get(k, d1.get(k))  # type: ignore
        for k in set(d1) | set(d2)
    }


@overload
def seq_pivot(
    seq: Iterable[Mapping[T, U]], index: T, /,
) -> Mapping[U, Sequence[Mapping[T, U]]]: ...
@overload
def seq_pivot(
    seq: Iterable[T], /, *, attr: str,
) -> Mapping[Any, Sequence[T]]: ...
def seq_pivot(
    seq: Iterable[Mapping[T, U]] | Iterable[V], index: T | None=None,
    /, *, attr: str='',
) -> Mapping[U, Sequence[Mapping[T, U]]] | Mapping[Any, Sequence[V]]:
    if attr:
        seq = cast(Iterable[V], seq)
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


class Predicate(Protocol, Generic[T_contra]):
    def __call__(self, arg: T_contra, /) -> bool: ...


@overload
def bucketize(  # pyright: ignore
    it: Iterable[T], /, *filters: Predicate[T],
) -> Sequence[Sequence[T]]: ...
@overload
def bucketize(
    it: Iterable[T], /, **kwfilters: Predicate[T],
) -> Mapping[str, Sequence[T]]: ...
@overload
def bucketize(
    it: Iterable[T], filter: Predicate[T], /, *filters: Predicate[T],
    **kwfilters: Predicate[T],
) -> Mapping[int | str, Sequence[T]]: ...
def bucketize(
    it: Iterable[T], /, *filters: Predicate[T],
    **kwfilters: Predicate[T],
) -> (
    Sequence[Sequence[T]]
    | Mapping[str, Sequence[T]]
    | Mapping[int | str, Sequence[T]]
):
    """
    Arrange an iterable into non-overlapping buckets

    Buckets are identified by predicates provided as either positional
    arguments or keyword arguments. If only positional arguments are provided,
    the result is a sequence where each position in the sequence contains the
    items from the iterable which matched the corresponding predicate from the
    arguments. If keyword arguments are provided, the result is a mapping
    between the corresponding keywords and the matching iterable items. If
    positional arguments are also provided, their index will be their keys.

    Only the first matching predicate is considered. Positional predicates come
    before keyword predicates.

    """
    results: Mapping[str | int, list[T]] | Mapping[int, list[T]]
    filter_funcs: list[tuple[str | int, Callable[[T], bool]]] = [
        *enumerate(filters), *kwfilters.items()
    ]
    if kwfilters:
        results = (
            { idx: [] for idx in range(len(filters)) }
            | { key: [] for key in kwfilters }
        )
    else:
        results = cast(
            Mapping[str | int, list[T]],
            tuple([] for idx in range(len(filters)))
        )
    for item in it:
        for key, filter_func in filter_funcs:
            if filter_func(item):
                results[key].append(item)
                break
    return results


@overload
def iter_buckets(
    it: Iterable[T], /, *filters: Predicate[T],
) -> Iterable[tuple[int, T]]: ...
@overload
def iter_buckets(
    it: Iterable[T], /, **kwfilters: Predicate[T],
) -> Iterable[tuple[str, T]]: ...
@overload
def iter_buckets(
    it: Iterable[T], filter: Predicate[T], /, *filters: Predicate[T],
    **kwfilters: Predicate[T],
) -> Iterable[tuple[int | str, T]]: ...
def iter_buckets(
    it: Iterable[T], /, *filters: Predicate[T],
    **kwfilters: Predicate[T],
) -> Iterable[tuple[int | str, T]]:
    """
    Make an iterable that categorizes elements from multiple predicates

    This works similarly to `groupby`, but with more flexibility. Predicates
    are provided as either positional arguments or keyword arguments, and the
    iterated items are `(id, item)` pairs, where `id` is the 0-based index of
    the positional predicate that matched or the keyword of the keyword
    predicate that matched.

    Only the first matching predicate is considered. Positional predicates come
    before keyword predicates.

    """
    filter_funcs = [*enumerate(filters), *kwfilters.items()]
    for item in it:
        for key, filter_func in filter_funcs:
            if filter_func(item):
                yield key, item  # type: ignore
                break


class CaseInsensitiveStr(str):
    def __hash__(self) -> int:
        return hash(super().lower())

    def __eq__(self, other: object, /) -> bool:
        try:
            other = cast(str, other)
            return super().lower() == other.lower()
        except (AttributeError, TypeError):
            return False


def slugify(text: str) -> str:
    """Replace bad characters for use in a path"""
    return re.sub(
        '[^a-z0-9-]+', '-',
        text.lower().replace("'", ""),
    ).strip(' -')


class ReturningGenerator(Generic[T, U]):
    """A helper for iterating a generator and getting its return value"""
    def __init__(self, generator: Generator[T, Any, U]):
        self._generator = generator
        self._iterated = False
        self._value = cast(U, None)

    def __iter__(self) -> Iterator[T]:
        if not self._iterated:
            self._value = yield from self._generator
            self._iterated = True
        else:
            yield from iter(())

    @property
    def value(self) -> U:
        if not self._iterated:
            consumeall(self._generator)
        return self._value


def batched(iterable: Iterable[T], n: int) -> Iterable[Sequence[T]]:
    """batched('ABCDEFG', 3) → ABC DEF G"""
    if n < 1:
        yield tuple(iterable)
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


class NullMapping(Mapping[Any, Any]):
    def __len__(self) -> int:
        return 0
    def __iter__(self) -> Iterator[Any]:
        return iter(())
    def __getitem__(self, key: Any) -> Any:
        raise KeyError(key)


TO_FORMAT_STR_RE = re.compile(r':(\w+)')
def format_permalink(
    permalink: str,
    values: Mapping[str, str]=NullMapping(),
    **kwargs: str,
) -> str:
    fmt = TO_FORMAT_STR_RE.sub('{\\1}', permalink)
    return fmt.format(**{**values, **kwargs})
