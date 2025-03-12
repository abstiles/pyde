"""
Common utility functions
"""

from itertools import chain
from typing import Any, Callable, Iterable, TypeVar


T = TypeVar('T')
U = TypeVar('U')


def flatmap(f: Callable[[T], Iterable[U]], it: Iterable[T]) -> Iterable[U]:
    """Map, then flatten the contents"""
    return chain.from_iterable(map(f, it))

from collections import deque
from itertools import count

# Avoid constructing a deque each time, reduces fixed overhead enough
# that this beats the sum solution for all but length 0-1 inputs
consumeall = deque(maxlen=0).extend

def ilen(it: Iterable[Any]) -> int:
    # Make a stateful counting iterator
    cnt = count()
    # zip it with the input iterator, then drain until input exhausted at C level
    consumeall(zip(it, cnt)) # cnt must be second zip arg to avoid advancing too far
    # Since count 0 based, the next value is the count
    return next(cnt)


def prepend(value: Any, it: Iterable[T]) -> Iterable[T]:
    return chain([value], it)


def ilast(it: Iterable[T]) -> T:
    container: deque[T] = deque(maxlen=1)
    container.extend(it)
    return container[0]
