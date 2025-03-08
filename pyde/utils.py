"""
Common utility functions
"""

from itertools import chain
from typing import Callable, Iterable, TypeVar


T = TypeVar('T')
U = TypeVar('U')


def flatmap(f: Callable[[T], Iterable[U]], it: Iterable[T]) -> Iterable[U]:
    """Map, then flatten the contents"""
    return chain.from_iterable(map(f, it))
