from dataclasses import dataclass, asdict
from pathlib import Path

from pyde.utils import dict_to_dataclass


@dataclass
class Inner:
    one: int
    two: str

@dataclass
class Outer:
    inner: Inner
    path: Path

@dataclass
class WithList:
    items: list[Outer]

def test_dict_to_dataclass() -> None:
    inner1 = Inner(42, 'hello')
    inner2 = Inner(42, 'hello')
    outer1 = Outer(inner=inner1, path=Path('/path/to/somewhere.txt'))
    outer2 = Outer(inner=inner2, path=Path('/path/elsewhere.txt'))
    with_list = WithList(items=[outer1, outer2])
    assert dict_to_dataclass(WithList, asdict(with_list)) == with_list
