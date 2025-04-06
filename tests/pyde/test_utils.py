from dataclasses import asdict, dataclass
from pathlib import Path

from pyde.utils import Predicate, bucketize, dict_to_dataclass, iter_buckets, seq_pivot


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

def test_seq_pivot() -> None:
    seq = [
        {'type': 'one', 'value': 1},
        {'type': 'two', 'value': 2},
        {'type': 'three', 'value': 3},
        {'type': 'one', 'value': 4},
        {'type': 'two', 'value': 5},
        {'type': 'three', 'value': 6},
    ]
    expected = {
        'one': [
            {'type': 'one', 'value': 1},
            {'type': 'one', 'value': 4},
        ],
        'two': [
            {'type': 'two', 'value': 2},
            {'type': 'two', 'value': 5},
        ],
        'three': [
            {'type': 'three', 'value': 3},
            {'type': 'three', 'value': 6},
        ],
    }
    assert seq_pivot(seq, 'type') == expected

def test_bucketize() -> None:
    seq = [
        {'type': 'one', 'value': 1},
        {'type': 'two', 'value': 2},
        {'type': 'three', 'value': 3},
        {'type': 'one', 'value': 4},
        {'type': 'two', 'value': 5},
        {'type': 'three', 'value': 6},
    ]
    expected = {
        'one': [
            {'type': 'one', 'value': 1},
            {'type': 'one', 'value': 4},
        ],
        'two': [
            {'type': 'two', 'value': 2},
            {'type': 'two', 'value': 5},
        ],
        'three': [
            {'type': 'three', 'value': 3},
            {'type': 'three', 'value': 6},
        ],
        'four': [],
    }

    types = ('one', 'two', 'three', 'four')
    calls = {type: 0 for type in types}
    def by_type(type: str) -> tuple[str, Predicate[dict[str, object]]]:
        def filterer(d: dict[str, object]) -> bool:
            calls[type] += 1
            return d['type'] == type
        return (type, filterer)
    assert bucketize(
        seq, **dict(map(by_type, types))
    ) == expected
    assert calls['one'] == 6
    assert calls['two'] == 4
    assert calls['three'] == 2
    assert calls['four'] == 0

def test_iter_buckets() -> None:
    seq = [
        {'type': 'one', 'value': 1},
        {'type': 'two', 'value': 2},
        {'type': 'three', 'value': 3},
        {'type': 'one', 'value': 4},
        {'type': 'two', 'value': 5},
        {'type': 'three', 'value': 6},
    ]
    expected = [
        ('one', {'type': 'one', 'value': 1}),
        ('two', {'type': 'two', 'value': 2}),
        ('three', {'type': 'three', 'value': 3}),
        ('one', {'type': 'one', 'value': 4}),
        ('two', {'type': 'two', 'value': 5}),
        ('three', {'type': 'three', 'value': 6}),
    ]

    types = ('one', 'two', 'three', 'four')
    calls = {type: 0 for type in types}
    def by_type(type: str) -> tuple[str, Predicate[dict[str, object]]]:
        def filterer(d: dict[str, object]) -> bool:
            calls[type] += 1
            return d['type'] == type
        return (type, filterer)
    assert [*iter_buckets(
        seq, **dict(map(by_type, types))
    )] == expected
    assert calls['one'] == 6
    assert calls['two'] == 4
    assert calls['three'] == 2
    assert calls['four'] == 0
