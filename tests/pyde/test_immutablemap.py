from pyde.immutablemap import ImmutableMap


def test_construct_from_dict_arg() -> None:
    data = {'key': 'value'}
    im = ImmutableMap(data)
    assert im['key'] == 'value'
    assert im.to_dict() == data


def test_construct_from_iterable_of_pairs() -> None:
    data = [('key', 'value')]
    im = ImmutableMap(data)
    assert im['key'] == 'value'
    assert [*im.items()] == data


def test_construct_from_kwargs() -> None:
    im = ImmutableMap(key='value')
    assert im['key'] == 'value'
    assert [*im.keys()] == ['key']
    assert [*im.values()] == ['value']


def test_or() -> None:
    data = {'key': 'value'}
    other = {'answer': 42}
    result = ImmutableMap(data) | other
    assert result == {**data, **other}


def test_sub() -> None:
    data = {'key': 'value'}
    other = {'answer': 42}
    result = ImmutableMap({**data, **other}) - other
    assert result == data
