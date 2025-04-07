from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import TypeAlias, cast

import yaml
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore


YamlType: TypeAlias = (
    str | int | float | bool | datetime | list['YamlType'] | dict[str, 'YamlType']
)


class AutoDate:
    def __init__(self, when: str | AutoDate, /):
        if isinstance(when, AutoDate):
            when = str(when)
        self._when = self.to_date_or_datetime(when)

    def __str__(self) -> str:
        datefmt, timefmt = '%Y-%m-%d', ' %H:%M:%S %z'
        if isinstance(self._when, datetime):
            return self._when.strftime(datefmt + timefmt)
        return self._when.strftime(datefmt)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self})'

    def __lt__(self, other: AutoDate | date | datetime) -> bool:
        if not isinstance(other, AutoDate):
            other = AutoDate(str(other))
        return self.datetime < AutoDate(other).datetime

    def __le__(self, other: AutoDate | date | datetime) -> bool:
        if not isinstance(other, AutoDate):
            other = AutoDate(str(other))
        return self.datetime <= AutoDate(other).datetime

    def __gt__(self, other: AutoDate | date | datetime) -> bool:
        if not isinstance(other, AutoDate):
            other = AutoDate(str(other))
        return self.datetime > AutoDate(other).datetime

    def __ge__(self, other: AutoDate | date | datetime) -> bool:
        if not isinstance(other, AutoDate):
            other = AutoDate(str(other))
        return self.datetime >= AutoDate(other).datetime

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (AutoDate, date, datetime)):
            return False
        if not isinstance(other, AutoDate):
            other = AutoDate(str(other))
        return self.datetime == AutoDate(other).datetime

    @staticmethod
    def to_date_or_datetime(dt: str) -> datetime | date:
        if dt == 'now':
            return datetime.now(timezone.utc)
        if dt == 'today':
            return date.today()
        try:
            return date.fromisoformat(dt)
        except ValueError:
            return datetime.fromisoformat(dt).replace(tzinfo=timezone.utc)

    @property
    def datetime(self) -> datetime:
        try:
            dt = cast(datetime, self._when)
            if (dt.hour, dt.minute, dt.second) != (0, 0, 0):
                return dt
        except AttributeError:
            pass
        year, month, day, *_ = self._when.timetuple()
        return (
            datetime(year, month, day)
                .replace(hour=18)
                .replace(tzinfo=timezone.utc)
        )

    @property
    def date(self) -> date:
        year, month, day, *_ = self._when.timetuple()
        return date(year, month, day)


def parse_yaml_dict(yaml_str: str) -> Mapping[str, YamlType | AutoDate]:
    yaml_dict = yaml.load(yaml_str, Loader=Loader)
    if not isinstance(yaml_dict, Mapping):
        return {}
    return _transform_types(cast(dict[str, YamlType], yaml_dict))


def _transform_types(data: dict[str, YamlType]) -> dict[str, YamlType | AutoDate]:
    dates_fixed = (
        (key, (AutoDate(val.isoformat()) if isinstance(val, (datetime, date)) else val))
        for key, val in data.items()
    )
    nones_iterable = (
        (key, ([] if val is None else val)) for key, val in dates_fixed
    )
    result = nones_iterable
    return dict(result)
