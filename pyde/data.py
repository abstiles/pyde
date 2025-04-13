"""Data models"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date, datetime, timezone
from typing import Any, NoReturn, cast

from jinja2.runtime import Undefined


class Data(Mapping[str, Any]):
    _d: dict[str, Any]

    def __init__(
        self,
        d: dict[str, Any] | None=None,
        /,
        _from: str | Undefined='',
        **kwargs: Any
    ):
        super().__setattr__('_from', _from)
        super().__setattr__('_d', d or {})
        self._d.update(kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Data):
            return False
        return self._d == other._d

    def __html__(self) -> str:
        return ''

    def __str__(self) -> str:
        return str(self._d).strip('{}')

    def __int__(self) -> int:
        if self:
            raise TypeError('Cannot cast nontrivial Data instance as int')
        return 0

    def __iter__(self) -> Iterator[str]:
        yield from iter(self._d)

    def __len__(self) -> int: # pylint: disable=invalid-length-returned
        return len(self._d)

    def __call__(self) -> NoReturn:
        raise TypeError(f'Data object {self._from} is not callable')

    def __bool__(self) -> bool:
        return bool(self._d)

    def __contains__(self, key: str) -> bool:  # type: ignore [override]
        return key in self._d

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __getattr__(self, key: str) -> Any:
        try:
            return self._d[key]
        except KeyError:
            return Data(_from=f'{self._from}.{key}')

    def __setitem__(self, key: str, val: Any) -> None:
        self._d[key] = val

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __delitem__(self, key: str) -> None:
        del self._d[key]

    def __delattr__(self, key: str) -> None:
        del self[key]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._d!r})'


class AutoDate:
    def __init__(self, when: str | date | AutoDate, /):
        self._when = (
            when if isinstance(when, date) else self.to_date_or_datetime(when)
        )

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
    def to_date_or_datetime(dt: Any) -> datetime | date:
        if not isinstance(dt, str):
            dt = str(dt)
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
