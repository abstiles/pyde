"""Data models"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, TypeAlias, cast

import yaml
from jinja2.runtime import Undefined
from markupsafe import Markup

from .config import Config, SourceFile
from .markdown import markdownify
from .url import UrlPath
from .utils import ilen


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


PARA_RE = re.compile('<p[^>]*>(.*?)</p>', flags=re.DOTALL)

@dataclass
class FileData:
    file: SourceFile
    meta: Data = field(default_factory=Data)
    has_frontmatter: bool = False
    is_binary: bool = False

    @property
    def path(self) -> Path:
        return self.file.path

    @property
    def type(self) -> str:
        """Get the file's type"""
        return ''.join(self.path.suffix).lstrip('.')

    @property
    def content(self) -> str:
        if self.is_binary:
            raise RuntimeError(
                f'Cannot extract string content of binary file {self.path}')
        return get_content(self.file.read_text())

    @classmethod
    def load(cls, file: SourceFile, root: UrlPath) -> 'FileData':
        is_binary = False
        try:
            frontmatter = get_frontmatter(file.path.read_text())
        except UnicodeDecodeError:
            frontmatter = None
            is_binary = True
        has_frontmatter = frontmatter is not None
        values = {**file.values, **parse_yaml_dict(frontmatter or '')}
        excerpt = ''
        word_count: None | int = None

        suffix = ''
        basename = file.path.name
        if file.path.suffix == '.md':
            word_count = 0
            basename = file.path.stem
            suffix = '.html'
            if not values.get('skip'):
                try:
                    html = markdownify(get_content(file.path.read_text()))
                    excerpt = PARA_RE.search(html)[0]  # type: ignore [index]
                    word_count = 1 + ilen(re.finditer(r'\s+', Markup(html).striptags()))
                except TypeError:
                    pass
        meta = Data(
            _from=f'File({file.path}).page',
            page=Data({
                'word_count': word_count,
                'excerpt': excerpt,
                **values,
            }, f'File({file.path}).page'),
            **file.values,
            path=str(file.path.parent.relative_to(file.root)),
            basename=basename,
        )
        meta.title = meta.title or meta.basename
        file_path = get_path(meta)
        full_url = root / UrlPath(file_path)
        meta.file_path = Path(f'{file_path}{suffix}')
        url = full_url.dir if full_url.stem == 'index' else full_url
        meta.page.url = str(url)
        meta.page.path = url.path
        meta.page.dir = url.dir.path
        # A stupid hack
        if meta.page.tags is None:
            meta.page.tags = []
        if meta.page.links is None:
            meta.page.links = []
        if 'alt' not in meta.page:
            meta.page.alt = None
        return cls(file, meta, has_frontmatter, is_binary)

    @classmethod
    def iter_files(cls, config: Config) -> Iterable['FileData']:
        return (cls.load(file, config.url) for file in config.iter_files())


def get_path(meta: Data) -> Path:
    """Generate a path based on file metadata"""
    path = str(meta.permalink or '/:path/:basename')
    def get(match: re.Match[str]) -> str:
        try:
            return str(meta[match.group(1)])
        except KeyError:
            return ''
    return Path(re.sub(r':(\w+)', get, path).lstrip('/'))


def get_frontmatter(source: str) -> str | None:
    """Split a file into the frontmatter and text file components"""
    text = text_file(source)
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 3)
    return text[4:end]


def get_content(source: str) -> str:
    """Split a file into the frontmatter and text file components"""
    text = text_file(source)
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 3)
    return text[end + 5:]


def split_frontmatter(source: str) -> tuple[str | None, str]:
    """Split a file into the frontmatter and text file components"""
    text = text_file(source)
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 3)
    frontmatter = text[4:end]
    text = text[end + 5:]
    return frontmatter, text


def text_file(contents: str) -> str:
    """Make sure this is a valid text file ending in newline"""
    if contents.endswith('\n'):
        return contents
    return contents + '\n'


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

YamlType: TypeAlias = (
    str | int | float | bool | datetime | list['YamlType'] | dict[str, 'YamlType']
)

def parse_yaml_dict(yaml_str: str) -> dict[str, YamlType | AutoDate]:
    yaml_dict = yaml.safe_load(yaml_str)
    if not isinstance(yaml_dict, Mapping):
        return {}
    return transform_dates(cast(dict[str, YamlType], yaml_dict))


def transform_dates(data: dict[str, YamlType]) -> dict[str, YamlType | AutoDate]:
    return {
        key: AutoDate(val.isoformat()) if isinstance(val, (datetime, date))
        else val for (key, val) in data.items()
    }
