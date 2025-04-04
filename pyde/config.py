"""
Handle config file parsing
"""
from dataclasses import InitVar, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml

from .path import FilePath, UrlPath
from .utils import dict_to_dataclass

PathType = str | PathLike[str]

ReadErrorType = Literal[
    'strict', 'ignore', 'replace', 'surrogateescape', 'backslashreplace'
]


@dataclass(frozen=True)
class ScopeSpec:
    ALL: ClassVar['ScopeSpec']
    path: Path = Path('')

    def __post_init__(self) -> None:
        object.__setattr__(self, 'path', Path(self.path))

    def matches(self, path: FilePath) -> bool:
        if self is ScopeSpec.ALL:
            return True
        return self.path in path.parents

ScopeSpec.ALL = ScopeSpec()


@dataclass(frozen=True)
class DefaultSpec:
    scope: ScopeSpec = ScopeSpec.ALL
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make(cls, scope: Path, /, **values: Any) -> 'DefaultSpec':
        return cls(ScopeSpec(scope), values)


@dataclass(frozen=True)
class TagSpec:
    template: str = ''
    permalink: str = '/tag/:tag'
    minimum: int | bool = 2

    @property
    def enabled(self) -> bool:
        return bool(self.template)



@dataclass(frozen=True)
class CollectionSpec:
    name: str
    source: Path = field(init=False)
    source_dir: InitVar[Path | str] = '_:collection'
    permalink: str = '/:collection/:path/:basename'

    def __post_init__(self, source_dir: Path | str) -> None:
        if isinstance(source_dir, Path):
            object.__setattr__(self, 'source', source_dir)
        else:
            object.__setattr__(
                self, 'source', Path(source_dir.replace(':collection', self.name))
            )


@dataclass(frozen=True)
class PaginationSpec:
    template: str = ''
    size: int = 20
    permalink: str = '/:collection/page.:num'

    def __bool__(self) -> bool:
        return bool(self.template) and self.size > 0


@dataclass
class Config:
    """Model of the config values in the config file"""
    config_file: Path | None = None
    url: UrlPath = UrlPath('/')
    root: Path = Path('.')
    drafts: bool = False
    output_dir: Path = Path('_site')
    permalink: str = '/:path/:basename'
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    defaults: list[DefaultSpec] = field(default_factory=list)
    layouts_dir: Path = Path('_layouts')
    includes_dir: Path = Path('_includes')
    drafts_dir: Path = Path('_drafts')
    posts: CollectionSpec = CollectionSpec('posts')
    tags: TagSpec = TagSpec()
    paginate: PaginationSpec = PaginationSpec()

    def __post_init__(self) -> None:
        self.defaults.extend([
            DefaultSpec.make(
                self.posts.source, type='post', collection=self.posts.name,
                collection_root=Path(self.posts.source),
                permalink=self.posts.permalink
            ),
            DefaultSpec.make(
                self.drafts_dir, type='post', draft=True, collection='drafts',
            ),
        ])

    @classmethod
    def parse(cls, file: Path) -> 'Config':
        """Parse the given config file"""
        with file.open() as f:
            config_data: dict[str, Any] = yaml.safe_load(f)
        return dict_to_dataclass(cls, {'config_file': file, **config_data})
