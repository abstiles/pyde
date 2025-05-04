"""
Handle config file parsing
"""
import dataclasses
from dataclasses import InitVar, dataclass, field
from os import PathLike
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, ClassVar, Literal

import yaml

from .path import FilePath, UrlPath
from .utils import dict_to_dataclass, dictfilter, first

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

    def matches(self, path: FilePath | Path) -> bool:
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
    source_dir: Path | str = '_:collection'
    permalink: str = '/:collection/:basename'

    @property
    def source(self) -> Path:
        if isinstance(self.source_dir, Path):
            return self.source_dir
        return Path(self.source_dir.replace(':collection', self.name))


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
    name: str = 'Website Name'
    url: UrlPath = UrlPath('http://localhost/')
    root: Path = Path('.')
    drafts: bool = False
    permalink: str = '/:path/:basename'
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    layouts_dir: Path = Path('_layouts')
    includes_dir: Path = Path('_includes')
    plugins_dir: Path = Path('_plugins')
    drafts_dir: Path = Path('_drafts')
    output_dir: Path = Path('_site')
    posts: CollectionSpec = CollectionSpec('posts')
    tags: TagSpec = TagSpec()
    paginate: PaginationSpec = PaginationSpec()
    defaults: list[DefaultSpec] = field(default_factory=list)
    skip_post_config: InitVar[bool] = False
    _config_file: Path | None = field(init=False, default=None)
    _config_root: Path = field(init=False, default=Path('.'))

    @property
    def config_root(self) -> Path:
        return self.config_file.parent if self.config_file else Path('.')

    def __post_init__(self, skip_post_config: bool=False) -> None:
        default_drafts = DefaultSpec.make(
            self.drafts_dir, type='post', draft=True, collection='drafts',
        )

        try:
            draft_spec = first(
                spec for spec in self.defaults
                if spec.scope.path == self.drafts_dir
            )
            default_drafts.values.update(draft_spec.values)
            draft_spec.values.update(default_drafts.values)
        except ValueError:
            self.defaults.insert(0, default_drafts)

        if skip_post_config:
            return

        self.defaults.append(
            DefaultSpec.make(
                self.posts.source, type='post', collection=self.posts.name,
                collection_root=Path(self.posts.source),
                permalink=self.posts.permalink
            )
        )

    @classmethod
    def parse(cls, file: Path, raw: bool=False) -> 'Config':
        """Parse the given config file"""
        if not file.exists():
            return Config(skip_post_config=raw)
        with file.open() as f:
            config_data: dict[str, Any] = yaml.safe_load(f)
        return dict_to_dataclass(
            cls, {
                'config_file': file,
                **config_data,
                'skip_post_config': raw,
            },
        )

    def as_yaml(self) -> str:
        represent_path = lambda dumper, data: dumper.represent_str(str(data))
        yaml.add_representer(UrlPath, represent_path)
        # For some reason pyyaml doesn't seem to respect passing just the
        # "Path" parent class of both these two.
        yaml.add_representer(PosixPath, represent_path)
        yaml.add_representer(WindowsPath, represent_path)
        config_data = dictfilter(
            dataclasses.asdict(self),
            keys=lambda k: not k.startswith('_'),
        )
        return yaml.dump(config_data, sort_keys=False)
