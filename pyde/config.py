"""
Handle config file parsing
"""

from dataclasses import dataclass, field, fields
from functools import partial
from glob import glob
from os import PathLike
from os.path import isdir
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

from .utils import flatmap
from .url import UrlPath

PathType = str | PathLike[str]

ReadErrorType = Literal[
    'strict', 'ignore', 'replace', 'surrogateescape', 'backslashreplace'
]

@dataclass
class SourceFile:
    """A source file with additional metadata attached"""
    path: Path
    root: Path
    values: dict[str, str]

    def read_text(
        self, encoding: str='utf8', errors: ReadErrorType | None=None,
    ) -> str:
        return self.path.read_text(encoding=encoding, errors=errors)

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()


@dataclass
class Config:
    """Model of the config values in the config file"""
    config_file: Path
    url: UrlPath = UrlPath('/')
    root: Path = Path('.')
    drafts: bool = False
    output_dir: Path = Path('_site')
    permalink: str = '/:path/:basename'
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    defaults: list[dict[str,dict[str,str]]] = field(default_factory=list)
    layouts_dir: Path = Path('_layouts')
    includes_dir: Path = Path('_includes')

    def iter_files(self) -> Iterable[SourceFile]:
        """Iterate through all files included in the build"""
        globber = partial(iterglob, root=self.root)
        exclude_patterns = set(map(str, [
            self.output_dir, self.layouts_dir, self.includes_dir,
            self.config_file, *self.exclude
        ]))
        if not self.drafts:
            exclude_patterns.add('_drafts')
        excluded = set(flatmap(globber, exclude_patterns))
        excluded_dirs = set(f for f in excluded if isdir(f))
        included = set(flatmap(globber, set(['**', *self.include])))
        paths = map(Path, included - excluded)
        files = filter(Path.is_file, paths)
        return (
            self.source_file(file) for file in files
            if not excluded_dirs.intersection(map(str, file.parents))
        )

    @classmethod
    def parse(cls, file: Path) -> 'Config':
        """Parse the given config file"""
        with file.open() as f:
            config_data: dict[str, Any] = yaml.safe_load(f)
        types = {
            field.name: field.type for field in fields(cls)
            if isinstance(field.type, type)
        }
        coerced_data = {
            key: (val if isinstance(val, types.get(key, object)) else types[key](val))
            for key, val in config_data.items()
        }
        return cls(config_file=file, **coerced_data)  # type: ignore

    def source_file(self, path: Path) -> SourceFile:
        """Attach metadata for file given scope defaults"""
        values: dict[str, str] = {"permalink": self.permalink, "layout": "default"}
        for default in self.defaults:
            scope_path = Path(default.get('scope', {}).get('path', '.'))
            if scope_path in path.relative_to(self.root).parents:
                values.update(default.get('values', {}))
        return SourceFile(path=path, root=self.root, values=values)


def iterglob(pattern: str, root: PathType='.') -> Iterable[str]:
    root = Path(root)
    for path in glob(pattern, root_dir=root, recursive=True):
        yield str(root / path)
