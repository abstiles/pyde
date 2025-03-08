"""
Handle config file parsing
"""

from   dataclasses              import dataclass, field
from   functools                import partial
from   glob                     import glob
from   os                       import PathLike
from   pathlib                  import Path
from   typing                   import Iterable

import yaml

from   .utils                   import flatmap

PathType = str | PathLike[str]


@dataclass
class SourceFile:
    """A source file with additional metadata attached"""
    path: Path
    values: dict[str, str]


@dataclass
class Config:
    """Model of the config values in the config file"""
    url: str = ''
    permalink: str = '/:path/:basename'
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    defaults: list[dict[str,dict[str,str]]] = field(default_factory=list)
    layouts_dir: Path = Path('_layouts')

    def iter_files(
        self, root: PathType='.', exclude: Iterable[PathType]=()
    ) -> Iterable[SourceFile]:
        """Iterate through all files included in the build"""
        globber = partial(glob, root_dir=root, recursive=True)
        excluded = set(flatmap(globber, set([*map(str, exclude), *self.exclude])))
        excluded_dirs = set(f for f in excluded if Path(f).is_dir())
        included = set(flatmap(globber, set(['**', *self.include])))
        filenames = map(str, included - excluded)
        paths = map(Path, filenames)
        files = filter(Path.is_file, paths)
        return (
            self.source_file(file) for file in files
            if not excluded_dirs.intersection(map(str, file.parents))
        )

    @classmethod
    def parse(cls, file: Path) -> 'Config':
        """Parse the given config file"""
        with file.open() as f:
            return cls(**yaml.safe_load(f))

    def source_file(self, path: Path) -> SourceFile:
        """Attach metadata for file given scope defaults"""
        values: dict[str, str] = {"permalink": self.permalink}
        for default in self.defaults:
            scope_path = Path(default.get('scope', {}).get('path', '.'))
            if scope_path in path.parents:
                values.update(default.get('values', {}))
        return SourceFile(path=path, values=values)
