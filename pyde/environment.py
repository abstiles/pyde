from __future__ import annotations

from functools import partial
from glob import glob
from os import PathLike
from pathlib import Path
from typing import Iterable, TypeGuard, TypeVar

from .config import Config
from .utils import flatmap


T = TypeVar('T')


class Environment:
    def __init__(
        self,
        config: Config, /,
        exec_dir: Path=Path('.')
    ):
        self.config = config
        self.exec_dir = exec_dir

    def source_files(self) -> Iterable[Path]:
        globber = partial(iterglob, root=self.config.root)
        exclude_patterns = set(filter(_not_none, [
            self.config.output_dir,
            self.config.layouts_dir,
            self.config.includes_dir,
            self.config.config_file,
            *self.config.exclude,
        ]))
        if not self.config.drafts:
            exclude_patterns.add('_drafts')
        excluded = set(flatmap(globber, exclude_patterns))
        excluded_dirs = set(filter(Path.is_dir, excluded))
        included = set(flatmap(globber, self.config.include))
        files = set(flatmap(globber, set(['**'])))
        yield from {
            file.relative_to(self.config.root)
            for file in filter(Path.is_file, files - excluded)
            if file in included or not excluded_dirs.intersection(file.parents)
        }

    def output_files(self) -> Iterable[Path]:
        pass

    def _tree(self, dir: Path) -> Iterable[Path]:
        return (
            f.relative_to(self.exec_dir.absolute())
            for f in dir.absolute().rglob('*')
            if not f.name.startswith('.')
        )

    def layout_files(self) -> Iterable[Path]:
        return self._tree(self.config.layouts_dir)

    def include_files(self) -> Iterable[Path]:
        return self._tree(self.config.includes_dir)

    def draft_files(self) -> Iterable[Path]:
        return self._tree(self.config.drafts_dir)


def _not_none(item: T | None) -> TypeGuard[T]:
    return item is not None


def _not_dotfile(item: Path) -> bool:
    return not item.name.startswith('.')


def iterglob(pattern: str | PathLike[str], root: Path=Path('.')) -> Iterable[Path]:
    for path in glob(str(pattern), root_dir=root, recursive=True):
        yield root / path
