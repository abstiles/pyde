from __future__ import annotations

import dataclasses
import re
import shutil
from functools import partial
from glob import glob
from itertools import islice
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    TypeAlias,
    TypeGuard,
    TypeVar,
    overload,
)

from .config import Config
from .data import Data
from .templates import TemplateManager
from .transformer import CopyTransformer, Transformer
from .utils import flatmap

T = TypeVar('T')
HEADER_RE = re.compile(b'^---\r?\n')


class Environment:
    def __init__(
        self,
        config: Config, /,
        exec_dir: Path=Path('.')
    ):
        pyde = Data(
            environment='development' if config.drafts else 'production',
            **dataclasses.asdict(config)
        )
        self.config = config
        self.exec_dir = exec_dir
        self.template_manager = TemplateManager(
            self.config.url,
            self.config.includes_dir,
            self.config.layouts_dir,
            globals={
                'site': Data(url=self.config.url),
                'pyde': pyde,
                'jekyll': pyde,
            },
        )
        self.site_filer = SiteFiler(self)

    def site_globals(self) -> dict[str, Any]:
        pyde = Data(
            drafts=self.config.drafts,
            environment="development" if self.config.drafts else "production"
        )
        return {
            'site': Data(url=self.config.url),
            'pyde': pyde,
            'jekyll': pyde,
        }

    def build(self, output_dir: Path) -> None:
        output_dir.mkdir(exist_ok=True)
        # Check to see what already exists in the output directory.
        existing_files = set(output_dir.rglob('*'))
        built_files = (
            tf.transform_file(self.config.root, output_dir)
            for tf in self.transforms()
        )
        # Grab the output files and all the parent directories that might have
        # been created as part of the build.
        outputs = flatmap(file_and_parents(upto=output_dir), built_files)
        for file in outputs:
            existing_files.discard(file)
        for file in existing_files:
            print(f'Removing: {file}')
            if file.is_dir():
                shutil.rmtree(file, ignore_errors=True)
            else:
                file.unlink(missing_ok=True)

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

    def render_template(self, source: str | bytes | Path, **metadata: Any) -> str:
        if isinstance(source, Path):
            return self.template_manager.get_template(source).render(metadata)
        template = source.decode('utf') if isinstance(source, bytes) else source
        return self.template_manager.render(template, metadata)

    def transforms(self) -> Iterable[Transformer]:
        base: dict[str, Any] = {
            "permalink": self.config.permalink, "layout": "default",
            "metaprocessor": self.render_template,
        }
        for source in self.source_files():
            if not self.should_transform(source):
                yield CopyTransformer(source)
                continue
            values = {}
            for default in self.config.defaults:
                if default.scope.matches(source):
                    values.update(default.values)
            tf = Transformer(source, **(base | values)).preprocess(self.config.root)
            layout = tf.metadata.get('layout', values['layout'])
            template_name = f'{layout}{tf.outputs.suffix}'
            template = self.template_manager.get_template(template_name)
            tf = tf.pipe(template=template, page=tf.metadata)
            yield tf

    def should_transform(self, source: Path) -> bool:
        """Return true if this file should be transformed in some way."""
        with (self.config.root / source).open('rb') as f:
            header = f.read(5)
            if HEADER_RE.match(header):
                return True
        return False

    def output_files(self) -> Iterable[Path]:
        for transform in self.transforms():
            yield transform.outputs

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


SiteFileType: TypeAlias = Literal['post', 'page', 'raw']

class SiteFile:
    def __init__(self, env: Environment, tf: Transformer, type: SiteFileType):
        self.env, self.tf, self.type = env, tf, type
        self.metadata = Data(tf.metadata)

    def render(self) -> None:
        self.tf.transform_file(
            self.env.config.root,
            self.env.config.output_dir
        )
        self.metadata = Data(self.tf.metadata)


class SiteFiler:
    def __init__(self, env: Environment):
        self.env = env

    def page(self, tf: Transformer) -> SiteFile:
        return SiteFile(self.env, tf, 'page')

    def post(self, tf: Transformer) -> SiteFile:
        return SiteFile(self.env, tf, 'post')

    def raw(self, tf: Transformer) -> SiteFile:
        return SiteFile(self.env, tf, 'raw')

    def __call__(self, tf: Transformer) -> SiteFile:
        if isinstance(tf, CopyTransformer):
            return self.raw(tf)
        if tf.source.suffix in ('.md', '.html'):
            return self.page(tf)
        return self.raw(tf)


def _not_none(item: T | None) -> TypeGuard[T]:
    return item is not None


def _not_dotfile(item: Path) -> bool:
    return not item.name.startswith('.')


def iterglob(pattern: str | PathLike[str], root: Path=Path('.')) -> Iterable[Path]:
    for path in glob(str(pattern), root_dir=root, recursive=True):
        yield root / path


@overload
def file_and_parents(*, upto: Path) -> Callable[[Path], Iterable[Path]]: ...
@overload
def file_and_parents(path: Path, /, *, upto: Path=Path('/')) -> Iterable[Path]: ...
def file_and_parents(
    path: Path | None=None, /, *, upto: Path=Path('/')
) -> Iterable[Path] | Callable[[Path], Iterable[Path]]:
    def generator(file: Path, /) -> Iterable[Path]:
        assert upto is not None
        yield file
        parents = file.relative_to(upto).parents
        # Use islice(reversed(...))) to skip the last parent, which will be
        # "upto" itself.
        yield from (
            upto / parent for parent in islice(reversed(parents), 1, None)
        )
    if path is None:
        return generator
    return generator(path)
