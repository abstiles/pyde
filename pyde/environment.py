from __future__ import annotations

from collections.abc import Mapping
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
    Iterator,
    Literal,
    Self,
    TypeAlias,
    TypeGuard,
    TypeVar,
    overload,
)

from pyde.url import UrlPath

from .config import Config
from .data import Data
from .templates import TemplateManager
from .transformer import CopyTransformer, Transformer
from .utils import flatmap, seq_pivot

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
        self.includes_dir = exec_dir / self.config.includes_dir
        self.layouts_dir = exec_dir / self.config.layouts_dir
        self.output_dir = exec_dir / self.config.output_dir
        self.drafts_dir = exec_dir / self.config.drafts_dir
        self.posts_dir = exec_dir / self.config.posts.source
        self.tags_dir = exec_dir / self.config.tags.path
        self.root = exec_dir / self.config.root
        self._site = SiteFileManager(self.config.url)
        self.template_manager = TemplateManager(
            self.config.url,
            exec_dir / self.includes_dir,
            exec_dir / self.layouts_dir,
            globals={
                'site': self._site,
                'pyde': pyde,
                'jekyll': pyde,
            },
        )

    @property
    def site(self) -> SiteFileManager:
        return self._site.load(self.transforms())

    def build(self) -> None:
        self.output_dir.mkdir(exist_ok=True)
        # Check to see what already exists in the output directory.
        existing_files = set(self.output_dir.rglob('*'))
        built_files = (
            file.render(self.root, self.output_dir) for file in self.site
        )
        # Grab the output files and all the parent directories that might have
        # been created as part of the build.
        outputs = flatmap(file_and_parents(upto=self.output_dir), built_files)
        for file in outputs:
            existing_files.discard(file)
        for file in existing_files:
            print(f'Removing: {file}')
            if file.is_dir():
                shutil.rmtree(file, ignore_errors=True)
            else:
                file.unlink(missing_ok=True)

    def source_files(self) -> Iterable[Path]:
        globber = partial(iterglob, root=self.root)
        exclude_patterns = set(filter(_not_none, [
            self.config.output_dir,
            self.config.layouts_dir,
            self.config.includes_dir,
            self.config.config_file,
            *self.config.exclude,
        ]))
        if not self.config.drafts:
            exclude_patterns.add(self.config.drafts_dir)
        excluded = set(flatmap(globber, exclude_patterns))
        excluded_dirs = set(filter(Path.is_dir, excluded))
        included = set(flatmap(globber, self.config.include))
        files = set(flatmap(globber, set(['**'])))
        yield from {
            file.relative_to(self.root)
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
            values = self.get_default_values(source)
            tf = Transformer(source, **(base | values)).preprocess(self.root)
            layout = tf.metadata.get('layout', values['layout'])
            template_name = f'{layout}{tf.outputs.suffix}'
            template = self.template_manager.get_template(template_name)
            yield tf.pipe(template=template, page=tf.metadata)

    def get_default_values(self, source: Path) -> dict[str, Any]:
        values = {}
        for default in self.config.defaults:
            if default.scope.matches(source):
                values.update(default.values)
        return values

    def should_transform(self, source: Path) -> bool:
        """Return true if this file should be transformed in some way."""
        with (self.root / source).open('rb') as f:
            header = f.read(5)
            if HEADER_RE.match(header):
                return True
        return False

    def output_files(self) -> Iterable[Path]:
        for transform in self.transforms():
            yield transform.outputs

    def _tree(self, dir: Path) -> Iterable[Path]:
        return (
            f.relative_to(self.root.absolute())
            for f in dir.absolute().rglob('*')
            if not f.name.startswith('.')
        )

    def layout_files(self) -> Iterable[Path]:
        return self._tree(self.layouts_dir)

    def include_files(self) -> Iterable[Path]:
        return self._tree(self.includes_dir)

    def draft_files(self) -> Iterable[Path]:
        return self._tree(self.drafts_dir)


SiteFileType: TypeAlias = Literal['post', 'page', 'raw', 'meta']

class SiteFile:
    def __init__(self, tf: Transformer, type: SiteFileType):
        self.tf, self.type = tf, type
        self.metadata = Data(tf.metadata)

    def render(self, input_dir: Path, output_dir: Path) -> Path:
        result = self.tf.transform_file(input_dir, output_dir)
        self.metadata = Data(self.tf.metadata)
        return result

    @property
    def source(self) -> Path:
        return self.tf.source

    @property
    def outputs(self) -> Path:
        return self.tf.outputs


class SiteFileManager(Iterable[SiteFile]):
    def __init__(self, url: UrlPath):
        self._files: Mapping[SiteFileType, Iterable[SiteFile]] = {}
        self._loaded = False
        self.url = url

    def _page_data(self, type: SiteFileType) -> Iterable[Mapping[str, Any]]:
        return [
            f.metadata for f in self._files.get(type, ())
        ]

    @property
    def pages(self) -> Iterable[Mapping[str, Any]]:
        return self._page_data('page')

    @property
    def posts(self) -> Iterable[Mapping[str, Any]]:
        return self._page_data('post')

    @property
    def raw(self) -> Iterable[Mapping[str, Any]]:
        return self._page_data('raw')

    @property
    def meta(self) -> Iterable[Mapping[str, Any]]:
        return self._page_data('meta')

    @classmethod
    def site_file(cls, tf: Transformer) -> SiteFile:
        if isinstance(tf, CopyTransformer):
            return SiteFile(tf, 'raw')
        if type := tf.metadata.get('type'):
            return SiteFile(tf, type)
        if tf.source.suffix in ('.md', '.html'):
            return SiteFile(tf, 'page')
        return SiteFile(tf, 'raw')

    def load(self, transformers: Iterable[Transformer]) -> Self:
        if self._loaded:
            return self
        site_files = map(self.site_file, transformers)
        self._files = seq_pivot(site_files, attr='type')
        self._loaded = True
        return self

    def __iter__(self) -> Iterator[SiteFile]:
        type_ordering: Iterable[SiteFileType] = (
            'post', 'page', 'meta', 'raw')
        for file_type in type_ordering:
            yield from self._files.get(file_type, ())


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
