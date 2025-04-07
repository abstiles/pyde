from __future__ import annotations

import dataclasses
import re
import shutil
from functools import partial
from glob import glob
from itertools import chain, islice
from os import PathLike
from pathlib import Path
from typing import Any, Callable, ChainMap, Iterable, TypeGuard, TypeVar, overload

from .config import Config
from .data import Data
from .path import FilePath, LocalPath
from .site import SiteFileManager
from .templates import TemplateManager
from .transformer import CopyTransformer, Transformer
from .utils import flatmap

T = TypeVar('T')
HEADER_RE = re.compile(b'^---\r?\n')


class Environment:
    def __init__(
        self,
        config: Config, /,
    ):
        pyde = Data(
            environment='development' if config.drafts else 'production',
            **dataclasses.asdict(config)
        )

        self.exec_dir = LocalPath(
            config.config_file.parent if config.config_file else '.'
        )
        self.config = config
        self._site = SiteFileManager(self.root, self.output_dir, self.config.url)
        self._site_loaded = False
        self.template_manager = TemplateManager(
            self.includes_dir,
            self.layouts_dir,
            globals={
                'site': self._site,
                'pyde': pyde,
                'jekyll': pyde,
            },
        )
        self.global_defaults: ChainMap[str, Any] = ChainMap({
            "permalink": config.permalink,
            "layout": "default",
            "metaprocessor": self.template_manager.render,
            "site_url": self.config.url,
        })

    @property
    def includes_dir(self) -> LocalPath:
        return self.exec_dir / self.config.includes_dir

    @property
    def layouts_dir(self) -> LocalPath:
        return self.exec_dir / self.config.layouts_dir

    @property
    def output_dir(self) -> LocalPath:
        return self.exec_dir / self.config.output_dir

    @property
    def drafts_dir(self) -> LocalPath:
        return self.exec_dir / self.config.drafts_dir

    @property
    def posts_dir(self) -> LocalPath:
        return self.exec_dir / self.config.posts.source

    @property
    def root(self) -> LocalPath:
        return self.exec_dir / self.config.root

    @property
    def site(self) -> SiteFileManager:
        if self._site_loaded:
            return self._site
        self._process_sources()
        self._process_collections()
        if self.config.tags.enabled:
            self._process_tags()
        self._site_loaded = True
        return self._site.load()

    def build(self) -> None:
        self.output_dir.mkdir(exist_ok=True)
        # Check to see what already exists in the output directory.
        existing_files = set(self.output_dir.rglob('*'))
        built_files = (
            file.render() for file in self.site
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

    def source_files(self) -> Iterable[LocalPath]:
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
        excluded_dirs = set(filter(LocalPath.is_dir, excluded))
        included = set(flatmap(globber, self.config.include))
        files = set(flatmap(globber, set(['**'])))
        yield from {
            file.relative_to(self.root)
            for file in chain(filter(LocalPath.is_file, files - excluded), included)
            if file in included or not excluded_dirs.intersection(file.parents)
        }

    def get_default_values(self, source: FilePath) -> dict[str, Any]:
        values = {}
        for default in self.config.defaults:
            if default.scope.matches(source):
                values.update(default.values)
        return values

    def should_transform(self, source: LocalPath) -> bool:
        """Return true if this file should be transformed in some way."""
        with (self.root / source).open('rb') as f:
            header = f.read(5)
            if HEADER_RE.match(header):
                return True
        return False

    def _process_sources(self) -> None:
        for source in map(LocalPath, self.source_files()):
            if not self.should_transform(source):
                self._site.append(
                    CopyTransformer(
                        source, file=source
                    ).preprocess(source, self.root, self.output_dir),
                    'raw',
                )
                continue

            values = self.global_defaults.new_child(self.get_default_values(source))
            tf = Transformer(source, **values).preprocess(
                source, self.root, self.output_dir
            )
            layout = tf.metadata.get('layout', values['layout'])
            template_name = f'{layout}{tf.outputs.suffix}'
            template = self.template_manager.get_template(template_name)

            self._site.append(tf.pipe(template=template))

    def _process_collections(self) -> None:
        pagination = self.config.paginate
        template = self.template_manager.get_template(f'{pagination.template}.html')
        self._site.generate_collection_pages(
            template, pagination.permalink, pagination.size, self.global_defaults,
        )

    def _process_tags(self) -> None:
        tag_spec = self.config.tags
        template = self.template_manager.get_template(f'{tag_spec.template}.html')
        self._site.generate_tag_pages(
            template, tag_spec.permalink, tag_spec.minimum, self.global_defaults,
        )

    def output_files(self) -> Iterable[FilePath]:
        for site_file in self.site:
            yield site_file.outputs

    def _tree(self, dir: LocalPath) -> Iterable[LocalPath]:
        return (
            f.relative_to(self.root.absolute())
            for f in dir.absolute().rglob('*')
            if not f.name.startswith('.')
        )

    def layout_files(self) -> Iterable[LocalPath]:
        return self._tree(self.layouts_dir)

    def include_files(self) -> Iterable[LocalPath]:
        return self._tree(self.includes_dir)

    def draft_files(self) -> Iterable[LocalPath]:
        return self._tree(self.drafts_dir)


def _not_none(item: T | None) -> TypeGuard[T]:
    return item is not None


def _is_dotfile(filename: str) -> TypeGuard[object]:
    return filename.startswith('.')


def _not_hidden(path_str: str) -> bool:
    return not any(map(_is_dotfile, Path(path_str).parts))


def iterglob(
    pattern: str | PathLike[str], root: LocalPath=LocalPath('.'),
) -> Iterable[LocalPath]:
    include_hidden = False
    if any(filter(_is_dotfile, Path(pattern).parts)):
        include_hidden = True
    all_matching = glob(
        str(pattern), root_dir=root, recursive=True,
        include_hidden=include_hidden,
    )
    for path in all_matching:
        yield root / str(path)


F = TypeVar('F', bound=FilePath)

@overload
def file_and_parents(*, upto: F) -> Callable[[FilePath], Iterable[F]]: ...
@overload
def file_and_parents(path: FilePath, /) -> Iterable[LocalPath]: ...
@overload
def file_and_parents(path: FilePath, /, *, upto: F) -> Iterable[F]: ...
def file_and_parents(
    path: FilePath | None=None, /, *, upto: FilePath=LocalPath('/')
) -> Iterable[FilePath] | Callable[[FilePath], Iterable[FilePath]]:
    def generator(file: FilePath, /) -> Iterable[FilePath]:
        assert upto is not None
        yield file
        parents = file.relative_to(str(upto)).parents
        # Use islice(reversed(...))) to skip the last parent, which will be
        # "upto" itself.
        yield from (
            upto / str(parent) for parent in islice(reversed(parents), 1, None)
        )
    if path is None:
        return generator
    return generator(path)
