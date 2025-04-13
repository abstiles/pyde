from __future__ import annotations

import contextlib
import dataclasses
from datetime import datetime
import re
import shutil
import time
from collections.abc import Collection
from functools import partial
from glob import glob
from itertools import islice
from os import PathLike
from pathlib import Path
from typing import Any, Callable, ChainMap, Iterable, TypeGuard, TypeVar, overload

from .config import Config
from .data import Data
from .markdown.handler import MarkdownParser
from .path import FilePath, LocalPath
from .plugins import PluginManager
from .site import NullPaginator, Paginator, SiteFileManager
from .templates import TemplateManager
from .transformer import CopyTransformer, MarkdownTransformer, Transformer
from .utils import Maybe, flatmap
from .watcher import SourceWatcher

T = TypeVar('T')
HEADER_RE = re.compile(b'^---\r?\n')


class Environment:
    def __init__(
        self,
        config: Config, /,
    ):
        self.exec_dir = LocalPath(config.config_root)
        self.config = config
        self.global_defaults: ChainMap[str, Any] = ChainMap({
            "permalink": config.permalink,
            "layout": "default",
            "site_url": self.config.url,
        })

        pyde = Data(
            environment='development' if config.drafts else 'production',
            env=self,
            **dataclasses.asdict(config)
        )
        self.template_manager = TemplateManager(
            self.includes_dir, self.layouts_dir,
            globals={'pyde': pyde, 'jekyll': pyde},
        )
        self._site_loaded = False
        self._site = SiteFileManager(
            tag_paginator=self._tag_paginator(),
            collection_paginator=self._collection_paginator(),
            page_defaults=self.global_defaults,
        )
        self._site.data.url = self.config.url
        self.template_manager.globals['site'] = self._site
        self.global_defaults["metaprocessor"] = self.template_manager.render

        PluginManager(self.plugins_dir).import_plugins(self)

    @property
    def markdown_parser(self) -> MarkdownParser:
        return MarkdownTransformer.markdown_parser

    @markdown_parser.setter
    def markdown_parser(self, new_parser: MarkdownParser) -> None:
        MarkdownTransformer.markdown_parser = new_parser

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
    def plugins_dir(self) -> LocalPath:
        return self.exec_dir / self.config.plugins_dir

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
        for path in map(LocalPath, self.source_files()):
            self.process_file(path)
        self._site_loaded = True
        return self._site

    def build(self) -> None:
        print(f'Building contents of {self.root}...')
        start = datetime.now()
        self.output_dir.mkdir(exist_ok=True)
        # Check to see what already exists in the output directory.
        existing_files = set(self.output_dir.rglob('*'))
        built_files = (file.render() for file in self.site)
        # Grab the output files and all the parent directories that might have
        # been created as part of the build.
        outputs = flatmap(file_and_parents(upto=self.output_dir), built_files)
        for file in outputs:
            existing_files.discard(file)
        print('Build complete. Cleaning up stale files.')
        for file in existing_files:
            print(f'Removing: {file}')
            if file.is_dir():
                shutil.rmtree(file, ignore_errors=True)
            else:
                file.unlink(missing_ok=True)
        end = datetime.now()
        print(f'Done in {(end - start).total_seconds():.2f}s')

    def watch(self) -> None:
        self.build()
        class SiteUpdater:
            @staticmethod
            def update(path: LocalPath) -> None:
                self.process_file(path)
            @staticmethod
            def delete(*_: LocalPath) -> None: ...

        watcher = SourceWatcher(
            self.root,
            excluded=self._excluded_paths(),
            included=self.config.include
        )
        watcher.register(SiteUpdater).start()
        while True:
            try:
                for file in self.site:
                    # It's fine if the file gets deleted before we can process it.
                    # Nothing else to do but ignore and move on.
                    with contextlib.suppress(FileNotFoundError):
                        file.render()
                time.sleep(1)
            except KeyboardInterrupt:
                print('\nStopping.')
                break

    def _excluded_paths(self) -> Collection[Path | str]:
        exclude_patterns = set(filter(_not_none, [
            self.config.output_dir,
            self.config.layouts_dir,
            self.config.includes_dir,
            self.config.plugins_dir,
            *Maybe(self.config.config_file),
            *self.config.exclude,
        ]))
        if not self.config.drafts:
            exclude_patterns.add(self.config.drafts_dir)
        return exclude_patterns

    def source_files(self) -> Iterable[LocalPath]:
        globber = partial(iterglob, root=self.root)
        excluded = set(flatmap(globber, self._excluded_paths()))
        excluded_dirs = set(filter(LocalPath.is_dir, excluded))
        included = set(flatmap(globber, self.config.include))
        files = set(flatmap(globber, set(['**'])))
        yield from {
            file.relative_to(self.root)
            for file in filter(LocalPath.is_file, (files - excluded) | included)
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

    def process_file(self, source: LocalPath) -> None:
        try:
            if not self.should_transform(source):
                self._site.append(
                    CopyTransformer(
                        source, file=source
                    ).preprocess(source, self.root, self.output_dir),
                    'raw',
                )
                return

            values = self.global_defaults.new_child(self.get_default_values(source))
            tf = Transformer(source, **values).preprocess(
                source, self.root, self.output_dir
            )
            layout = tf.metadata.get('layout', values['layout'])
            template_name = f'{layout}{tf.outputs.suffix}'
            template = self.template_manager.get_template(template_name)

            self._site.append(tf.pipe(template=template))
        except FileNotFoundError:
            # If a file is deleted before we can process it, that's fine. We
            # should just move on without making a fuss.
            pass

    def _collection_paginator(self) -> Paginator:
        pagination = self.config.paginate
        template = self.template_manager.get_template(f'{pagination.template}.html')
        return Paginator(
            template, pagination.permalink, self.root, self.output_dir,
            maximum=pagination.size,
        )

    def _tag_paginator(self) -> Paginator:
        if not self.config.tags.enabled:
            return NullPaginator()
        tag_spec = self.config.tags
        template = self.template_manager.get_template(f'{tag_spec.template}.html')
        return Paginator(
            template, tag_spec.permalink, self.root, self.output_dir,
            minimum=tag_spec.minimum if tag_spec.enabled else -1,
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
