from __future__ import annotations

import dataclasses
import re
import shutil
import time
from collections.abc import Collection
from datetime import datetime
from functools import partial
from glob import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain, islice
from os import PathLike
from pathlib import Path
from threading import Thread
from typing import Any, Callable, ChainMap, Iterable, TypeGuard, TypeVar, overload

from pyde.livereload import LiveReloadServer

from .config import Config
from .data import Data
from .markdown.handler import MarkdownParser
from .path import FilePath, LocalPath
from .plugins import PluginManager
from .site import NullPaginator, Paginator, SiteFile, SiteFileManager
from .templates import TemplateManager
from .transformer import (
    CopyTransformer,
    MarkdownTransformer,
    TextTransformer,
    Transformer,
)
from .userlogging import AnimateLoader
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
            "site_name": self.config.name,
        })

        self.pyde_data = Data(
            environment='development' if config.drafts else 'production',
            env=self,
            **dataclasses.asdict(config)
        )
        self._site_loaded = False
        self.template_manager = self.create_template_manager()
        self._site = SiteFileManager(
            tag_paginator=self._tag_paginator(),
            collection_paginator=self._collection_paginator(),
            page_defaults=self.global_defaults,
        )
        self.template_manager.globals['site'] = self._site
        self._site.data.url = self.config.url
        self._site.data.name = self.config.name

        PluginManager(self.plugins_dir).import_plugins(self)

    def create_template_manager(self) -> TemplateManager:
        template_manager = TemplateManager(
            self.includes_dir, self.layouts_dir,
            globals={
                'pyde': self.pyde_data,
                'jekyll': self.pyde_data,
            },
        )
        self.global_defaults["metaprocessor"] = template_manager.render
        return template_manager

    def restart_template_manager(self) -> None:
        self.template_manager = self.create_template_manager()
        self.template_manager.globals['site'] = self._site
        self.global_defaults["metaprocessor"] = self.template_manager.render

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

    def spawn_http_server(self) -> HTTPServer:
        serve_dir = str(self.output_dir)
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, directory=serve_dir, **kwargs)
            def do_GET(self) -> None:
                try:
                    super().do_GET()
                except BrokenPipeError:
                    pass
            def translate_path(self, path: str) -> str:
                path = super().translate_path(path)
                if (
                    not (local_path := LocalPath(path)).suffix
                    and (html_path := local_path.with_suffix('.html')).exists()
                ):
                    return str(html_path)
                return path

        address, port = '', 8000
        httpd = ThreadingHTTPServer((address, port), Handler)
        thread = Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()
        print(f'Started HTTP Server on {address}:{port}')
        return httpd

    def build(self, serve: bool=False) -> None:
        if str(self.root) == '.':
            print('Building contents...')
        else:
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
        if serve:
            server = self.spawn_http_server()
            while True:
                try:
                    time.sleep(10)
                except KeyboardInterrupt:
                    server.server_close()
                    break

    def watch(self, serve: bool=False) -> None:
        reloader = LiveReloadServer().start()
        class ReloadTransformer(
            TextTransformer, pattern='*.html',
        ):  # pylint: disable=unused-variable
            """Append the reloader script to all HTML files"""
            def transform_text(self, text: str) -> str:
                path = self.metadata['file']
                return text + reloader.client_js()
        self.build()
        server = self.spawn_http_server() if serve else None
        loading_widget = AnimateLoader('Processing...', end='')
        class SiteUpdater:
            @staticmethod
            def update(path: LocalPath) -> None:
                loading_widget.start()
                try:
                    if {self.layouts_dir, self.includes_dir} & set(path.parents):
                        # Restart the template manager to avoid cached templates.
                        self.restart_template_manager()
                        for path in map(LocalPath, self.source_files()):
                            self.process_file(path)
                    else:
                        self.process_file(path)
                except Exception as exc:
                    print(f'Error processing {path} -', exc)
            @staticmethod
            def delete(*_: LocalPath) -> None: ...

        watcher = SourceWatcher(
            self.root,
            excluded=self._excluded_paths(),
            included=chain(
                self.config.include,
                [self.layouts_dir, self.includes_dir],
            )
        )
        watcher.register(SiteUpdater).start()
        try:
            while True:
                change_types = set()
                updates = self._update_changed()
                for file in updates:
                    if file.outputs.suffix == '.css':
                        change_types.add(LiveReloadServer.ReloadType.CSS)
                    else:
                        change_types.add(LiveReloadServer.ReloadType.PAGE)
                loading_widget.stop()
                if change_types:
                    reloader.reload(*change_types)
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print('\nStopping.')
            if server:
                server.server_close()
                reloader.stop()

    def _update_changed(self) -> Iterable[SiteFile]:
        updated: list[SiteFile] = []
        errors: list[str] = []
        for file in self.site:
            try:
                if not file.rendered:
                    updated.append(file)
                file.render()
            except FileNotFoundError:
                # It's fine if the file gets deleted before we can
                # process it. Nothing to do but ignore it.
                pass
            except Exception as exc:
                if file.type != 'meta':
                    errors.append(f'Error processing {file.source} - {exc}')
        yield from updated
        if errors:
            print()
        for error in errors:
            print(error)

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
        match = root / str(path)
        yield match
        if match.is_dir():
            children = match.glob('**/*')
            if not include_hidden:
                children = filter(_not_hidden, children)
            yield from map(LocalPath, children)


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
