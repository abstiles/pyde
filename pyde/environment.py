from __future__ import annotations

import dataclasses
import re
import shutil
from collections.abc import Generator, Mapping, Sequence
from functools import partial
from glob import glob
from itertools import chain, islice
from math import ceil
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    ChainMap,
    Iterable,
    Iterator,
    Literal,
    Self,
    TypeAlias,
    TypeGuard,
    TypeVar,
    overload,
)

from .config import Config
from .data import Data
from .path import UrlPath, FilePath, LocalPath, ReadablePath, WriteablePath
from .path.virtual import VirtualPath
from .templates import TemplateManager
from .transformer import CopyTransformer, Transformer
from .utils import (
    CaseInsensitiveStr,
    ReturningGenerator,
    batched,
    flatmap,
    seq_pivot,
    slugify,
)

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

        exec_dir = LocalPath(config.config_file.parent if config.config_file else '.')
        self.config = config
        self.exec_dir = exec_dir
        self.includes_dir = exec_dir / self.config.includes_dir
        self.layouts_dir = exec_dir / self.config.layouts_dir
        self.output_dir = exec_dir / self.config.output_dir
        self.drafts_dir = exec_dir / self.config.drafts_dir
        self.posts_dir = exec_dir / self.config.posts.source
        self.root = exec_dir / self.config.root
        self.global_defaults: ChainMap[str, Any] = ChainMap({
            "permalink": self.config.permalink,
            "layout": "default",
            "metaprocessor": self.render_template,
            "site_url": self.config.url,
            "links": [],
        })
        self._site = SiteFileManager(self.config.url)
        self.template_manager = TemplateManager(
            exec_dir / self.includes_dir,
            exec_dir / self.layouts_dir,
            globals={
                'site': self._site,
                'pyde': pyde,
                'jekyll': pyde,
            },
        )

    @property
    def globals(self) -> Mapping[str, Any]:
        return self.template_manager.globals

    @property
    def site(self) -> SiteFileManager:
        transform_processor = FileProcessor(self.transforms())
        return self._site.load(transform_processor)

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
            for file in filter(LocalPath.is_file, files - excluded)
            if file in included or not excluded_dirs.intersection(file.parents)
        }

    def render_template(self, source: str | bytes | Path, **metadata: Any) -> str:
        if isinstance(source, Path):
            return self.template_manager.get_template(source).render(metadata)
        template = source.decode('utf') if isinstance(source, bytes) else source
        return self.template_manager.render(template, metadata)

    def transforms(self) -> Generator[SiteFile, None, dict[Tag, list[SiteFile]]]:
        tags: dict[Tag, list[SiteFile]] = {}
        collections: dict[str, list[SiteFile]] = {}
        source: ReadablePath
        for source in map(LocalPath, self.source_files()):
            if not self.should_transform(source):
                yield SiteFile(
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

            site_file = SiteFile.classify(tf.pipe(template=template))
            if site_file.type == 'post':
                collections.setdefault(site_file.metadata.collection, []).append(
                    site_file
                )
                for tag in (tf.metadata.get('tags') or []):
                    tags.setdefault(tag, []).append(site_file)
            yield site_file

        yield from self.collection_meta_pages(collections)

        if self.config.tags.enabled:
            yield from self.tag_meta_pages(tags)

        return tags

    def make_virtual_page(
        self, path: FilePath, layout: str, metadata: Mapping[str, Any],
    ) -> SiteFile:
        source = VirtualPath(path)
        template = self.template_manager.get_template(layout)
        values = dict(metadata) | {'template': template}
        tf = Transformer(source, **values).preprocess(
            source, self.root, self.output_dir
        )
        return SiteFile.meta(tf)

    def tag_meta_pages(self, tags: Mapping[Tag, list[SiteFile]]) -> Iterable[SiteFile]:
        for tag, pages in tags.items():
            if len(pages) < self.config.tags.minimum:
                continue
            pages.sort(key=lambda p: p.metadata.date, reverse=True)
            yield from self.iter_pages(
                tag, pages, landing=slugify(tag),
                page_permalink=self.config.tags.permalink,
                tag=tag, layout=self.config.tags.template,
                force_single_page=True,
            )

    def collection_meta_pages(
        self, collections: Mapping[str, list[SiteFile]],
    ) -> Iterable[SiteFile]:
        for collection, posts in collections.items():
            if len(posts) / self.config.paginate.size <= 1:
                continue
            posts.sort(key=lambda p: p.metadata.date, reverse=True)
            yield from self.iter_pages(collection, posts)

    def iter_pages(
        self,
        collection_name: str,
        posts: list[SiteFile],
        *,
        page_permalink: str = '',
        layout: str = '',
        landing: str = 'index',
        force_single_page: bool = False,
        **overrides: Any,
    ) -> Iterable[SiteFile]:
        page_permalink = page_permalink or self.config.paginate.permalink
        if self.config.paginate and not force_single_page:
            paginate_size = self.config.paginate.size
            total_pages = ceil(len(posts) / self.config.paginate.size)
            paginations = iter(batched(posts, paginate_size))
            # Generate the "landing" page as a first page in addition to the
            # usual "page 1" page, but don't generate two instances of the
            # first page if there are no other pages.
            if total_pages > 1:
                first = [*islice(paginations, 1)] * 2
                paginations = chain(first, paginations)
        else:
            paginate_size = 0
            total_pages = 1
            paginations = iter([posts])
        permalink = '/'.join(page_permalink.split('/')[:-1]) + f'/{landing}'
        pages: list[SiteFile] = []
        for idx, page_posts in enumerate(paginations):
            title = (
                f'{collection_name.title()} Page {idx}' if idx
                else collection_name.title()
            )
            source = VirtualPath(self.config.posts.source / 'page.html')
            values = self.global_defaults.new_child(
                self.get_default_values(source)
            ).new_child({
                'title': title,
                'permalink': permalink,
                'num': idx,
                'collection': Collection(
                    collection_name, page_posts, total_posts=len(posts),
                    total_pages=total_pages,
                ),
            }).new_child(overrides)
            template_name = f'{layout or self.config.paginate.template}.html'
            page = self.make_virtual_page(source, template_name, values)
            permalink = page_permalink
            pages.append(page)

        if self.config.paginate:
            for idx, collection in enumerate(
                page.metadata.collection for page in pages
            ):
                if idx not in (0, 1):
                    collection.previous = pages[idx - 1].metadata
                if idx != len(pages) - 1:
                    collection.next = pages[max(2, idx + 1)].metadata
                if len(pages) > 1:
                    collection.start = pages[1].metadata
                    collection.end = pages[-1].metadata
        return pages

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

    def output_files(self) -> Iterable[FilePath]:
        for transform in self.transforms():
            yield transform.outputs

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


SiteFileType: TypeAlias = Literal['post', 'page', 'raw', 'meta']


class Tag(CaseInsensitiveStr):
    pass


class SiteFile:
    def __init__(self, tf: Transformer, type: SiteFileType):
        self.tf, self.type = tf, type

    @classmethod
    def raw(cls, tf: Transformer) -> Self:
        return cls(tf, 'raw')

    @classmethod
    def page(cls, tf: Transformer) -> Self:
        return cls(tf, 'page')

    @classmethod
    def post(cls, tf: Transformer) -> Self:
        return cls(tf, 'post')

    @classmethod
    def meta(cls, tf: Transformer) -> Self:
        return cls(tf, 'meta')

    @classmethod
    def classify(cls, tf: Transformer) -> Self:
        if isinstance(tf, CopyTransformer):
            return cls.raw(tf)
        if type := tf.metadata.get('type'):
            return cls(tf, type)
        if tf.source.suffix in ('.md', '.html'):
            return cls.page(tf)
        return cls.raw(tf)

    def render(self) -> WriteablePath:
        return self.tf.transform()

    @property
    def source(self) -> ReadablePath:
        return self.tf.source

    @property
    def outputs(self) -> WriteablePath:
        return self.tf.outputs

    @property
    def metadata(self) -> Data:
        return Data(self.tf.metadata)

    @property
    def tags(self) -> Iterable[Tag]:
        return map(Tag, self.metadata.get('tags', ()))


class SiteFileManager(Iterable[SiteFile]):
    def __init__(self, url: UrlPath):
        self._files: Mapping[SiteFileType, Iterable[SiteFile]] = {}
        self._loaded = False
        self.url = url
        self._tags: TagMap = {}

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

    @property
    def tags(self) -> Mapping[str, list[Data]]:
        return {
            tag: [post.metadata for post in posts]
            for tag, posts in self._tags.items()
        }

    def load(self, site_files: FileProcessor) -> Self:
        if self._loaded:
            return self
        self._files = seq_pivot(site_files, attr='type')
        self._tags = site_files.tag_map
        self._loaded = True
        return self

    def __iter__(self) -> Iterator[SiteFile]:
        type_ordering: Iterable[SiteFileType] = ('post', 'page', 'raw', 'meta')
        for file_type in type_ordering:
            yield from self._files.get(file_type, ())


TagMap: TypeAlias = Mapping[Tag, Sequence[SiteFile]]

class FileProcessor(Iterable[SiteFile]):
    def __init__(self, tf_generator: Generator[SiteFile, None, TagMap]):
        self._generator = ReturningGenerator(tf_generator)
        self._files: list[SiteFile] | None = None

    def _process(self) -> list[SiteFile]:
        if self._files is None:
            self._files = list(self._generator)
        return self._files

    def __iter__(self) -> Iterator[SiteFile]:
        return iter(self.site_files)

    @property
    def site_files(self) -> Iterable[SiteFile]:
        return self._process()

    @property
    def tag_map(self) -> TagMap:
        self._process()
        return self._generator.value


class Collection:
    def __init__(self, name: str, posts: Sequence[SiteFile] = (), **kwargs: Any):
        self._name = name
        self._posts = [post.metadata for post in posts]
        self._metadata = kwargs

    @property
    def name(self) -> str:
        return self._name

    @property
    def posts(self) -> Sequence[Data]:
        return self._posts

    @property
    def size(self) -> int:
        return len(self._posts)

    def __getattr__(self, attr: str) -> Any:
        return self[attr]

    def __getitem__(self, key: str) -> Any:
        return self._metadata.get(key, Data(_from=f'{self!r}.{key}'))

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.name!r}, posts[{len(self.posts)}])'

    def __iter__(self) -> Iterator[Data]:
        return iter(self.posts)


def _not_none(item: T | None) -> TypeGuard[T]:
    return item is not None


def _is_dotfile(filename: str) -> bool:
    return filename.startswith('.')


def _not_hidden(path_str: str) -> bool:
    return not any(map(_is_dotfile, Path(path_str).parts))


def iterglob(
    pattern: str | PathLike[str], root: LocalPath=LocalPath('.'),
) -> Iterable[LocalPath]:
    all_matching = glob(str(pattern), root_dir=root, recursive=True)
    for path in filter(_not_hidden, all_matching):
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
