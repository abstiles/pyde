from __future__ import annotations

import contextlib
from collections import ChainMap
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from itertools import chain, islice
from math import ceil
from operator import attrgetter
import threading
from types import MappingProxyType
from typing import Any, Callable, Generator, Literal, Protocol, Self, TypeAlias, cast

from jinja2 import Template

from .data import Data
from .path import ReadablePath, UrlPath, VirtualPath, WriteablePath
from .transformer import CopyTransformer, Transformer
from .utils import CaseInsensitiveStr, Maybe, batched, format_permalink, slugify

SiteFileType: TypeAlias = Literal['post', 'page', 'raw', 'meta', 'none']


class Tag(CaseInsensitiveStr):
    pass


class SiteFile:
    tf: Transformer
    type: SiteFileType

    def __init__(self, tf: Transformer, type: SiteFileType):
        self.tf = tf
        self.type = type
        self._rendered = False

    def __bool__(self) -> bool:
        return self.type != 'none'

    @classmethod
    def none(cls) -> Self:
        # Should use a proper null object for either this or the Transformer.
        return cls(cast(Transformer, None), 'none')

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
        if self._rendered:
            return self.tf.outputs
        self._rendered = True
        return self.tf.transform()

    @property
    def rendered(self) -> bool:
        return self._rendered

    @property
    def source(self) -> ReadablePath:
        return self.tf.source

    @property
    def outputs(self) -> WriteablePath:
        return self.tf.outputs

    @property
    def path(self) -> UrlPath:
        path = self.metadata.path
        if not isinstance(path, UrlPath):
            raise RuntimeError(f'Unknown path for {self.metadata}')
        return path

    @property
    def metadata(self) -> Data:
        if not self:
            return Data()
        return Data(self.tf.metadata)

    @property
    def tags(self) -> set[Tag]:
        return set(map(Tag, self.metadata.get('tags', ())))

    @property
    def collection(self) -> str | None:
        return self.metadata.get('collection')


class SiteFileManager(Iterable[SiteFile]):
    def __init__(self,
        *,
        tag_paginator: Paginator,
        collection_paginator: Paginator,
        page_defaults: dict[str, Any] | ChainMap[str, Any] | None = None,
    ) -> None:
        self._file_processor = FileProcessor(
            tag_paginator, collection_paginator,
            # Mypy seems to simplify the type here to just the common "Mapping"
            # shared between dict and ChainMap.
            Maybe(page_defaults).get({}), # type: ignore [arg-type]
        )
        self._site_data: Data = Data()

    @property
    def data(self) -> Data:
        return self._site_data

    def __getitem__(self, key: str) -> Any:
        return self._site_data[key]

    def append(
        self, transformer: Transformer, type: SiteFileType | None = None,
    ) -> Self:
        self._file_processor.update(transformer, type)
        return self

    def _page_data(self, type: SiteFileType) -> Iterable[Mapping[str, Any]]:
        return [
            f.metadata for f in self._file_processor.type_map.get(type, ())
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
            for tag, posts in self._file_processor.tag.items()
        }

    def __iter__(self) -> Iterator[SiteFile]:
        type_ordering: Iterable[SiteFileType] = ('post', 'page', 'raw', 'meta')
        for file_type in type_ordering:
            yield from self._file_processor.type_map.get(file_type, ())



class PostCollection:
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


class SupportsLT(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...

class SupportsGT(Protocol):
    def __gt__(self, other: Any, /) -> bool: ...

Sortable: TypeAlias = SupportsLT | SupportsGT


@dataclass(frozen=True, slots=True)
class Paginator:
    template: Template
    permalink: str
    source_dir: ReadablePath
    dest_dir: WriteablePath
    minimum: int = 1
    maximum: int = 0
    sort_by: InitVar[str | Callable[[Data], Sortable]] = attrgetter('date')
    key: Callable[[Data], Sortable] = field(init=False)

    def __post_init__(self, sort_by: str | Callable[..., Sortable]) -> None:
        key = attrgetter(sort_by) if isinstance(sort_by, str) else sort_by
        object.__setattr__(self, 'key', key)

    def paginate(
        self, name: str, sources: Collection[SiteFile], metadata: ChainMap[str, Any],
        index: str | None='index',
    ) -> Iterable[SiteFile]:
        if len(sources) < max(1, self.minimum):
            return []
        sources = sorted(sources, key=lambda s: self.key(s.metadata), reverse=True)
        if self.maximum > 0:
            total_pages = ceil(len(sources) / self.maximum)
            paginations = iter(batched(sources, self.maximum))
        else:
            total_pages = 1
            paginations = iter([sources])
        if index is not None:
            # Generate the index page as a first page in addition to the
            # usual "page 1" page, but don't generate two instances of the
            # first page if there are no other pages.
            if total_pages > 1:
                first = [*islice(paginations, 1)] * 2
                paginations = chain(first, paginations)
            permalink = '/'.join(self.permalink.split('/')[:-1]) + f'/{index}'
        else:
            permalink = self.permalink
        pages: list[SiteFile] = []
        for idx, page_posts in enumerate(paginations, start=0 if index else 1):
            title = f'{name.title()} Page {idx}' if idx else name.title()
            basename = (
                index if (index and idx == 0) else f'page{idx}'
            )
            collection_dir = format_permalink(
                self.permalink,
                num=str(idx),
                collection=name,
                tag=basename,
                basename=basename,
                name=basename,
            )
            source = VirtualPath(collection_dir + '.html').relative_to('/')
            values = metadata.new_child({
                'title': title,
                'permalink': permalink,
                'num': idx,
                'template': self.template,
                'collection': PostCollection(
                    name, page_posts, total_posts=len(sources),
                    total_pages=total_pages,
                ),
            })
            tf = Transformer(source, **values).preprocess(
                source, self.source_dir, self.dest_dir
            )
            page = SiteFile.meta(tf)
            permalink = self.permalink
            pages.append(page)

        if total_pages > 1:
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


TagMap: TypeAlias = Mapping[Tag, Sequence[SiteFile]]


class FileProcessor(Iterable[SiteFile]):
    _buffer: list[SiteFile]
    _cache: list[SiteFile]
    _files: dict[ReadablePath, SiteFile]
    _tag_map: dict[Tag, set[SiteFile]]
    _type_map: dict[SiteFileType, set[SiteFile]]
    _collections: dict[str, set[SiteFile]]

    def __init__(self,
        tag_paginator: Paginator,
        collection_paginator: Paginator,
        page_defaults: dict[str, Any] | ChainMap[str, Any],
    ) -> None:
        self._tag_paginator = tag_paginator
        self._collection_paginator = collection_paginator
        if not isinstance(page_defaults, ChainMap):
            page_defaults = ChainMap(page_defaults)
        self.defaults = page_defaults

        self._files = {}
        self._tag_map = {}
        self._type_map = {}
        self._collections = {}
        self._buffer_lock = threading.Lock()
        self._generator = self._process()
        next(self._generator)

    def _process(self) -> Generator[None, SiteFile, None]:
        self._buffer = []
        while True:
            site_file = yield
            with self._buffer_lock:
                self._buffer.append(site_file)

    def _update_model(self, site_file: SiteFile) -> None:
        if existing := self._files.get(site_file.source, SiteFile.none()):
            self._type_map.get(existing.type, set()).discard(existing)
            with contextlib.suppress(ValueError):
                self._buffer.remove(existing)
            if collection := existing.collection:
                self._collections.get(collection, set()).discard(existing)
        self._buffer.append(site_file)
        self._files[site_file.source] = site_file
        self._type_map.setdefault(site_file.type, set()).add(site_file)
        if site_file.type == 'post':
            if collection := site_file.collection:
                self._collections.setdefault(collection, set()).add(site_file)
            for tag in existing.tags:
                self._tag_map.setdefault(tag, set()).discard(existing)
            for tag in site_file.tags:
                self._tag_map.setdefault(tag, set()).add(site_file)

    def __iter__(self) -> Iterator[SiteFile]:
        yield from self.files

    @property
    def tag(self) -> Mapping[Tag, Collection[SiteFile]]:
        self._process_files()
        return MappingProxyType(self._tag_map)

    @property
    def type_map(self) -> Mapping[SiteFileType, Collection[SiteFile]]:
        self._process_files()
        return MappingProxyType(self._type_map)

    @property
    def files(self) -> Iterable[SiteFile]:
        yield from self._process_files()

    def _process_files(self) -> Collection[SiteFile]:
        """Processes any new files, returning the current collection"""
        def new_files() -> Iterable[SiteFile]:
            with self._buffer_lock:
                for site_file in self._buffer:
                    self._update_model(site_file)
                    yield site_file
                self._buffer.clear()
        if len([*new_files()]) > 0:
            self._cache = [*self.generate_files()]
        return self._cache

    def generate_files(self) -> Iterable[SiteFile]:
        files: Iterable[SiteFile] = self._files.values()
        for tag, posts in self._tag_map.items():
            if all(post.rendered for post in posts):
                # No need to regenerate the tags if there are no new,
                # unrendered posts.
                continue
            files = chain(files, self._generate_tag_pages(tag, posts))
        for collection, posts in self._collections.items():
            if all(post.rendered for post in posts):
                continue
            files = chain(files, self._generate_collection_pages(collection, posts))
        return files

    def _generate_tag_pages(
        self, tag: Tag, posts: Collection[SiteFile]
    ) -> Iterable[SiteFile]:
        for page in self._tag_paginator.paginate(
            tag, posts, self.defaults, index=slugify(tag),
        ):
            self._type_map.setdefault('meta', set()).add(page)
            yield page

    def _generate_collection_pages(
        self, collection: str, posts: Collection[SiteFile]
    ) -> Iterable[SiteFile]:
        for page in self._collection_paginator.paginate(
            collection, posts, self.defaults,
        ):
            self._type_map.setdefault('meta', set()).add(page)
            yield page

    def update(
        self, transformer: Transformer, type: SiteFileType | None = None,
    ) -> Self:
        self._generator.send(
            SiteFile(transformer, type) if type is not None
            else SiteFile.classify(transformer)
        )
        return self


class NullPaginator(Paginator):
    def __init__(self, *_: Any, **__: Any):  # pylint: disable=super-init-not-called
        pass

    def paginate(self, *_: Any, **__: Any) -> Iterable[SiteFile]:
        return []
