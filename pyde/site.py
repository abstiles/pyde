from __future__ import annotations

from collections import ChainMap
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from itertools import chain, islice
from math import ceil
from operator import attrgetter
from types import MappingProxyType
from typing import Any, Callable, Generator, Literal, Protocol, Self, TypeAlias

from jinja2 import Template

from .data import Data
from .path import ReadablePath, UrlPath, VirtualPath, WriteablePath
from .transformer import CopyTransformer, Transformer
from .utils import CaseInsensitiveStr, batched, seq_pivot, slugify

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
    def __init__(
        self,
        source_root: ReadablePath,
        dest_root: WriteablePath,
        url: UrlPath,
    ) -> None:
        self._file_processor = FileProcessor(source_root, dest_root)
        self._url = url
        self._files: Mapping[SiteFileType, Iterable[SiteFile]] = {}
        self._tags: Mapping[Tag, Sequence[SiteFile]] = {}
        self._loaded = False

    def append(
        self, transformer: Transformer, type: SiteFileType | None = None,
    ) -> Self:
        self._file_processor.append(transformer, type)
        return self

    def generate_tag_pages(
        self,
        template: Template,
        permalink: str,
        minimum: int,
        defaults: dict[str, Any] | ChainMap[str, Any]
    ) -> Self:
        self._file_processor.generate_tag_pages(
            template, permalink, minimum, defaults
        )
        return self

    def generate_collection_pages(
        self,
        template: Template,
        permalink: str,
        page_size: int,
        defaults: dict[str, Any] | ChainMap[str, Any]
    ) -> Self:
        self._file_processor.generate_collection_pages(
            template, permalink, page_size, defaults
        )
        return self

    def _page_data(self, type: SiteFileType) -> Iterable[Mapping[str, Any]]:
        return [
            f.metadata for f in self.files.get(type, ())
        ]

    @property
    def url(self) -> UrlPath:
        return self._url

    @property
    def files(self) -> Mapping[SiteFileType, Iterable[SiteFile]]:
        self.load()
        return self._files

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
        self.load()
        return {
            tag: [post.metadata for post in posts]
            for tag, posts in self._file_processor.tag.items()
        }

    def load(self) -> Self:
        if self._loaded:
            return self
        self._files = seq_pivot(self._file_processor, attr='type')
        self._loaded = True
        return self

    def __iter__(self) -> Iterator[SiteFile]:
        type_ordering: Iterable[SiteFileType] = ('post', 'page', 'raw', 'meta')
        for file_type in type_ordering:
            yield from self.files.get(file_type, ())


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
    page_size: int = 0
    sort_by: InitVar[str | Callable[[Data], Sortable]] = attrgetter('date')
    key: Callable[[Data], Sortable] = field(init=False)

    def __post_init__(self, sort_by: str | Callable[..., Sortable]) -> None:
        key = attrgetter(sort_by) if isinstance(sort_by, str) else sort_by
        object.__setattr__(self, 'key', key)

    def paginate(
        self, name: str, posts: list[SiteFile], metadata: ChainMap[str, Any],
        index: str | None='index',
    ) -> Iterable[SiteFile]:
        if len(posts) == 0:
            return []
        posts = sorted(posts, key=lambda s: self.key(s.metadata), reverse=True)
        source_dir = posts[0].source.parent
        if self.page_size > 0:
            total_pages = ceil(len(posts) / self.page_size)
            paginations = iter(batched(posts, self.page_size))
        else:
            total_pages = 1
            paginations = iter([posts])
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
        for idx, page_posts in enumerate(paginations):
            title = f'{name.title()} Page {idx}' if idx else name.title()
            source = VirtualPath(source_dir / 'page.html')
            values = metadata.new_child({
                'title': title,
                'permalink': permalink,
                'num': idx,
                'template': self.template,
                'collection': Collection(
                    name, page_posts, total_posts=len(posts),
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
    _files: list[SiteFile]
    _tag_map: dict[Tag, list[SiteFile]]
    _collections: dict[str, list[SiteFile]]

    def __init__(self, source_root: ReadablePath, dest_root: WriteablePath) -> None:
        self._source_root = source_root
        self._dest_root = dest_root
        self._generator = self._process()
        next(self._generator)

    def _process(self) -> Generator[None, SiteFile, None]:
        self._files = []
        self._tag_map = {}
        self._collections = {}
        while True:
            site_file = yield
            self._files.append(site_file)
            if site_file.type == 'post':
                self._collections.setdefault(site_file.metadata.collection, []).append(
                    site_file
                )
                for tag in (site_file.tf.metadata.get('tags') or []):
                    self._tag_map.setdefault(tag, []).append(site_file)

    def generate_tag_pages(
        self,
        template: Template,
        permalink: str,
        minimum: int,
        defaults: dict[str, Any] | ChainMap[str, Any]
    ) -> Self:
        if not isinstance(defaults, ChainMap):
            defaults = ChainMap(defaults)
        paginator = Paginator(
            template, permalink, self._source_root, self._dest_root,
        )
        for tag, posts in self._tag_map.items():
            if len(posts) < minimum:
                continue
            self._files.extend(paginator.paginate(
                tag, posts, defaults, index=slugify(tag),
            ))
        return self

    def generate_collection_pages(
        self,
        template: Template,
        permalink: str,
        page_size: int,
        defaults: dict[str, Any] | ChainMap[str, Any]
    ) -> Self:
        if not isinstance(defaults, ChainMap):
            defaults = ChainMap(defaults)
        paginator = Paginator(
            template, permalink, self._source_root, self._dest_root,
            page_size=page_size,
        )
        for collection, posts in self._collections.items():
            self._files.extend(paginator.paginate(collection, posts, defaults))
        return self

    def __iter__(self) -> Iterator[SiteFile]:
        return iter(self._files)

    @property
    def tag(self) -> Mapping[Tag, Sequence[SiteFile]]:
        return MappingProxyType(self._tag_map)

    def append(
        self, transformer: Transformer, type: SiteFileType | None = None,
    ) -> Self:
        self._generator.send(
            SiteFile(transformer, type) if type is not None
            else SiteFile.classify(transformer)
        )
        return self
