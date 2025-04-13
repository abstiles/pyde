from __future__ import annotations

import os
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    ClassVar,
    Literal,
    Protocol,
    Self,
    Type,
    cast,
    overload,
)

from jinja2 import Template
from markupsafe import Markup
from typing_extensions import Concatenate

from .data import AutoDate, Data
from .markdown import MarkdownParser
from .path import (
    AnyDest,
    AnySource,
    FilePath,
    LocalPath,
    ReadablePath,
    UrlPath,
    VirtualPath,
    WriteablePath,
    dest,
    source,
)
from .utils import Maybe, ilen, merge_dicts, format_permalink
from .yaml import parse_yaml_dict

DEFAULT_PERMALINK = '/:path/:name'
UNSET: Any = object()

TransformerMatcher = Callable[Concatenate[FilePath, ...], bool]


class TransformerType(Protocol):
    def __init__(self, src_path: ReadablePath, /, **kwargs: Any): ...

    __matcher: TransformerMatcher = lambda *a, **kw: False

    @property
    def outputs(self) -> WriteablePath: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    @metadata.setter
    def metadata(self, meta: dict[str, Any]) -> None: ...

    def preprocess(
        self, path: ReadablePath, src_root: ReadablePath, dest_root: WriteablePath,
    ) -> Self: ...

    def transform_data(self, data: AnyStr) -> str | bytes: ...

    def transform_file(
        self, source: ReadablePath, dest: WriteablePath
    ) -> WriteablePath:
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        transformed = self.transform_data(data)  # pylint: disable=assignment-from-no-return
        if isinstance(transformed, bytes):
            dest.write_bytes(transformed)
        else:
            dest.write_text(transformed)
        return dest


@dataclass(frozen=True)
class TransformerRegistration:
    transformer: Type[TransformerType]
    wants: TransformerMatcher


class Transformer(TransformerType):
    __slots__ = ('_source', '_meta')
    _source: ReadablePath
    _meta: dict[str, Any]
    registered: ClassVar[list[TransformerRegistration]] = []  # pylint: disable=declare-non-slot

    def __new__(
        cls,
        src_path: AnySource,
        /, *,
        parse_frontmatter: bool=True,
        permalink: str=DEFAULT_PERMALINK,
        template: Template | None=None,
        metaprocessor: MetaProcessor | None=None,
        **meta: Any,
    ) -> Transformer:
        src_path = source(src_path)
        kwargs = {
            'parse_frontmatter': parse_frontmatter,
            'permalink': permalink,
            'template': template,
            'metaprocessor': metaprocessor,
            **meta
        }
        if cls is Transformer:
            transformers: list[type[TransformerType]] = []
            for registration in cls.registered:
                if registration.wants(src_path, **kwargs):
                    transformers.append(registration.transformer)
            transformers.append(CopyTransformer)
            if len(transformers) > 1:
                return PipelineTransformer.build(
                    src_path, transformers=transformers, **kwargs
                )
            # If there's only one, it's the obligatory CopyTransformer.
            cls = cast(type[CopyTransformer], transformers[0])
        return super().__new__(cls)  # pyright: ignore

    def __init_subclass__(cls, /, pattern: str | None=None, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        Transformer.register(cls, pattern)

    def __init__(self, src_path: AnySource, /, **kwargs: Any):
        self._source = source(src_path)
        self._meta = kwargs

    def transform(self) -> WriteablePath:
        raise NotImplementedError('Base class does not implement transform')

    @property
    def source(self) -> ReadablePath:
        return self._source

    def pipe(self, **meta: Any) -> Transformer:
        next = Transformer(self.outputs, **meta)
        return PipelineTransformer(
            self.source, pipeline=[self, next],
        )

    @staticmethod
    def _pattern_matcher(pattern: str) -> TransformerMatcher:
        def matcher(src_path: FilePath, /, **_kwargs: Any) -> bool:
            return src_path.match(pattern)
        return matcher

    @classmethod
    def register(
        cls, transformer: Type[Transformer],
        /, matcher: TransformerMatcher | str | None=None,
        *, index: int | None=None,
    ) -> Type[Self]:
        if callable(matcher):
            matcher_func = matcher
        elif isinstance(matcher, str):
            matcher_func = cls._pattern_matcher(matcher)
        elif matcher_classmethod := getattr(
            transformer, f'_{transformer.__name__}__matcher', None,
        ):
            matcher_func = matcher_classmethod
        else:
            return cls
        index = len(cls.registered) if index is None else index
        cls.registered.insert(index, TransformerRegistration(transformer, matcher_func))
        return cls

    # Phony implementations of the TransformerType protocol just to convince
    # type checkers that it's okay to try calling `Transformer(...)` even
    # though it's abstract.
    if TYPE_CHECKING:
        @property
        def outputs(self) -> WriteablePath: ...
        @property
        def metadata(self) -> dict[str, Any]:
            return {}
        @metadata.setter
        def metadata(self, meta: dict[str, Any]) -> None:
            _ = meta
        def preprocess(
            self, path: AnySource, src_root: AnySource='.', dest_root: AnyDest='.'
        ) -> Self:
            _ = source, src_root, dest_root
            return self
        def transform_file(
            self, source: ReadablePath, dest: WriteablePath
        ) -> WriteablePath:
            return cast(WriteablePath, (source, dest))
        def transform_data(self, data: AnyStr) -> str | bytes:
            return data


class BaseTransformer(Transformer):
    __slots__ = ('_src_root', '_dest_root', '_source', '_meta')
    _src_root: ReadablePath | None
    _dest_root: WriteablePath | None

    def __init__(
        self,
        src_path: AnySource,
        /, *,
        parse_frontmatter: bool=True,
        permalink: str=DEFAULT_PERMALINK,
        template: Template | None=None,
        metaprocessor: MetaProcessor | None=None,
        **meta: Any,
    ):
        super().__init__(src_path, **meta)
        # The init method needs to handle this for the ease of implementing
        # subclasses that only care about some of this information, but here
        # we explicitly ignore the Template constructor arguments that spawn
        # each instance within a pipeline.
        _ = parse_frontmatter, permalink, template, metaprocessor
        self._source = source(src_path)
        self._src_root = None
        self._dest_root = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.source)!r})'

    def transformed_name(self) -> str:
        return self.source.name

    def get_output_path(self) -> WriteablePath:
        return self.source.parent / self.transformed_name()

    @property
    def outputs(self) -> WriteablePath:
        return self.get_output_path()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._meta

    @metadata.setter
    def metadata(self, meta: dict[str, Any]) -> None:
        self._set_meta(meta)

    def _set_meta(self, meta: dict[str, Any]) -> Self:
        meta.update(merge_dicts(self._meta, meta))
        self._meta = meta
        return self

    def pipe(self, **meta: Any) -> Transformer:
        pipeline = super().pipe(**meta)
        if self.has_preprocessed():
            pipeline.preprocess(self.source, self.src_root, self.dest_root)
        return pipeline

    @property
    def src_root(self) -> ReadablePath:
        if self._src_root is not None:
            return self._src_root
        raise RuntimeError(
            "Transformer hasn't been preprocessed. Unknown source/dest directories."
        )

    @property
    def dest_root(self) -> WriteablePath:
        if self._dest_root is not None:
            return self._dest_root
        raise RuntimeError(
            "Transformer hasn't been preprocessed. Unknown source/dest directories."
        )

    def preprocess(
        self, path: AnySource, src_root: AnySource='.', dest_root: AnyDest='.'
    ) -> Self:
        self._source = source(path)
        self._src_root = source(src_root)
        self._dest_root = dest(dest_root)
        return self

    def transform(self) -> WriteablePath:
        input = self.src_root / self.source
        output = self.dest_root / self.outputs
        return self.transform_file(input, output)

    def has_preprocessed(self) -> bool:
        return (self._src_root, self._dest_root) != (None, None)


class PipelineTransformer(BaseTransformer):
    _pipeline: Sequence[TransformerType]

    def get_output_path(self) -> WriteablePath:
        return self._pipeline[-1].outputs

    def _set_meta(self, meta: dict[str, Any]) -> Self:
        super()._set_meta(meta)
        for pipe in self._pipeline:
            pipe.metadata = self.metadata
        return self

    def preprocess(
        self, path: AnySource, src_root: AnySource='.', dest_root: AnyDest='.'
    ) -> Self:
        if self.has_preprocessed():
            return self
        super().preprocess(path, src_root, dest_root)
        metadata: dict[str, Any] = self.metadata
        input = self.source
        input_dir = self.src_root
        output_dir = VirtualPath(self.dest_root)
        for pipe in self._pipeline:
            pipe.metadata = metadata
            pipe.preprocess(input, input_dir, output_dir)
            input = pipe.outputs
            input_dir = VirtualPath(src_root)
        return self

    def transform(self) -> WriteablePath:
        input = self.src_root / self.source
        dest = self.dest_root / self.outputs
        for pipe in self._pipeline[:-1]:
            input = pipe.transform_file(input, VirtualPath(dest))
        pipe = self._pipeline[-1]
        return pipe.transform_file(input, dest)

    def transform_data(self, data: AnyStr) -> str | bytes:
        current_data: str | bytes = data
        metadata: dict[str, Any] = self.metadata
        for pipe in self._pipeline:
            pipe.metadata = metadata
            current_data = pipe.transform_data(cast(AnyStr, current_data))
        return current_data

    def __repr__(self) -> str:
        args = ', '.join(map(repr, self._pipeline))
        return f'{self.__class__.__name__}({args})'

    def __init__(
        self, src_path: AnySource, /, *,
        pipeline: Sequence[TransformerType]=UNSET,
        permalink: str=DEFAULT_PERMALINK,
        **meta: Any
    ):
        super().__init__(src_path, **meta)
        if pipeline is not UNSET:
            self._pipeline = pipeline
        self._permalink = permalink

    def __getitem__(self, idx: int | slice) -> TransformerType | PipelineTransformer:
        if isinstance(idx, slice):
            pipe_tf = PipelineTransformer(
                self.source,
                pipeline=self._pipeline[idx.start:idx.stop:idx.step]
            )
            pipe_tf._meta = self.metadata
            return pipe_tf
        return self._pipeline[idx]

    def _partitioned(
        self
    ) -> tuple[
        Maybe[MetaTransformer], Sequence[TransformerType], Maybe[CopyTransformer]
    ]:
        meta_tf: Maybe[MetaTransformer] = Maybe.no()
        tfs: list[TransformerType] = []
        copy_tf: Maybe[CopyTransformer] = Maybe.no()
        for tf in self._pipeline:
            match tf:
                case MetaTransformer():
                    if not meta_tf:
                        meta_tf = Maybe(tf)
                case CopyTransformer():
                    copy_tf = Maybe(tf)
                case _:
                    tfs.append(tf)
        return meta_tf, tfs, copy_tf

    def pipe(self, **meta: Any) -> Transformer:
        next = cast(PipelineTransformer,
            Transformer(self.outputs, permalink=self._permalink, **meta)
        )
        # Before joining, split off the CopyTransformer from the end of this
        # pipeline.
        current_metatf, current_pipeline, current_copytf = self._partitioned()
        # Also split off the extra MetaTransformer from the start of next.
        next_metatf, next_pipeline, next_copytf = next._partitioned()
        # Pipelines created by a call to `Transformer` should start with a
        # MetaTransformer and end with a CopyTransformer, but there's no way of
        # knowing if self was created that way. Either way, make sure the
        # pipeline starts and ends properly.
        head = [
            *current_metatf.or_maybe(next_metatf),
            *current_pipeline,
        ]
        if 'permalink' in meta:
            # If a permalink has been specified, allow it to override this one.
            tail = [
                *next_pipeline,
                *next_copytf,
            ]
        else:
            # If no permalink specified, don't let the CopyTransformer at the
            # end of the new one override one that might already be in this
            # pipeline.
            tail = [
                *next_pipeline,
                *current_copytf.or_maybe(next_copytf),
            ]
        updated = PipelineTransformer(self.source, pipeline=head + tail)
        updated.metadata = self.metadata
        if self.has_preprocessed():
            updated.preprocess(self.source, self.src_root, self.dest_root)
        return updated

    @classmethod
    def build(
        cls,
        src_path: AnySource,
        /, *,
        transformers: Sequence[Type[TransformerType]],
        **meta: Any
    ) -> PipelineTransformer:
        tfs: list[TransformerType] = []
        src_path = source(src_path)
        metadata: dict[str, Any] = {}
        for transformer_type in transformers:
            transformer = transformer_type(src_path, **meta)
            tfs.append(transformer)
            metadata.update(merge_dicts(metadata, transformer.metadata))
        new_pipeline = cls(src_path, pipeline=tfs)
        new_pipeline._pipeline = tfs
        new_pipeline.metadata = metadata
        return new_pipeline


class CopyTransformer(BaseTransformer):
    """A simple transformer that copies a file to its destination"""

    def __init__(
        self,
        src_path: AnySource,
        /, *,
        permalink: str=DEFAULT_PERMALINK,
        collection_root: AnySource='.',
        **meta: Any,
    ):
        super().__init__(src_path, **meta)
        self._permalink = permalink
        self._collection_root = source(collection_root)
        if collection_root not in self.source.parents:
            # This state indicates the path has already been generated,
            # an outcome that can happen if two CopyTransformers are
            # present in the same pipeline.
            self._collection_root = LocalPath('.')

    def _generate_path_info(self) -> Self:
        self.metadata['path'] = self._get_path()
        self.metadata['file'] = self._get_path(as_filename=True)
        self.metadata['dir'] = str(self.metadata['path'].parent)
        try:
            self.metadata['url'] = self.metadata['site_url'] / self.metadata['path']
        except KeyError:
            self.metadata['url'] = self.metadata['path']
        return self

    def preprocess(
        self, path: AnySource, src_root: AnySource='.', dest_root: AnyDest='.'
    ) -> Self:
        if not self.has_preprocessed():
            super().preprocess(path, src_root, dest_root)
            self._generate_path_info()
        return self

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(source={str(self.source)!r},'
            f' permalink={self._permalink!r})'
        )

    def transform_file(
        self, source: ReadablePath, dest: WriteablePath
    ) -> WriteablePath:
        # If these are both real paths, no need read and write the bytes, just
        # copy the file.
        if isinstance(source, os.PathLike) and isinstance(dest, os.PathLike):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, dest)
            return dest
        return super().transform_file(source, dest)

    def transform_data(self, data: AnyStr) -> AnyStr:
        return data

    @overload
    def _get_path(self) -> UrlPath: ...
    @overload
    def _get_path(self, as_filename: Literal[True]) -> LocalPath: ...
    def _get_path(self, as_filename: bool=False) -> FilePath:
        path = self.source.parent / self.transformed_name()
        path_components = {
            'path': path.parent.relative_to(self._collection_root),
            'name': path.name,
            'basename': path.stem,
            'ext': path.suffix,
        }

        try:
            result = format_permalink(self._permalink, {**self.metadata, **path_components})
            if as_filename:
                if not result.endswith(path.suffix):
                    result += path.suffix
                return LocalPath(result).relative_to('/')
            return UrlPath(result).absolute()
        except KeyError as exc:
            raise ValueError(
                f'Cannot create filename from metadata element {exc}'
                f' - metadata: {self.metadata}'
            ) from exc

    def get_output_path(self) -> LocalPath:
        try:
            return cast(LocalPath, self.metadata['file'])
        except KeyError:
            self._generate_path_info()
            return cast(LocalPath, self.metadata['file'])


class TextTransformer(BaseTransformer, ABC):
    def transform_data(self, data: AnyStr) -> str:
        text = data.decode('utf8') if isinstance(data, bytes) else data
        return self.transform_text(text)

    @abstractmethod
    def transform_text(self, text: str) -> str: ...


class MetaProcessor(Protocol):
    def __call__(
        self,
        src_path: str | bytes | AnySource,
        **metadata: Any,
    ) -> str: ...


class MetaTransformer(TextTransformer):
    @classmethod
    def __matcher(  # pylint: disable=unused-private-member
        cls, *_: Any, parse_frontmatter: bool=True, **__: Any
    ) -> bool:
        return parse_frontmatter

    def __init__(
        self, src_path: AnySource, /,
        **meta: Any,
    ):
        super().__init__(src_path, **meta)

    def preprocess(
        self, path: AnySource, src_root: AnySource='.', dest_root: AnyDest='.'
    ) -> Self:
        super().preprocess(path, src_root, dest_root)
        input = self.src_root / self.source
        self.transform_data(input.read_bytes())
        if 'date' not in self.metadata:
            self.metadata['date'] = AutoDate(input.timestamp())
        return self

    def transform_text(self, text: str) -> str:
        frontmatter, content = self.split_frontmatter(text)
        metadata = parse_yaml_dict(frontmatter) if frontmatter else {}
        self.metadata.update(merge_dicts(self.metadata, metadata))
        return content

    @staticmethod
    def split_frontmatter(text: str) -> tuple[str | None, str]:
        """Split a file into the frontmatter and text file components"""
        if not text.startswith("---\n"):
            return None, text
        end = text.find("\n---\n", 3)
        frontmatter = text[4:end]
        text = text[end + 5:]
        return frontmatter, text


class MetaProcessorTransformer(TextTransformer):
    @classmethod
    def __matcher(  # pylint: disable=unused-private-member
        cls, *_: Any, metaprocessor: MetaProcessor | None=None, **__: Any
    ) -> bool:
        return metaprocessor is not None

    def __init__(
        self, src_path: AnySource, /,
        metaprocessor: MetaProcessor,
        **meta: Any,
    ):
        super().__init__(src_path, **meta)
        self._processor = metaprocessor

    def transform_text(self, text: str) -> str:
        return self._processor(text, **self.metadata)


class MarkdownTransformer(TextTransformer, pattern='*.md'):
    """Transform markdown to HTML"""
    PARA_RE = re.compile('<p[^>]*>(.*?)</p>', flags=re.DOTALL)
    markdown_parser = MarkdownParser()

    def transformed_name(self) -> str:
        return str(self.source.with_suffix('.html').name)

    def transform_text(self, text: str) -> str:
        html = self.markdown_parser.parse(text)
        page = self.metadata
        try:
            if page.get('description'):
                page['description'] = self.markdown_parser.parse(page['description'])
            page['excerpt'] = self.PARA_RE.search(html)[0]  # type: ignore [index]
            page['word_count'] = 1 + ilen(re.finditer(r'\s+', Markup(html).striptags()))
        except (TypeError, IndexError):
            page['excerpt'] = ''
            page['word_count'] = 0
        return html


class TemplateApplyTransformer(TextTransformer):
    """Apply a Jinja2 Template to a text file"""

    @classmethod
    def __matcher(  # pylint: disable=unused-private-member
        cls, *_: Any, template: Template | None=None, **__: Any
    ) -> bool:
        return template is not None

    def __init__(self, src_path: AnySource, /, *, template: Template, **meta: Any):
        super().__init__(src_path, **meta)
        self._template = template

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'{str(self.source)!r}, {self._template!r}, **{self.metadata})'
        )

    def transform_text(self, text: str) -> str:
        results = self._template.render(
            content=text,
            page=Data(self.metadata | {'content': text}),
        )
        return results
