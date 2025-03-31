from __future__ import annotations

import os
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
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

from pyde.path.filepath import WriteablePath

from .data import Data
from .markdown import markdownify
from .path import AnyDest, AnySource, FilePath, LocalPath, ReadablePath, UrlPath, source
from .utils import Maybe, ilen, merge_dicts
from .yaml import parse_yaml_dict

TO_FORMAT_STR_RE = re.compile(r':(\w+)')
DEFAULT_PERMALINK = '/:path/:name'
UNSET: Any = object()

class TransformerType(Protocol):
    @property
    def source(self) -> ReadablePath: ...

    @property
    def outputs(self) -> WriteablePath: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    @metadata.setter
    def metadata(self, meta: dict[str, Any]) -> None: ...

    def transform(self, data: AnyStr) -> str | bytes: ...

    def pipe(self, **meta: Any) -> Transformer: ...

    def preprocess(self, src_root: AnySource) -> Self: ...

    def transform_from_file(self, src_root: AnySource) -> str | bytes:
        input = source(src_root) / self.source
        data = input.read_bytes()
        return self.transform(data)

    def transform_to_file(
        self, data: AnyStr, dest_root: AnyDest
    ) -> WriteablePath:
        output = dest_root / self.outputs
        output.parent.mkdir(parents=True, exist_ok=True)
        transformed = self.transform(data)  # pylint: disable=assignment-from-no-return
        if isinstance(transformed, bytes):
            output.write_bytes(transformed)
        else:
            output.write_text(transformed)
        return output

    def transform_file(
        self, src_root: AnySource, dest_root: AnyDest
    ) -> WriteablePath:
        input = source(src_root) / self.source
        data = input.read_bytes()
        return self.transform_to_file(data, dest_root)


@dataclass(frozen=True)
class TransformerRegistration:
    transformer: Type[Transformer]
    wants: Callable[[FilePath], bool]


class Transformer(TransformerType):
    __slots__ = ('_source', '_meta')
    _source: ReadablePath
    _meta: dict[str, Any]
    registered: ClassVar[list[TransformerRegistration]] = []  # pylint: disable=declare-non-slot

    def __new__(
        cls,
        src_path: ReadablePath | Path | str,
        /, *,
        parse_frontmatter: bool=True,
        permalink: str=DEFAULT_PERMALINK,
        template: Template | None=None,
        metaprocessor: MetaProcessor | None=None,
        **meta: Any,
    ) -> Transformer:
        _ = meta
        src_path = source(src_path)
        if cls is Transformer:
            transformers: list[type[Transformer]] = []
            if parse_frontmatter:
                transformers.append(MetaTransformer)
            if metaprocessor:
                transformers.append(MetaProcessorTransformer)
            for registration in cls.registered:
                if registration.wants(src_path):
                    transformers.append(registration.transformer)
            if template is not None:
                transformers.append(TemplateApplyTransformer)
            transformers.append(CopyTransformer)
            if len(transformers) > 1:
                return PipelineTransformer.build(
                    src_path, permalink=permalink, template=template,
                    transformers=transformers, metaprocessor=metaprocessor,
                    **meta,
                )
            cls = transformers[0]
        return super().__new__(cls)  # pyright: ignore

    def __init_subclass__(cls, /, pattern: str='', **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if pattern:
            Transformer.register(cls, Transformer._pattern_matcher(pattern))
        elif matcher := getattr(cls, f'__{cls.__name__}_match_path', None):
            Transformer.register(cls, matcher)


    @staticmethod
    def _pattern_matcher(pattern: str) -> Callable[[FilePath], bool]:
        def matcher(path: FilePath) -> bool:
            return path.match(pattern)
        return matcher

    @classmethod
    def register(
        cls,
        transformer: Type[Transformer],
        wants: Callable[[FilePath], bool],
    ) -> Type[Self]:
        cls.registered.append(TransformerRegistration(transformer, wants))
        return cls

    @property
    def source(self) -> ReadablePath:
        return self._source

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
        def _set_meta(self, meta: dict[str, Any]) -> Self:
            _ = meta
            return self
        def pipe(self, **meta: Any) -> Transformer:
            _ = meta
            return self
        def preprocess(self, src_root: AnySource) -> Self:
            _ = src_root
            return self
        def transform_from_file(self, src_root: AnySource) -> str | bytes:
            return str(src_root)
        def transform_to_file(
            self, data: AnyStr, dest_root: AnyDest
        ) -> WriteablePath:
            return cast(WriteablePath, data)
        def transform_file(
            self, src_root: AnySource, dest_root: AnyDest
        ) -> WriteablePath:
            return cast(WriteablePath, (src_root, dest_root))
        def transform(self, data: AnyStr) -> str | bytes:
            return data


class BaseTransformer(Transformer):
    __slots__ = ('_preprocessed',)
    _preprocessed: ReadablePath | None

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
        super().__init__()
        _ = parse_frontmatter, permalink, template, metaprocessor
        self._source = source(src_path)
        self._meta = meta
        self._preprocessed = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.source)!r})'

    def transformed_name(self) -> str:
        return self._source.name

    def get_output_path(self) -> WriteablePath:
        return self._source.parent / self.transformed_name()

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
        next = Transformer(self.outputs, **meta)
        return PipelineTransformer(
            self.source, pipeline=[self, next],
        )

    def preprocess(self, src_root: AnySource) -> Self:
        self._preprocessed = source(src_root)
        return self


class PipelineTransformer(BaseTransformer):
    _pipeline: Sequence[Transformer]

    @property
    def outputs(self) -> WriteablePath:
        return self._pipeline[-1].outputs

    def _set_meta(self, meta: dict[str, Any]) -> Self:
        super()._set_meta(meta)
        for pipe in self._pipeline:
            pipe.metadata = self.metadata
        return self

    def preprocess(self, src_root: AnySource) -> Self:
        src_root = source(src_root)
        if self._preprocessed == src_root:
            return self
        metadata: dict[str, Any] = self.metadata
        input = self.source
        for pipe in self._pipeline:
            pipe.metadata = metadata
            pipe._source = input
            pipe.preprocess(src_root)
            input = pipe.outputs
        self._preprocessed = src_root
        return self

    def transform(self, data: AnyStr) -> str | bytes:
        current_data: str | bytes = data
        metadata: dict[str, Any] = self.metadata
        for pipe in self._pipeline:
            pipe.metadata = metadata
            current_data = pipe.transform(cast(AnyStr, current_data))
        return current_data

    def __repr__(self) -> str:
        args = ', '.join(map(repr, self._pipeline))
        return f'{self.__class__.__name__}({args})'

    def __init__(
        self, src_path: AnySource, /, *,
        pipeline: Sequence[Transformer]=UNSET,
        permalink: str=DEFAULT_PERMALINK,
        **meta: Any
    ):
        super().__init__(src_path, **meta)
        if pipeline is not UNSET:
            self._pipeline = pipeline
        self._preprocessed = None
        self._permalink = permalink

    def __getitem__(self, idx: int | slice) -> Transformer | PipelineTransformer:
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
    ) -> tuple[Maybe[MetaTransformer], Sequence[Transformer], Maybe[CopyTransformer]]:
        meta_tf: Maybe[MetaTransformer] = Maybe.no()
        tfs: list[Transformer] = []
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
        # Pipelines created by a call to `Transformer` are guaranteed to start
        # with a MetaTransformer and end with a CopyTransformer, but there's no
        # way of knowing if self was created that way. Either way, make sure
        # the pipeline starts and ends properly.
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
        if (src_path := self._preprocessed) is not None:
            updated.preprocess(src_path)
        return updated

    @classmethod
    def build(
        cls,
        src_path: AnySource,
        /, *,
        transformers: Sequence[Type[Transformer]],
        **meta: Any
    ) -> PipelineTransformer:
        tfs: list[Transformer] = []
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
        self._permalink = TO_FORMAT_STR_RE.sub('{\\1}', permalink)
        self._collection_root = source(collection_root)
        if collection_root not in self._source.parents:
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

    def preprocess(self, src_root: AnySource) -> Self:
        if self._preprocessed is not None:
            super().preprocess(src_root)
            self._generate_path_info()
        return self

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(source={str(self._source)!r},'
            f' permalink={self._permalink!r})'
        )

    def transform_file(
        self, src_root: AnySource, dest_root: AnyDest
    ) -> WriteablePath:
        src_root = source(src_root)
        in_path = src_root / self.source
        out_path = dest_root / self.outputs
        # If these are both real paths, no need read and write the bytes, just
        # copy the file.
        if isinstance(in_path, os.PathLike) and isinstance(out_path, os.PathLike):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(in_path, out_path)
            return out_path
        return self.transform_to_file(in_path.read_bytes(), dest_root)

    def transform(self, data: AnyStr) -> AnyStr:
        return data

    @overload
    def _get_path(self) -> UrlPath: ...
    @overload
    def _get_path(self, as_filename: Literal[True]) -> LocalPath: ...
    def _get_path(self, as_filename: bool=False) -> FilePath:
        path = self._source.parent / self.transformed_name()
        path_components = {
            'path': path.parent.relative_to(self._collection_root),
            'name': path.name,
            'basename': path.stem,
            'ext': path.suffix,
        }

        try:
            result = self._permalink.format(**{**self.metadata, **path_components})
            if as_filename:
                if not result.endswith(path.suffix):
                    result += path.suffix
                return LocalPath(result).relative_to('/')
            return UrlPath(result).absolute().deindexed
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
    def transform(self, data: AnyStr) -> str:
        text = data.decode('utf8') if isinstance(data, bytes) else data
        return self.transform_text(text)

    @abstractmethod
    def transform_text(self, text: str) -> str: ...


class MarkdownTransformer(TextTransformer, pattern='*.md'):
    """Transform markdown to HTML"""
    PARA_RE = re.compile('<p[^>]*>(.*?)</p>', flags=re.DOTALL)

    def transformed_name(self) -> str:
        return str(self._source.with_suffix('.html').name)

    def transform_text(self, text: str) -> str:
        html = markdownify(text)
        page = self.metadata
        try:
            page['excerpt'] = self.PARA_RE.search(html)[0]  # type: ignore [index]
            page['word_count'] = 1 + ilen(re.finditer(r'\s+', Markup(html).striptags()))
        except (TypeError, IndexError):
            page['excerpt'] = ''
            page['word_count'] = 0
        return html


class TemplateApplyTransformer(TextTransformer):
    """Apply a Jinja2 Template to a text file"""

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


class MetaProcessor(Protocol):
    def __call__(
        self,
        src_path: str | bytes | AnySource,
        **metadata: Any,
    ) -> str: ...


class MetaTransformer(TextTransformer):
    @classmethod
    def __match_path(cls, path: FilePath) -> bool:  # pylint: disable=unused-private-member
        return bool(path)

    def __init__(
        self, src_path: AnySource, /,
        **meta: Any,
    ):
        super().__init__(src_path, **meta)

    def preprocess(self, src_root: AnySource) -> Self:
        self.transform_from_file(src_root)
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
    def __match_path(cls, path: FilePath) -> bool:  # pylint: disable=unused-private-member
        return bool(path)

    def __init__(
        self, src_path: AnySource, /,
        metaprocessor: MetaProcessor,
        **meta: Any,
    ):
        super().__init__(src_path, **meta)
        self._processor = metaprocessor

    def transform_text(self, text: str) -> str:
        return self._processor(text, **self.metadata)
