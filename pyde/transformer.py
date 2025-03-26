from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AnyStr, ClassVar, Protocol, Self, Type, cast

from jinja2 import Template
from markupsafe import Markup

from .markdown import markdownify
from .utils import Maybe, ilen, merge_dicts
from .yaml import parse_yaml_dict

TO_FORMAT_STR_RE = re.compile(r':(\w+)')
DEFAULT_PERMALINK = '/:path/:name'
UNSET: Any = object()


class TransformerType(Protocol):
    @property
    def source(self) -> Path: ...

    @property
    def outputs(self) -> Path: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    def set_meta(self, meta: dict[str, Any]) -> Self: ...

    def transform(self, data: AnyStr) -> str | bytes: ...

    def pipe(self, **meta: Any) -> Transformer: ...

    def preprocess(self, src_root: Path) -> Self: ...

    def transform_from_file(self, src_root: Path) -> str | bytes:
        input = src_root / self.source
        data = input.read_bytes()
        return self.transform(data)

    def transform_to_file(self, data: AnyStr, dest_root: Path) -> Path:
        output = dest_root / self.outputs
        output.parent.mkdir(parents=True, exist_ok=True)
        transformed = self.transform(data)  # pylint: disable=assignment-from-no-return
        if isinstance(transformed, bytes):
            output.write_bytes(transformed)
        else:
            output.write_text(transformed)
        return output

    def transform_file(self, src_root: Path, dest_root: Path) -> Path:
        input = src_root / self.source
        data = input.read_bytes()
        return self.transform_to_file(data, dest_root)


class Transformer:
    __slots__ = ('_source',)
    _source: Path
    registered: ClassVar[list[TransformerRegistration]] = []  # pylint: disable=declare-non-slot

    def __new__(
        cls,
        source: Path,
        /, *,
        parse_frontmatter: bool=True,
        permalink: str=DEFAULT_PERMALINK,
        template: Template | None=None,
        metaprocessor: MetaProcessor | None=None,
        **meta: Any,
    ) -> Transformer:
        _ = source, permalink, meta
        if cls is Transformer:
            transformers: list[type[Transformer]] = []
            if parse_frontmatter:
                transformers.append(MetaTransformer)
            if metaprocessor:
                transformers.append(MetaProcessorTransformer)
            for registration in cls.registered:
                if registration.wants(source):
                    transformers.append(registration.transformer)
            if template is not None:
                transformers.append(TemplateApplyTransformer)
            transformers.append(CopyTransformer)
            if len(transformers) > 1:
                return PipelineTransformer.build(
                    source, permalink=permalink, template=template,
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
    def _pattern_matcher(pattern: str) -> Callable[[Path], bool]:
        def matcher(path: Path) -> bool:
            return path.match(pattern)
        return matcher

    @classmethod
    def register(
        cls,
        transformer: Type[Transformer],
        wants: Callable[[Path], bool],
    ) -> Type[Self]:
        cls.registered.append(TransformerRegistration(transformer, wants))
        return cls

    @property
    def source(self) -> Path:
        return self._source

    if TYPE_CHECKING:
        _meta: dict[str, Any]
        @property
        def outputs(self) -> Path: ...
        @property
        def metadata(self) -> dict[str, Any]:
            return {}
        def set_meta(self, meta: dict[str, Any]) -> Self:
            _ = meta
            return self
        def pipe(self, **meta: Any) -> Transformer:
            _ = meta
            return self
        def preprocess(self, src_root: Path) -> Self:
            _ = src_root
            return self
        def transform_from_file(self, src_root: Path) -> str | bytes:
            return str(src_root)
        def transform_to_file(self, data: AnyStr, dest_root: Path) -> Path:
            _ = data
            return dest_root
        def transform_file(self, src_root: Path, dest_root: Path) -> Path:
            _ = src_root, dest_root
            return dest_root
        def transform(self, data: AnyStr) -> str | bytes:
            return data


@dataclass(frozen=True)
class TransformerRegistration:
    transformer: Type[Transformer]
    wants: Callable[[Path], bool]


class BaseTransformer(Transformer, TransformerType):
    __slots__ = ('_meta',)
    _meta: dict[str, Any]

    def __init__(
        self,
        source: Path,
        /, *,
        parse_frontmatter: bool=True,
        permalink: str=DEFAULT_PERMALINK,
        template: Template | None=None,
        metaprocessor: MetaProcessor | None=None,
        **meta: Any,
    ):
        super().__init__()
        _ = parse_frontmatter, permalink, template, metaprocessor
        self._source = source
        self._meta = meta

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.source)!r})'

    def transformed_name(self) -> str:
        return self._source.name

    def get_output_path(self) -> Path:
        return self._source.parent / self.transformed_name()

    @property
    def outputs(self) -> Path:
        return self.get_output_path()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._meta

    def set_meta(self, meta: dict[str, Any]) -> Self:
        #self._meta.setdefault('page', {}).update(meta.get('page', {}))
        #self._meta.update({k: meta[k] for k in meta if k != 'page'})
        self._meta = merge_dicts(self._meta, meta)
        return self

    def pipe(self, **meta: Any) -> Transformer:
        next = Transformer(self.outputs, **meta)
        return PipelineTransformer(
            self.source, pipeline=[self, next],
        )

    def preprocess(self, src_root: Path) -> Self:
        _ = src_root
        return self


class PipelineTransformer(BaseTransformer):
    _pipeline: Sequence[Transformer]
    _preprocessed: Path | None

    @property
    def outputs(self) -> Path:
        return self._pipeline[-1].outputs

    def preprocess(self, src_root: Path) -> Self:
        if self._preprocessed == src_root:
            return self
        metadata: dict[str, Any] = self.metadata
        input = self.source
        for pipe in self._pipeline:
            pipe.set_meta(metadata)
            pipe._source = input
            pipe.preprocess(src_root)
            input = pipe.outputs
            metadata = pipe.metadata
        self._meta = metadata
        self._preprocessed = src_root
        return self

    def transform(self, data: AnyStr) -> str | bytes:
        current_data: str | bytes = data
        metadata: dict[str, Any] = self.metadata
        for pipe in self._pipeline:
            pipe.set_meta(metadata)
            current_data = pipe.transform(cast(AnyStr, current_data))
            metadata = pipe.metadata
        return current_data

    def __repr__(self) -> str:
        args = ', '.join(map(repr, self._pipeline))
        return f'{self.__class__.__name__}({args})'

    def __init__(
        self, source: Path, /, *,
        pipeline: Sequence[Transformer]=UNSET,
        **meta: Any
    ):
        super().__init__(source, **meta)
        if pipeline is not UNSET:
            self._pipeline = pipeline
        self._preprocessed = None

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
        next = cast(PipelineTransformer, Transformer(self.outputs, **meta))
        if (src_path := self._preprocessed) is not None:
            next[0].set_meta(self._meta)
            next[1:].set_meta(self._meta).preprocess(src_path)
        # Before joining, split off the CopyTransformer from the end of this
        # pipeline.
        current_metatf, current_pipeline, current_copytf = self._partitioned()
        # Also split off the extra MetaTransformer from the start of next.
        next_metatf, next_pipeline, next_copytf = next._partitioned()
        # Pipelines created by a call to `Transformer` are guaranteed to start
        # with a MetaTransformer and end with a CopyTransformer, but there's no
        # way of knowing if self was created that way. Either way, make sure
        # the pipeline starts and ends properly.
        if 'permalink' not in meta:
            # If no permalink specified, don't let the CopyTransformer at the
            # end of the new one override one that might already be in this
            # pipeline.
            pipeline = [
                *current_metatf.or_maybe(next_metatf),
                *current_pipeline,
                *next_pipeline,
                *current_copytf.or_maybe(next_copytf),
            ]
        else:
            # If a permalink has been specified, allow it to override this one.
            pipeline = [
                *current_metatf.or_maybe(next_metatf),
                *current_pipeline,
                *next_pipeline,
                *next_copytf,
            ]
        return PipelineTransformer(self.source, pipeline=pipeline).set_meta(self._meta)

    @classmethod
    def build(
        cls,
        source: Path,
        /, *,
        transformers: Sequence[Type[Transformer]],
        **meta: Any
    ) -> PipelineTransformer:
        tfs: list[Transformer] = []
        input = source
        for transformer_type in transformers:
            transformer = transformer_type(input, **meta)
            tfs.append(transformer)
        new_pipeline = cls(source, pipeline=tfs)
        new_pipeline._pipeline = tfs
        return new_pipeline


class CopyTransformer(BaseTransformer):
    """A simple transformer that copies a file to its destination"""

    def __init__(
        self,
        source: Path,
        /, *,
        permalink: str=DEFAULT_PERMALINK,
        collection_root: Path=Path('.'),
        **meta: Any,
    ):
        super().__init__(source, **meta)
        self._permalink = TO_FORMAT_STR_RE.sub('{\\1}', permalink)
        self._collection_root = collection_root

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(source={str(self._source)!r},'
            f' permalink={self._permalink!r})'
        )

    def transform_file(self, src_root: Path, dest_root: Path) -> Path:
        in_path = src_root / self.source
        out_path = dest_root / self.outputs
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(in_path, out_path)
        return out_path

    def transform(self, data: AnyStr) -> AnyStr:
        return data

    def _get_path(self, as_filename: bool=False) -> Path:
        path = self._source.parent / self.transformed_name()
        path_components = {
            'path': path.parent.relative_to(self._collection_root),
            'name': path.name,
            'basename': path.stem,
            'ext': path.suffix,
        }

        try:
            result = self._permalink.format(**path_components, **self.metadata)
            if as_filename and not result.endswith(path.suffix):
                result += path.suffix
            return Path(result).relative_to('/')
        except KeyError as exc:
            raise ValueError(
                f'Cannot create filename from metadata element {exc}'
                f' - metadata: {self.metadata}'
            ) from exc

    def get_permalink(self) -> Path:
        return self._get_path()

    def get_output_path(self) -> Path:
        return self._get_path(as_filename=True)


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
        page = {}
        page = self._meta.setdefault('page', {})
        try:
            page['excerpt'] = self.PARA_RE.search(html)[0]  # type: ignore [index]
            page['word_count'] = 1 + ilen(re.finditer(r'\s+', Markup(html).striptags()))
        except (TypeError, IndexError):
            page['excerpt'] = ''
            page['word_count'] = 0
        return html


class TemplateApplyTransformer(TextTransformer):
    """Apply a Jinja2 Template to a text file"""

    def __init__(self, source: Path, /, *, template: Template, **meta: Any):
        super().__init__(source)
        self._template = template
        self._meta = meta

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'{str(self.source)!r}, {self._template!r}, **{self._meta})'
        )

    def transform_text(self, text: str) -> str:
        results = self._template.render(content=text, **self._meta)
        return results


class MetaProcessor(Protocol):
    def __call__(
        self,
        source: str | bytes | Path,
        **metadata: Any,
    ) -> str: ...


class MetaTransformer(TextTransformer):
    @classmethod
    def __match_path(cls, path: Path) -> bool:  # pylint: disable=unused-private-member
        return bool(path)

    def __init__(
        self, source: Path, /,
        **meta: Any,
    ):
        super().__init__(source, **meta)
        self._meta: dict[str, Any] = {}

    def preprocess(self, src_root: Path) -> Self:
        self.transform_from_file(src_root)
        return self

    def transform_text(self, text: str) -> str:
        frontmatter, content = self.split_frontmatter(text)
        self._meta.update(
            parse_yaml_dict(frontmatter) if frontmatter else {}
        )
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
    def __match_path(cls, path: Path) -> bool:  # pylint: disable=unused-private-member
        return bool(path)

    def __init__(
        self, source: Path, /,
        metaprocessor: MetaProcessor,
        **meta: Any,
    ):
        super().__init__(source, **meta)
        self._processor = metaprocessor

    def transform_text(self, text: str) -> str:
        return self._processor(text, **self._meta)
