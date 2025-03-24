from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AnyStr, ClassVar, Protocol, Self, Type, cast

from jinja2 import Template

from .markdown import markdownify
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

    def transform(self, data: AnyStr) -> str | bytes: ...

    def pipe(self, **meta: Any) -> Transformer: ...

    def preprocess(self, src_root: Path) -> None: ...

    def transform_from_file(self, src_root: Path) -> str | bytes:
        input = src_root / self.source
        data = input.read_bytes()
        return self.transform(data)

    def transform_to_file(self, data: AnyStr, dest_root: Path) -> Path:
        output = dest_root / self.outputs
        transformed = self.transform(data)  # pylint: disable=assignment-from-no-return
        if isinstance(transformed, bytes):
            output.write_bytes(transformed)
        else:
            output.write_text(transformed)
        return output

    def transform_file(self, src_root: Path, dest_root: Path) -> Path:
        data = cast(bytes, self.transform_from_file(src_root))
        return self.transform_to_file(data, dest_root)


class Transformer:
    __slots__ = ('_source',)
    _source: Path
    registered: ClassVar[list[TransformerRegistration]] = []  # pylint: disable=declare-non-slot

    def __new__(
        cls,
        source: Path,
        /, *,
        permalink: str=DEFAULT_PERMALINK,
        template: Template | None=None,
        metaprocessor: MetaProcessor | None=None,
        **meta: Any,
    ) -> Transformer:
        _ = source, permalink, meta
        if cls is Transformer:
            transformers: list[type] = [MetaTransformer]
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
        return super().__new__(cls)

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
        @property
        def outputs(self) -> Path: ...
        @property
        def metadata(self) -> dict[str, Any]:
            return {}
        def pipe(self, **meta: Any) -> Transformer:
            _ = meta
            return self
        def preprocess(self, src_root: Path) -> None:
            _ = src_root
        def transform_from_file(self, src_root: Path) -> str | bytes:
            _ = src_root
        def transform_to_file(self, data: AnyStr, dest_root: Path) -> Path:
            _ = data, dest_root
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
        /, **kwargs: Any,
    ):
        super().__init__()
        _ = kwargs
        self._source = source
        self._meta = {}

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

    def pipe(self, **meta: Any) -> Transformer:
        next = Transformer(self.outputs, **meta)
        return PipelineTransformer(
            self.source, pipeline=[self, next],
        )

    def preprocess(self, src_root: Path) -> None:
        _ = src_root


class PipelineTransformer(BaseTransformer):
    _pipeline: Sequence[Transformer]

    @property
    def outputs(self) -> Path:
        return self._pipeline[-1].outputs

    def preprocess(self, src_root: Path) -> None:
        metadata: dict[str, Any] = {}
        input = self.source
        for pipe in self._pipeline:
            print(metadata)
            pipe.metadata.update(metadata)
            pipe.preprocess(src_root)
            pipe._source = input
            input = pipe.outputs
            metadata = pipe.metadata

    def transform(self, data: AnyStr) -> str | bytes:
        current_data: str | bytes = data
        metadata: dict[str, Any] = {}
        for pipe in self._pipeline:
            pipe.metadata.update(metadata)
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
        try:
            # This should be set either through init or the build classmethod.
            self._pipeline
        except AttributeError:
            self._pipeline = ()

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
            #input = transformer.outputs
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
        **meta: Any,
    ):
        super().__init__(source, **meta)
        self._permalink = TO_FORMAT_STR_RE.sub('{\\1}', permalink)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(source={str(self._source)!r},'
            f' permalink={self._permalink!r})'
        )

    def transform_file(self, src_root: Path, dest_root: Path) -> Path:
        in_path = src_root / self.source
        out_path = dest_root / self.outputs
        shutil.copy(in_path, out_path)
        return out_path

    def transform(self, data: AnyStr) -> AnyStr:
        return data

    def get_output_path(self) -> Path:
        path = self._source.parent / self.transformed_name()
        try:
            return Path(
                self._permalink.format(
                    path=path.parent,
                    name=path.name,
                    basename=path.stem,
                    ext=path.suffix,
                    **self.metadata,
                )
            ).relative_to('/')
        except KeyError as exc:
            raise ValueError(
                f'Cannot create filename from metadata element: {exc}'
            ) from exc


class TextTransformer(BaseTransformer, ABC):
    def transform(self, data: AnyStr) -> str:
        text = data.decode('utf8') if isinstance(data, bytes) else data
        return self.transform_text(text)

    @abstractmethod
    def transform_text(self, text: str) -> str: ...


class MarkdownTransformer(TextTransformer, pattern='*.md'):
    """Transform markdown to HTML"""

    def transformed_name(self) -> str:
        return str(self._source.with_suffix('.html').name)

    def transform_text(self, text: str) -> str:
        return markdownify(text)


class TemplateApplyTransformer(TextTransformer):
    """Apply a Jinja2 Template to a text file"""

    def __init__(self, source: Path, /, *, template: Template, **meta: Any):
        super().__init__(source)
        self._template = template
        self._meta = meta

    def transform_text(self, text: str) -> str:
        return self._template.render(content=text, **self._meta)


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

    def preprocess(self, src_root: Path) -> None:
        self.transform_from_file(src_root)

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
