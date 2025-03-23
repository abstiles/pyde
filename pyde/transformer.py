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

TO_FORMAT_STR_RE = re.compile(r':(\w+)')
DEFAULT_PERMALINK = '/:path/:name'
UNSET: Any = object()


class TransformerType(Protocol):
    @property
    def source(self) -> Path: ...

    @property
    def outputs(self) -> Path: ...

    def transform(self, data: AnyStr) -> str | bytes: ...

    def transform_file(self, root: Path) -> None: ...

    def pipe(self, **meta: Any) -> Transformer: ...


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
        **meta: Any,
    ) -> Transformer:
        _ = source, permalink, meta
        if cls is Transformer:
            transformers: list[type] = []
            for registration in cls.registered:
                if registration.wants(source):
                    transformers.append(registration.transformer)
            if template is not None:
                transformers.append(TemplateApplyTransformer)
            transformers.append(CopyTransformer)
            if len(transformers) > 1:
                return PipelineTransformer.build(
                    source, permalink=permalink, template=template,
                    transformers=transformers, **meta,
                )
            cls = transformers[0]
        return super().__new__(cls)

    def __init_subclass__(cls, /, pattern: str='', **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if pattern:
            Transformer.register(cls, Transformer._pattern_matcher(pattern))

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
        def transform_file(self, root: Path) -> None:
            _ = root
        def transform(self, data: AnyStr) -> str | bytes:
            return data
        def pipe(self, **meta: Any) -> Transformer:
            _ = meta
            return self


@dataclass(frozen=True)
class TransformerRegistration:
    transformer: Type[Transformer]
    wants: Callable[[Path], bool]


class BaseTransformer(Transformer, TransformerType):
    __slots__ = ()

    def __init__(
        self,
        source: Path,
        /, **meta: Any,
    ):
        super().__init__()
        _ = meta
        self._source = source

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.source)!r})'

    def transformed_name(self) -> str:
        return self._source.name

    def get_output_path(self) -> Path:
        return self._source.parent / self.transformed_name()

    @property
    def outputs(self) -> Path:
        return self.get_output_path()

    def transform_file(self, root: Path) -> None:
        input = root / self.source
        output = root / self.outputs
        text = input.read_text('utf8')
        data = self.transform(text)
        if isinstance(data, bytes):
            output.write_bytes(data)
        else:
            output.write_text(data)

    def pipe(self, **meta: Any) -> Transformer:
        next = Transformer(self.outputs, **meta)
        return PipelineTransformer(
            self.source, pipeline=[self, next],
        )


class PipelineTransformer(BaseTransformer):
    _pipeline: Sequence[Transformer]

    @property
    def outputs(self) -> Path:
        return self._pipeline[-1].outputs

    def transform(self, data: AnyStr) -> str | bytes:
        current_data: str | bytes = data
        for pipe in self._pipeline:
            current_data = pipe.transform(cast(AnyStr, current_data))
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
            input = transformer.outputs
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

    def transform_file(self, root: Path) -> None:
        shutil.copy(self.source, self.outputs)

    def transform(self, data: AnyStr) -> AnyStr:
        return data

    def get_output_path(self) -> Path:
        path = self._source.parent / self.transformed_name()
        return Path(
            self._permalink.format(
                path=path.parent,
                name=path.name,
                basename=path.stem,
                ext=path.suffix,
            )
        ).relative_to('/')


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

    def __init__(
        self,
        source: Path,
        /, *,
        template: Template,
        **meta: Any
    ):
        super().__init__(source)
        self._template = template
        self._meta = meta

    def transform_text(self, text: str) -> str:
        return self._template.render(content=text, **self._meta)
