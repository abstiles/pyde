"""Handler for parsing Markdown"""

import io
import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Any, ClassVar, Literal, Self, TypeAlias

from markdown import Extension
from markdown import Markdown as MarkdownEngine

from .extensions.blockquote import BlockQuoteExtension
from .extensions.pm_attr_list import PMAttrListExtension


def default_extensions() -> list[Extension | str]:
    return [
        'md_in_html', 'smarty', 'sane_lists', 'footnotes',
        PMAttrListExtension(), BlockQuoteExtension(),
    ]


def default_ext_configs() -> dict[str, dict[str, Any]]:
    return {
        'smarty': {
            'substitutions': {
                'left-single-quote': '‘',
                'right-single-quote': '’',
                'left-double-quote': '“',
                'right-double-quote': '”',
                'left-angle-quote': '«',
                'right-angle-quote': '»',
                'ellipsis': '…',
                'ndash': '–',
                'mdash': '—',
            }
        }
    }


DASH_PATTERN = (
    r'--'                # hyphens
    r'|–|—'              # or Unicode
    r'|&[mn]dash;'       # or named dash entities
    r'|&#8211;|&#8212;'  # or decimal entities
)

ELLIPSIS_PATTERN = (
    r'…|&hellip;|&#8230;|\.\.\.'
)

SINGLE_QUOTE_PATTERN = (
    r"(?:'"
    r"|‘|’"
    r")"
)

SINGLE_QUOTE_ENTITY_PATTERN = (
    r"(?:"
    r"&apos;|&lsquo;|&rsquo;"
    r"|&#39;|&#8216;|&#8217;"
    r")"
)

DOUBLE_QUOTE_PATTERN = (
    r'(?:"'
    r'|“|”'
    r')'
)

DOUBLE_QUOTE_ENTITY_PATTERN = (
    r'(?:'
    r'&quot;|&ldquo;|&rdquo;'
    r'|&#34;|&#8220;|&#8221;'
    r')'
)

RDQUOTE_FIX_RE = re.compile(
    f'({DASH_PATTERN}|</[^>]+>)'
    f'{DOUBLE_QUOTE_PATTERN}'
    r'(?!\w)'
)

RDQUOTE_FIX_RE2 = re.compile(
    f'{DOUBLE_QUOTE_PATTERN}'
    f'(</[^>]+>)'
)

LDQUOTE_FIX_RE = re.compile(
    f'{DOUBLE_QUOTE_PATTERN}'
    f'({ELLIPSIS_PATTERN})'
)

BLOCKQUOTE_BR_RE = re.compile(r'^>(.*)\\$', flags=re.MULTILINE)
WITHIN_BLOCK_ATTR_RE = re.compile(
    r'\s*\n> (\{:?[^}]*\w[^}]*\})\s*$',
    flags=re.MULTILINE
)

BACKSLASH_LT_RE = re.compile(r'(?<!\\)\\((?:\\{2})*)(?!\\)<')

class MarkdownParser:
    """Markdown parser"""

    def __init__(self, parser: MarkdownEngine | None = None):
        if parser is None:
            parser = MarkdownConfig().get_engine()
        self.parser = parser

    def parse(self, markdown: str) -> str:
        preprocessed_md = self.preprocess(markdown)
        processed_html = self.parser.reset().convert(preprocessed_md)
        return HTMLCleaner.clean(processed_html)

    @staticmethod
    def preprocess(markdown: str) -> str:
        fixed = markdown
        fixed = BACKSLASH_LT_RE.sub(r'\1&lt;', fixed)
        fixed = BLOCKQUOTE_BR_RE.sub(r'>\1<br />', fixed)
        fixed = WITHIN_BLOCK_ATTR_RE.sub(r' \1', fixed)
        return fixed

    @staticmethod
    def clean(html: str) -> str:
        fixed = html
        fixed = RDQUOTE_FIX_RE.sub(r'\1”', fixed)
        fixed = RDQUOTE_FIX_RE2.sub(r'”\1', fixed)
        fixed = LDQUOTE_FIX_RE.sub(r'“\1', fixed)
        return fixed


@dataclass(frozen=True)
class MarkdownConfig(Mapping[str, Any]):
    extensions: list[Extension | str] = field(default_factory=default_extensions)
    extension_configs: dict[str, dict[str, Any]] = field(
        default_factory=default_ext_configs
    )

    def __iter__(self) -> Iterator[str]:
        return iter(asdict(self))

    def __len__(self) -> int:
        return len(asdict(self))

    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key]

    def get_engine(self) -> MarkdownEngine:
        return MarkdownEngine(**self)

    def make_parser(self) -> MarkdownParser:
        return MarkdownParser(self.get_engine())


HTMLEntityType: TypeAlias = Literal[
    'decl', 'start', 'end', 'startend', 'data', 'comment'
]

@dataclass(frozen=True)
class HTMLEntity:
    type: HTMLEntityType
    value: str
    format: ClassVar[dict[HTMLEntityType, str]] = {
        'decl': '<!{}>',
        'start': '<{}>',
        'end': '</{}>',
        'startend': '<{}/>',
        'data': '{}',
        'comment': '<!-- {} -->',
    }

    def __str__(self) -> str:
        return self.format[self.type].format(self.value)

    @property
    def name(self) -> str | None:
        if self.type in ('start', 'startend'):
            return self.value.partition(' ')[0]
        if self.type == 'end':
            return self.value
        return None

    @classmethod
    def decl(cls, value: str) -> 'HTMLEntity':
        return cls('decl', value)

    @classmethod
    def start(cls, value: str) -> 'HTMLEntity':
        return cls('start', value)

    @classmethod
    def end(cls, value: str) -> 'HTMLEntity':
        return cls('end', value)

    @classmethod
    def startend(cls, value: str) -> 'HTMLEntity':
        return cls('startend', value)

    @classmethod
    def data(cls, value: str) -> 'HTMLEntity':
        return cls('data', value)

    @classmethod
    def comment(cls, value: str) -> 'HTMLEntity':
        return cls('comment', value)


class HTMLCleaner(HTMLParser):
    _result: io.StringIO
    entities: list[HTMLEntity]

    @classmethod
    def clean(cls, html: str) -> str:
        parser = cls()
        parser.feed(html)
        parser.process_tree()
        return str(parser)

    def process_tree(self) -> Self:
        before: list[str] = []
        size = len(self.entities)
        for idx, entity in enumerate(self.entities):
            if entity.type == 'data':
                if 'code' in before:
                    self.write(str(entity))
                    continue

                last = str(self.entities[idx - 1]) if idx > 0 else ''
                nxt = str(self.entities[idx + 1]) if idx < (size - 1) else ''
                snippet = f'{last}{entity}{nxt}'
                clean_text = MarkdownParser.clean(snippet)
                text_only = clean_text[len(last):len(snippet) - len(nxt)]
                self.write(text_only)
                continue

            if entity.type == 'start':
                assert entity.name is not None
                before.append(entity.name)
            elif entity.type == 'end' and entity.name == before[-1]:
                before.pop()

            self.write(str(entity))
        return self

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)

    def __str__(self) -> str:
        return self._result.getvalue()

    def reset(self) -> None:
        super().reset()
        self.entities = []
        self._result = io.StringIO()

    def push(self, entity: HTMLEntity) -> Self:
        if self.entities and (self.entities[-1].type == entity.type == 'data'):
            last = self.pop()
            self.entities.append(
                HTMLEntity.data(last.value + entity.value)
            )
        else:
            self.entities.append(entity)
        return self

    def pop(self) -> HTMLEntity:
        return self.entities.pop()

    def handle_decl(self, decl: str) -> None:
        self.push(HTMLEntity.decl(decl))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        value = [tag]
        for attr, val in attrs:
            value.append(
                f'{attr}="{val}"' if val is not None else attr
            )
        self.push(HTMLEntity.start(' '.join(value)))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        value = [tag]
        for attr, val in attrs:
            value.append(
                f'{attr}="{val}"' if val is not None else attr
            )
        self.push(HTMLEntity.startend(' '.join(value)))

    def handle_endtag(self, tag: str) -> None:
        self.push(HTMLEntity.end(tag))

    def handle_data(self, data: str) -> None:
        self.push(HTMLEntity.data(data))

    def handle_entityref(self, name: str) -> None:
        ref = f'&{name};'
        self.push(HTMLEntity.data(ref))

    def handle_charref(self, name: str) -> None:
        ref = f'&#{name};'
        self.push(HTMLEntity.data(ref))

    def write(self, *data: str) -> Self:
        for chunk in data:
            self._result.write(chunk)
        return self
