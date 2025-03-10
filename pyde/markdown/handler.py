"""Handler for parsing Markdown"""

import re
from collections.abc import Mapping
from   dataclasses              import asdict, dataclass, field
from typing import Any, ClassVar
import sys

from markdown import Extension, Markdown as Parser
import yaml

from .extensions.pm_attr_list import PMAttrListExtension
from .extensions.blockquote import BlockQuoteExtension


def _extensions() -> list[Extension | str]:
    return ['md_in_html', 'smarty', 'sane_lists', PMAttrListExtension(), BlockQuoteExtension()]


def _ext_configs() -> dict[str, dict[str, Any]]:
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


@dataclass(frozen=True)
class MarkdownConfig(Mapping[str, Any]):
    extensions: list[Extension | str] = field(default_factory=_extensions)
    extension_configs: dict[str, dict[str, Any]] = field(default_factory=_ext_configs)

    def __iter__(self):
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))

    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key]


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
    r"|&apos;|&lsquo;|&rsquo;"
    r"|&#39;|&#8216;|&#8217;"
    r")"
)

DOUBLE_QUOTE_PATTERN = (
    r'(?:"'
    r'|“|”'
    r'|&quot;|&ldquo;|&rdquo;'
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

BLOCKQUOTE_BR_RE = re.compile(r'^> \\$', flags=re.MULTILINE)
WITHIN_BLOCK_ATTR_RE = re.compile(
    r'\s*\n> (\{:?[^}]*\w[^}]*\})\s*$',
    flags=re.MULTILINE
)

@dataclass
class Markdown:
    """Markdown parser"""
    markdown: str
    html: str = field(init=False)
    parser: ClassVar[Parser]

    def __post_init__(self) -> None:
        self.html = self.clean(
            self.get_parser().convert(self.preprocess(self.markdown))
        )

    @classmethod
    def get_parser(cls, config: MarkdownConfig=MarkdownConfig()):
        try:
            return cls.parser.reset()
        except AttributeError:
            cls.parser = Parser(**config)
        return cls.parser

    @staticmethod
    def preprocess(markdown: str) -> str:
        fixed = markdown
        fixed = re.sub(BLOCKQUOTE_BR_RE, '> <br />', fixed)
        fixed = re.sub(WITHIN_BLOCK_ATTR_RE, r' \1', fixed)
        return fixed

    @staticmethod
    def clean(html: str) -> str:
        fixed = html
        fixed = RDQUOTE_FIX_RE.sub(r'\1”', fixed)
        fixed = RDQUOTE_FIX_RE2.sub(r'”\1', fixed)
        fixed = LDQUOTE_FIX_RE.sub(r'“\1', fixed)
        return fixed


def markdownify(markdown: str) -> str:
    return Markdown(markdown).html
