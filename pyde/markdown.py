"""Handler for parsing Markdown"""

import re
from collections.abc import Mapping
from   dataclasses              import asdict, dataclass, field
from typing import Any, ClassVar
import sys

from markdown import Markdown as Parser, Extension
import yaml

from .smartier import SmartyExtension


def _extensions() -> list[Extension | str]:
    return ['extra', 'smarty', 'sane_lists']


def _ext_configs() -> dict[str, dict[str, Any]]:
    return {
        'smarty': {
            'substitutions': {
                'left-single-quote': '‘', # sb is not a typo!
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
    r'--'               # hyphens
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

@dataclass
class Markdown:
    """Markdown parser"""
    markdown: str
    html: str = field(init=False)
    parser: ClassVar[Parser]

    def __post_init__(self) -> None:
        self.html = self.clean_wrong_quotes(
            self.get_parser().convert(self.markdown)
        )

    @classmethod
    def get_parser(cls, config: MarkdownConfig=MarkdownConfig()):
        try:
            return cls.parser.reset()
        except AttributeError:
            cls.parser = Parser(**config)
        return cls.parser

    @staticmethod
    def clean_wrong_quotes(html: str):
        fixed = html
        fixed = RDQUOTE_FIX_RE.sub(r'\1”', fixed)
        fixed = RDQUOTE_FIX_RE2.sub(r'”\1', fixed)
        fixed = LDQUOTE_FIX_RE.sub(r'“\1', fixed)
        return fixed


def markdownify(markdown: str) -> str:
    return Markdown(markdown).html
