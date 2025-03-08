"""Handler for parsing Markdown"""

import sys
from dataclasses import dataclass, field

from markdown_it import MarkdownIt
from markdown_it.token import Token
from markdown_it.utils import EnvType
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
import yaml


@dataclass
class Markdown:
    """Markdown parser"""
    markdown: str
    html: str = field(init=False)
    meta: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        md = MarkdownHandler()
        self.html = md.parse(self.markdown)
        self.meta = md.meta


class MarkdownHandler:
    """Handles configuration and rendering of markdown to HTML"""
    def __init__(self) -> None:
        self.meta: dict[str, object] = {}
        self.md = (
            MarkdownIt(
                'commonmark', {'typographer': True},
            )
                .use(front_matter_plugin)
                .use(footnote_plugin)
                .enable(['replacements', 'smartquotes'])
        )
        self.md.add_render_rule('front_matter', self._front_matter)

    def _front_matter(
        self, tokens: list[Token], idx: int, _options: object, _env: object
    ) -> str:
        """Saves the front matter to meta"""
        front_matter = tokens[idx].content
        try:
            self.meta = yaml.load(front_matter, yaml.Loader)
        except yaml.YAMLError as exc:
            if hasattr(exc, 'problem_mark'):
                line = exc.problem_mark.line
                column = exc.problem_mark.column
                print(
                    f'Error at line {line + 1} column {column + 1}:',
                    file=sys.stderr
                )
                try:
                    err_line = front_matter.split('\n')[line]
                    print(err_line, file=sys.stderr)
                    print(' ' * column + '^', file=sys.stderr)
                except IndexError:
                    pass
        return ''

    def parse(self, markdown: str, env: None | EnvType=None) -> str:
        """Parse Markdown to HTML"""
        return str(self.md.render(markdown, env))


def markdownify(markdown: str) -> str:
    return Markdown(markdown).html
