"""Handler for templated result files"""

from   datetime                 import datetime
from   pathlib                  import Path
import re
from   typing                   import Callable, Iterable

import jinja2
from   jinja2                   import (BaseLoader, Environment,
                                        Template as JinjaTemplate,
                                        select_autoescape)
from   jinja2.ext               import Extension
from   jinja2.lexer             import Token, TokenStream

from   .config                  import Config
from   .markdown                import markdownify


class Template:
    def __init__(self, templates_dir: Path=Path('_layouts')):
        self.env = Environment(
            loader=TemplateLoader(templates_dir),
            autoescape=select_autoescape(
                enabled_extensions=(),
                default_for_string=False,
            ),
            extensions=[JekyllTranslator, 'jinja2.ext.loopcontrols'],
        )
        self.env.filters['markdownify'] = markdownify
        self.env.filters['slugify'] = slugify
        self.env.filters['append'] = append
        self.env.filters['date'] = date
        self.env.filters['size'] = size
        self.env.filters['plus'] = lambda x, y: x + y
        self.env.filters['divided_by'] = lambda x, y: (x + (y//2)) // y
        self.env.filters['absolute_url'] = lambda x: x
        self.env.filters['relative_url'] = lambda x: x

    @classmethod
    def from_config(cls, config: Config) -> 'Template':
        return cls(config.layouts_dir)

    def get_template(self, name: str) -> JinjaTemplate:
        return self.env.get_template(name)

    def apply(self, template: str, data: dict[str, object]) -> str:
        try:
            return self.get_template(template).render(data)
        except jinja2.exceptions.TemplateSyntaxError as exc:
            raise TemplateError("Invalid template syntax", exc.message) from exc
        except jinja2.exceptions.TemplateError as exc:
            raise TemplateError("Template error", exc.message) from exc

    def render(self, source: str, data: dict[str, object]) -> str:
        try:
            return JinjaTemplate(source).render(data)
        except jinja2.exceptions.TemplateSyntaxError as exc:
            raise TemplateError("Invalid template syntax", exc.message) from exc
        except jinja2.exceptions.TemplateError as exc:
            raise TemplateError("Template error", exc.message) from exc


class TemplateError(ValueError):
    @property
    def message(self) -> str:
        return ' - '.join(self.args)


class TemplateLoader(BaseLoader):
    """Main loader for templates"""
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str | None, Callable[[], bool] | None]:
        if template.startswith('_includes'):
            path = Path(template)
        else:
            path = self.templates_dir / template
        if path.is_file():
            mtime = path.stat().st_mtime
            source = path.read_text()
            return source, str(path), lambda: mtime == path.stat().st_mtime
        return '{{ content }}', "", lambda: True


class JekyllTranslator(Extension):
    tags = {'comment'}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.ExprStmt:
        node = jinja2.nodes.ExprStmt(lineno=next(parser.stream).lineno)
        parser.parse_statements(("name:endcomment",), drop_needle=True)
        node.node = jinja2.nodes.Const.from_untrusted(None)
        return node

    def preprocess(self, source: str, name: str | None=None, filename=None):
        return re.sub(r'\bfalse\b', 'False', re.sub(r'\btrue\b', 'True', source))

    def filter_stream(self, stream: TokenStream) -> TokenStream | Iterable[Token]:
        args = False
        for token in stream:
            if (token.type, token.value) == ("name", "assign"):
                yield Token(token.lineno, token.type, "set")
            elif (token.type, token.value) == ("name", "number_of_words"):
                yield Token(token.lineno, token.type, "wordcount")
            elif (token.type, token.value) == ("name", "strip_html"):
                yield Token(token.lineno, token.type, "striptags")
            elif (token.type, token.value) == ("name", "include"):
                yield Token(token.lineno, token.type, token.value)
                path = '_includes/'
                while (next_token := next(stream)).type != 'block_end':
                    path += next_token.value
                yield Token(token.lineno, 'string', path)
                yield Token(next_token.lineno, next_token.type, next_token.value)
            elif token.type == "dot":
                next_token = next(stream)
                if next_token.value == 'size':
                    yield Token(token.lineno, 'pipe', '|')
                    yield Token(token.lineno, 'name', 'size')
                else:
                    yield token
                    yield next_token
            elif (token.type, token.value) == ("name", "capture"):
                yield Token(token.lineno, token.type, "set")
            elif (token.type, token.value) == ("name", "endcapture"):
                yield Token(token.lineno, token.type, "endset")
            elif (token.type, token.value) == ("name", "unless"):
                args = True
                yield Token(token.lineno, token.type, "if")
                yield Token(token.lineno, token.type, "not")
                yield Token(token.lineno, 'lparen', "(")
            elif (token.type, token.value) == ("name", "endunless"):
                yield Token(token.lineno, token.type, "endif")
            elif token.value == ':':
                args = True
                yield Token(token.lineno, 'lparen', '(')
            elif token.type in {'pipe', 'block_end', 'variable_end'}:
                if args:
                    yield Token(token.lineno, 'rparen', ')')
                    args = False
                yield token
            else:
                yield token


def slugify(text: str) -> str:
    """Replace bad characters for use in a path"""
    return re.sub('[^a-z0-9-]+', '-', text.lower().replace("'", ""))


def append(base: str | Path, to: str) -> Path | str:
    if isinstance(base, Path):
        return base / to
    return base + to


def date(dt: str | datetime, fmt: str) -> str:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(datestr)
    return dt.strftime(fmt)


def size(it: object | None) -> int:
    try:
        return len(it)
    except TypeError:
        return 0
