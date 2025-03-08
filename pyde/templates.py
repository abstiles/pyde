"""Handler for templated result files"""

from   pathlib                  import Path
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
        self.env.filters['append'] = lambda x, y: x + y
        self.env.filters['plus'] = lambda x, y: x + y
        self.env.filters['divided_by'] = lambda x, y: x / y
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
        except jinja2.exceptions.TemplateError as exc:
            raise TemplateError("Invalid template syntax", exc.message) from exc

    def render(self, source: str, data: dict[str, object]) -> str:
        try:
            return JinjaTemplate(source).render(data)
        except jinja2.exceptions.TemplateError as exc:
            raise TemplateError("Invalid template syntax", exc.message) from exc


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
        path = self.templates_dir / template
        if path.is_file():
            mtime = path.stat().st_mtime
            source = path.read_text()
            return source, str(path), lambda: mtime == path.stat().st_mtime
        return '{{ content }}', "", lambda: True


class JekyllTranslator(Extension):
    tags = {'comment'}

    def parse(self, parser: jinja2.parser.Parser):
        lineno = next(parser.stream).lineno
        parser.parse_statements(("name:endcomment",), drop_needle=True)
        node = jinja2.nodes.ExprStmt(lineno=lineno)
        node.node = jinja2.nodes.Const.from_untrusted(None)
        return node

    def filter_stream(self, stream: TokenStream) -> TokenStream | Iterable[Token]:
        args = False
        for token in stream:
            if (token.type, token.value) == ("name", "assign"):
                print(token.type, "set")
                yield Token(token.lineno, token.type, "set")
            elif (token.type, token.value) == ("name", "number_of_words"):
                print(token.type, "wordcount")
                yield Token(token.lineno, token.type, "wordcount")
            elif (token.type, token.value) == ("name", "strip_html"):
                print(token.type, "striptags")
                yield Token(token.lineno, token.type, "striptags")
            elif (token.type, token.value) == ("name", "capture"):
                print(token.type, "set")
                yield Token(token.lineno, token.type, "set")
            elif (token.type, token.value) == ("name", "endcapture"):
                print(token.type, "set")
                yield Token(token.lineno, token.type, "endset")
            elif (token.type, token.value) == ("name", "unless"):
                args = True
                print(token.type, "if")
                yield Token(token.lineno, token.type, "if")
                print(token.type, "not")
                yield Token(token.lineno, token.type, "not")
                print('lparen', "(")
                yield Token(token.lineno, 'lparen', "(")
            elif (token.type, token.value) == ("name", "endunless"):
                print(token.type, "endif")
                yield Token(token.lineno, token.type, "endif")
            elif token.value == ':':
                args = True
                print('lparen', '(')
                yield Token(token.lineno, 'lparen', '(')
            elif token.type in {'pipe', 'block_end', 'variable_end'}:
                if args:
                    print('rparen', ')')
                    yield Token(token.lineno, 'rparen', ')')
                    args = False
                print(token.type, token.value)
                yield token
            else:
                print(token.type, token.value)
                yield token


def slugify(text: str) -> str:
    """Replace bad characters for use in a path"""
    return re.sub('[^a-z0-9-]+', '-', text.lower().replace("'", ""))
