"""Handler for templated result files"""

from   contextlib               import contextmanager
from   datetime                 import datetime, timedelta, timezone
from   operator                 import itemgetter
from   pathlib                  import Path
import re
from   typing                   import Any, Callable, Iterable

import jinja2
from   jinja2                   import (BaseLoader, Environment,
                                        Template as JinjaTemplate, lexer,
                                        select_autoescape)
from   jinja2.ext               import Extension
from   jinja2.lexer             import Lexer, Token, TokenStream

from   .config                  import Config
from   .markdown                import markdownify
from   .utils                   import prepend


class Template:
    def __init__(self, url: str, includes_dir: Path, templates_dir: Path):
        self.env = Environment(
            loader=TemplateLoader(templates_dir),
            autoescape=select_autoescape(
                enabled_extensions=(),
                default_for_string=False,
            ),
            extensions=[JekyllTranslator, 'jinja2.ext.loopcontrols'],
        )
        self.env.extend(pyde_includes_dir=includes_dir, pyde_templates_dir=templates_dir)
        self.env.filters['markdownify'] = markdownify
        self.env.filters['slugify'] = slugify
        self.env.filters['append'] = append
        self.env.filters['date'] = date
        self.env.filters['where_exp'] = where_exp
        self.env.filters['sort_natural'] = sort_natural
        self.env.filters['number_of_words'] = number_of_words
        self.env.filters['index'] = index
        self.env.filters['slice'] = real_slice
        self.env.filters['size'] = size
        self.env.filters['plus'] = lambda x, y: int(x) + int(y)
        self.env.filters['minus'] = lambda x, y: int(x) - int(y)
        self.env.filters['divided_by'] = lambda x, y: (x + (y//2)) // y
        self.env.filters['absolute_url'] = lambda x: url.rstrip('/') + '/' + x.lstrip('/')
        self.env.filters['relative_url'] = lambda x: x

    @classmethod
    def from_config(cls, config: Config) -> 'Template':
        return cls(config.url, config.includes_dir, config.layouts_dir)

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
            return self.env.from_string(source).render(data)
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
        elif template.startswith('_includes'):
            return '', '', lambda: True
        return '{{ content }}', "", lambda: True


class JekyllTranslator(Extension):
    tags = {'comment'}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.ExprStmt:
        node = jinja2.nodes.ExprStmt(lineno=next(parser.stream).lineno)
        parser.parse_statements(("name:endcomment",), drop_needle=True)
        node.node = jinja2.nodes.Const.from_untrusted(None)
        return node

    def preprocess(self, source: str, name: str | None=None, filename=None):
        return (
            re.sub(r'\bnil\b', 'None',
            re.sub(r'\bfalse\b', 'False',
            re.sub(r'\btrue\b', 'True',
            source
        ))))

    def filter_stream(self, stream: TokenStream) -> TokenStream | Iterable[Token]:
        debug = False
        args = False
        token_type, token_value, lineno = None, None, 0
        state: str | None = None

        @contextmanager
        def parse_state(new_state):
            nonlocal state
            old_state = state
            state = new_state
            yield
            state = old_state

        sublexer = Sublexer(self.environment, stream.name, stream.filename)
        def tokenize(s: str) -> Iterable[Token]:
            return sublexer.tokenize(s, lineno=lineno, state=state, debug=debug)

        def tok(*args, **kwargs: Any) -> Token:
            nonlocal token_type, token_value, lineno
            if args and isinstance(args[0], Token):
                token = args[0]
            elif args:
                token = Token(lineno, token_type, args[0])
            elif kwargs:
                token = Token(lineno, *next(iter(kwargs.items())))
            else:
                token = Token(lineno, token_type, token_value)
            if debug:
                print(f'Token: {token.type!r} {token.value!r}')
            return current(token)

        def current(token: Token) -> Token:
            nonlocal lineno, token_type, token_value
            lineno, token_type, token_value = token.lineno, token.type, token.value
            return token

        def passthrough() -> Token:
            return tok(next(stream))

        for token in map(current, stream):
            state = None
            if (token.type, token.value) == ("name", "assign"):
                yield tok('set')
            elif (token.type, token.value) == ("name", "strip_html"):
                yield tok('striptags')
            elif (token.type, token.value) == ("name", "forloop"):
                yield tok('loop')
            elif (token.type, token.value) == ("name", "where"):
                args = True
                yield tok('selectattr')
                next(stream) # colon
                yield tok(lparen='(')
                yield passthrough()
                yield tok(comma=',')
                yield tok(string='equalto')
            elif (token.type, token.value) == ("name", "include"):
                # Transform an include statement with arguments into a
                # with/include pair. Given:
                #     {% include f.html a=b x=y %}
                # Then emit:
                #     {% with includes = namespace() %}
                #     {% set includes.a = b %}
                #     {% set includes.x = y %}
                #     {% include "f.html" %}
                #     {% endwith %}
                # Also look for references to variables on "include"
                # and transform them to match. Given:
                #     {{ include.page }}
                # Then emit:
                #     {{ includes.page }}

                # First check if this is a namespace reference.
                next_token = next(stream)
                if next_token.type == 'dot':
                    yield tok(name='includes')
                    yield tok(next_token)
                    yield passthrough()
                    continue

                # We know this is an include statement. Replace the token we
                # just checked and set the state to "block".
                state = 'block'
                rest = iter(prepend(next_token, stream))

                # Two names in a row or a name after a literal indicate a new
                # argument. Split on that.
                rest_tokens: list[list[Token]] = []
                last_type = None
                while (next_token := next(rest)).type != 'block_end':
                    if (not rest_tokens or (
                            next_token.type == 'name'
                            and last_type in ('name', 'string', 'integer', 'float')
                    )):
                        rest_tokens.append([next_token])
                    else:
                        rest_tokens[-1].append(next_token)
                    last_type = next_token.type
                block_end = next_token

                # First argument to include is the include name.
                include = ''.join(str(t.value) for t in rest_tokens[0])

                # Capture the assignments and put them on 'includes'.
                assignments: list[tuple[str, list[Token]]] = []
                for arg in rest_tokens[1:]:
                    tokens = iter(arg)
                    name = ''
                    while (next_token := next(tokens)).type != lexer.TOKEN_ASSIGN:
                        name += next_token.value
                    rhs = [t if t.value != 'include' else tok(name='includes')
                           for t in tokens]
                    assignments.append((name, rhs))

                # If there are assignments, emit the 'with' block, followed by
                # the assignments
                if assignments:
                    yield from tokenize('with includes = namespace() %}')
                    with parse_state(None):
                        for (name, rhs) in assignments:
                            yield from tokenize(f'{{% set includes.{name} = ')
                            yield from map(tok, rhs)
                            yield tok(block_end='%}')
                    yield tok(block_begin='{%')

                # Emit the include statement
                path = self.environment.pyde_includes_dir / include
                yield from tokenize(f'include "{path}" %}}')
                state = None
                if assignments:
                    yield from tokenize('{% endwith %}')
            elif token.type == "dot":
                next_token = next(stream)
                if next_token.value == 'size':
                    yield tok(pipe='|')
                    yield tok(name='size')
                else:
                    yield tok()
                    yield tok(next_token)
            elif (token.type, token.value) == ("name", "capture"):
                yield tok('set')
            elif (token.type, token.value) == ("name", "endcapture"):
                yield tok('endset')
            elif (token.type, token.value) == ("name", "elsif"):
                yield tok('elif')
            elif (token.type, token.value) == ("name", "unless"):
                args = True
                yield tok('if')
                yield tok('not')
                yield tok(lparen='(')
            elif (token.type, token.value) == ("name", "endunless"):
                yield tok('endif')
            elif token.value == ':':
                args = True
                yield tok(lparen='(')
            elif token.type in {'pipe', 'block_end', 'variable_end'}:
                if args:
                    yield tok(rparen=')')
                    args = False
                yield tok(token)
            else:
                yield tok()


class Sublexer:
    def __init__(
            self,
            environment: Environment,
            name: str | None=None,
            filename: str | None=None
    ):
        self.lexer = Lexer(environment)
        self.name = name
        self.filename = filename

    def tokenize(
        self,
        source: str,
        lineno: int=0,
        state: str | None=None,
        debug=False,
    ) -> Iterable[Token]:
        for token in self.lexer.tokenize(source, self.name, self.filename, state):
            if debug:
                print(f'Token: {token.type!r} {token.value!r}')
            yield Token(token.lineno + lineno, token.type, token.value)


def slugify(text: str) -> str:
    """Replace bad characters for use in a path"""
    return re.sub(
        '[^a-z0-9-]+', '-',
        text.lower().replace("'", ""),
    ).strip(' -')


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


name_pattern = r'[\w.]+'
string_pattern = r'"[^"]*"' "|" r"'[^']*'"
arg_pattern = f'{name_pattern}|{string_pattern}'

contains_re = re.compile(f'({arg_pattern}) contains ({arg_pattern})')

def where_exp(iterable: Iterable[Any], var: str, expression: str) -> Iterable[Any]:
    fixed = contains_re.sub(r'\2 in \1', expression)
    condition = eval(f'lambda {var}: {fixed}')
    return [*filter(condition, iterable)]


def sort_natural(iterable: Iterable[Any], sort_type: str) -> Iterable[Any]:
    def get_date(it: Any):
        if isinstance(it.date, datetime):
            return it.date
        return (
            datetime(*it.date.timetuple()[:3])
            .replace(hour=12, tzinfo=timezone(timedelta(hours=-6)))
        )
    if sort_type == 'date':
        key = get_date
    else:
        key = itemgetter(sort_type)
    return sorted(iterable, key=key)


def number_of_words(s: str):
    return len(s.split())


def real_slice(iterable: Iterable[Any], offset: int, limit: int) -> Any:
    try:
        return iterable[offset:offset+limit]
    except TypeError:
        return list(iterable)[offset:offset+limit]

def index(iterable: Iterable[Any], idx: int) -> Any:
    try:
        return iterable[idx]
    except TypeError:
        return list(iterable)[idx]
