"""Handler for templated result files"""

from   collections.abc          import Mapping
from   contextlib               import contextmanager
from   datetime                 import datetime, timedelta, timezone
from   operator                 import itemgetter
from   pathlib                  import Path
import re
from   typing                   import Any, Callable, Iterable, TypeVar

import jinja2
from   jinja2                   import (BaseLoader, Environment,
                                        Template as JinjaTemplate, lexer,
                                        pass_context, select_autoescape)
from   jinja2.ext               import Extension
from   jinja2.lexer             import Lexer, Token, TokenStream
from   jinja2.runtime           import Context
from   jinja2.utils             import Namespace

from   .config                  import Config
from   .markdown                import markdownify
from   .utils                   import ilast, prepend


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
        self.env.globals["includes"] = self.env.globals["namespace"]()
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
        self.env.filters['debug'] = debug
        self.env.filters['limit'] = limit
        self.env.filters['last'] = last
        self.env.filters['namespace_to_dict'] = namespace_to_dict
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
            raise TemplateError(
                f"Invalid template syntax at {exc.filename}:{exc.lineno}",
                str(exc), exc.source,
            ) from exc
        except jinja2.exceptions.TemplateError as exc:
            if isinstance(exc.__cause__, jinja2.exceptions.TemplateSyntaxError):
                orig = exc.__cause__
                raise TemplateError(
                    f"Invalid template syntax at {orig.filename}:{orig.lineno}",
                    str(orig), orig.source
                ) from exc
            else:
                raise TemplateError("Template error", exc.message) from exc

    def render(self, source: str, data: dict[str, object]) -> str:
        try:
            return self.env.from_string(source).render(data)
        except jinja2.exceptions.TemplateSyntaxError as exc:
            raise TemplateError(
                f"Invalid template syntax at {exc.filename}:{exc.lineno}",
                str(exc), exc.source
            ) from exc
        except jinja2.exceptions.TemplateError as exc:
            if isinstance(exc.__cause__, jinja2.exceptions.TemplateSyntaxError):
                orig = exc.__cause__
                raise TemplateError(
                    f"Invalid template syntax at {orig.filename}:{orig.lineno}",
                    str(orig), orig.source
                ) from exc
            else:
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
        token_type, token_value, lineno = None, None, 1
        state: str | None = None
        block_idx = 0
        block_stack: list[list[str]] = []
        ns_vars = set()

        @contextmanager
        def parse_state(new_state):
            nonlocal state
            old_state = state
            state = new_state
            yield
            state = old_state

        sublexer = Sublexer(self.environment, stream.name, stream.filename)
        def tokenize(s: str) -> Iterable[Token]:
            nonlocal lineno, state, debug
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
                print(f'{stream.filename}:{lineno} - Token {token.type!r} {token.value!r}')
            return current(token)

        def current(token: Token) -> Token:
            nonlocal lineno, token_type, token_value
            lineno, token_type, token_value = token.lineno, token.type, token.value
            return token

        def passthrough() -> Token:
            return tok(next(stream))

        if stream.name:
            debug = True
            yield from tokenize('{% set ns = namespace() %}')
        for token in map(current, stream):
            debug = True
            state = None
            if (token.type, token.value) == ("name", "assign"):
                yield tok('set')
                if block_stack:
                    token = stream.expect(lexer.TOKEN_NAME)
                    var = token.value
                    ns_vars.add(var)
                    block_stack[-1].append(var)
                    yield tok(token)
                    had_colon = False
                    while (token := next(stream)).type != lexer.TOKEN_BLOCK_END:
                        if token.type == 'colon':
                            had_colon = True
                            yield tok(lparen='(')
                        elif token.value == 'forloop':
                            yield tok(name='loop')
                        else:
                            yield tok(token)
                    if had_colon:
                        yield tok(rparen=')')
                    yield tok(token)
                    yield from tokenize(f'{{% set ns.{var} = {var} %}}')
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
                print(f'NEW INCLUDE: {token.lineno} ({lineno})')
                debug = True
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
                print(f'next_token: {token.lineno} ({lineno})')
                if next_token.type == 'dot':
                    yield tok(name='includes')
                    yield tok(next_token)
                    yield passthrough()
                    debug = False
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
                    rhs = [t if t.value != 'include' else tok(name='_old_includes')
                           for t in tokens]
                    assignments.append((name, rhs))
                has_nested_include_refs = any(
                    tok for (lhs, rhs) in assignments for tok in rhs
                    if '_old_includes' == tok.value
                )

                # If there are assignments, emit the 'with' block, followed by
                # the assignments
                if assignments:
                    print(f'assignments: {block_end.lineno} ({lineno})')
                    if has_nested_include_refs:
                        yield from tokenize('with _old_includes = includes %}{%')
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
                   if has_nested_include_refs:
                        yield from tokenize('{% endwith %}')
                debug = False
            elif token.type == "dot":
                next_token = next(stream)
                if next_token.value == 'size':
                    yield tok(pipe='|')
                    yield tok(name='size')
                else:
                    yield tok()
                    yield tok(next_token)
            elif (token.type, token.value) == ("name", "for"):
                block_idx += 1
                block = f'for_vars{block_idx}'
                block_stack.append([block])
                with parse_state('block'):
                    yield from tokenize(f'set {block} = namespace() %}}{{%')
                    yield tok()
            elif (token.type, token.value) == ("name", "endfor"):
                block, *vars = block_stack.pop()
                vars = set(vars)
                with parse_state('block'):
                    #for var in ns_vars:
                    #    yield from tokenize(f'set {block}.{var} = {var} %}}{{%')
                    yield tok()
                    yield tok(stream.expect(lexer.TOKEN_BLOCK_END))
                # for var in vars:
                #     yield from tokenize(f'{{% set {var} = {block}.{var} %}}')
                for var in ns_vars:
                    yield from tokenize(f'{{% set {var} = ns.{var} %}}')
            elif (token.type, token.value) == ("name", "capture"):
                yield tok('set')
            elif (token.type, token.value) == ("name", "xml_escape"):
                yield tok('escape')
            elif (token.type, token.value) == ("name", "endcapture"):
                yield tok('endset')
            elif (token.type, token.value) == ("name", "elsif"):
                yield tok('elif')
            elif (token.type, token.value) == ("name", "forloop"):
                yield tok('loop')
            elif (token.type, token.value) == ("name", "unless"):
                args = True
                yield tok('if')
                yield tok('not')
                yield tok(lparen='(')
            elif (token.type, token.value) == ("name", "endunless"):
                yield tok('endif')
            elif (token.type, token.value) == ("name", "limit"):
                colon = next(stream)
                if colon.type != lexer.TOKEN_COLON:
                    yield tok()
                    yield tok(colon)
                    continue
                count = stream.expect(lexer.TOKEN_INTEGER)
                with parse_state('block'):
                    tokenize(f'| limit({count.value})')
            elif token.value == ':':
                args = True
                yield tok(lparen='(')
            elif token.type in {'pipe', 'block_end', 'variable_end'}:
                if args:
                    yield tok(rparen=')')
                    args = False
                yield tok(token)
            elif token.type != 'data':
                yield tok()
            else:
                debug = False
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
        lineno: int=1,
        state: str | None=None,
        debug=False,
    ) -> Iterable[Token]:
        for token in self.lexer.tokenize(source, self.name, self.filename, state):
            if debug:
                print(f'{self.filename}:{token.lineno+lineno-1} - Token {token.type!r} {token.value!r}')
            yield Token(token.lineno + lineno - 1, token.type, token.value)


def slugify(text: str) -> str:
    """Replace bad characters for use in a path"""
    return re.sub(
        '[^a-z0-9-]+', '-',
        text.lower().replace("'", ""),
    ).strip(' -')


def append(base: str | Path, to: str) -> Path | str:
    if isinstance(base, Path):
        return base / str(to)
    return str(base) + str(to)


def date(dt: str | datetime, fmt: str) -> str:
    if dt == 'now':
        dt = datetime.now()
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
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

@pass_context
def where_exp(context: Context, iterable: Iterable[Any], var: str, expression: str) -> Iterable[Any]:
    def gen():
        environment = context.environment
        fixed = contains_re.sub(r'\2 in \1', expression)
        condition = environment.compile_expression(fixed)
        # print(f'where_exp {var}, {expression}')
        for item in iterable:
            try:
                if condition(**{**context.parent, **context.vars, var: item}):
                    yield item
            except TypeError:
                #print(context.get("site").pages)
                #print(context.vars)
                #print(f'where_exp {var}, {expression}')
                #print(item)
                pass
    return [*gen()]


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
    try:
        return len(s.split())
    except:
        return 0


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

T = TypeVar('T')

@pass_context
def debug(context: Context, it: T, *labels: str) -> T:
    name = context.name
    label = ' '.join(map(str, labels))
    print(f'DEBUGGING {name}{f" - {label}" if label else ""}')
    if isinstance(it, jinja2.runtime.Undefined):
        print(f'DEBUG {name}: {label} = Undefined')
        if 'seriesIdx' in label:
            print(f'Looking up ns in the context: {context.resolve("ns")}')
            print(f'Looking up seriesIdx in the context: {context.resolve("seriesIdx")}')
        return it
    elif isinstance(it, Mapping):
        for key, item in dict(**it).items():
            print(f'DEBUG {name}: {label}[{key}] = {item!r}')
        return it
    elif isinstance(it, Iterable) and type(it) not in (str, bytes):
        def log_all() -> T:
            idx = 0
            for idx, item in enumerate(it, start=1):
                print(f'DEBUG {name}: {label}[{idx-1}] = {item!r}')
                yield item
            print(f'DEBUG {name}: len({label}) = {idx}{"" if idx else f" ({it!r})"}')
        return [*log_all()]
    print(f'DEBUG {name}: {label} = {it!r}')
    return it


def limit(it: Iterable[T], count: int) -> Iterable[T]:
    return list(it)[:count]


def last(it: Iterable[T]) -> T:
    try:
        return next(iter(reversed(it)))
    except TypeError:
        return ilast(it)


def namespace_to_dict(ns: Namespace) -> dict[str, Any]:
    return {k: v for (k, v) in ns.items()}
