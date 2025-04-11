"""Handler for templated result files"""

import re
import sys
from collections import namedtuple
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from contextlib import contextmanager
from datetime import date as date_only
from datetime import datetime
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Any, Literal, Self, TypeVar, cast, overload

import jinja2
from jinja2 import (
    BaseLoader,
    Environment,
    Template,
    lexer,
    pass_context,
    pass_environment,
    select_autoescape,
)
from jinja2.ext import Extension
from jinja2.lexer import Lexer, Token, TokenStream
from jinja2.runtime import Context, Undefined
from jinja2.utils import Namespace

from pyde.markdown.handler import MarkdownParser

from .data import AutoDate
from .path import AnyRealPath, ReadablePath, UrlPath
from .utils import first as ifirst
from .utils import last as ilast
from .utils import prepend, slugify

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class TemplateManager:
    def __init__(
        self, includes_dir: AnyRealPath, templates_dir: AnyRealPath,
        globals: Mapping[str, Any]={},
    ):
        self.env = Environment(
            loader=TemplateLoader(templates_dir, includes_dir),
            autoescape=select_autoescape(
                enabled_extensions=(),
                default_for_string=False,
            ),
            undefined=ChainingUndefined,
            extensions=[JekyllTranslator, 'jinja2.ext.loopcontrols'],
        )
        self.env.globals["includes"] = Namespace()
        self.env.globals.update(globals)
        self.env.extend(
            pyde_includes_dir=includes_dir,
            pyde_templates_dir=templates_dir
        )
        self.env.filters['markdownify'] = markdownify
        self.env.filters['slugify'] = slugify
        self.env.filters['append'] = append
        self.env.filters['date'] = date
        self.env.filters['where'] = where
        self.env.filters['where_exp'] = where_exp
        self.env.filters['sort_natural'] = sort_natural
        self.env.filters['number_of_words'] = number_of_words
        self.env.filters['index'] = index
        self.env.filters['slice'] = real_slice
        self.env.filters['size'] = size
        self.env.filters['debug'] = debug
        self.env.filters['limit'] = limit
        self.env.filters['last'] = last
        self.env.filters['plus'] = plus
        self.env.filters['minus'] = minus
        self.env.filters['divided_by'] = lambda x, y: (x + (y//2)) // y
        self.env.filters['absolute_url'] = absolute_url
        self.env.filters['relative_url'] = relative_url
        self.env.filters['reverse'] = reverse
        self.env.filters['dictmap'] = dictmap

    @property
    def globals(self) -> dict[str, Any]:
        return self.env.globals

    def get_template(self, template: str | ReadablePath, **globals: Any) -> Template:
        return self.env.get_template(str(template), globals=globals)

    def render(
        self, source: str | bytes | ReadablePath, **metadata: Any,
    ) -> str:
        if isinstance(source, (str, bytes)):
            return self.render_string(source, metadata)
        return self.render_template(source, metadata)

    def render_template(
        self, source: ReadablePath, data: dict[str, object],
    ) -> str:
        return self.get_template(source).render(data)

    def render_string(self, source: str | bytes, data: dict[str, object]) -> str:
        source = source.decode('utf8') if isinstance(source, bytes) else source
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
            raise TemplateError("Template error", exc.message) from exc


class TemplateError(ValueError):
    @property
    def message(self) -> str:
        return ' - '.join(self.args)


class TemplateLoader(BaseLoader):
    """Main loader for templates"""
    def __init__(self, templates_dir: AnyRealPath, includes_dir: AnyRealPath):
        self.templates_dir = templates_dir
        self.includes_dir = includes_dir

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str | None, Callable[[], bool] | None]:
        path: AnyRealPath
        if template.startswith(str(self.includes_dir)):
            path = Path(template)
            if not path.is_file():
                include_name = path.relative_to(str(self.includes_dir))
                raise ValueError('Cannot find template {include_name} to include')
        elif (path_by_name := self.templates_dir / template).is_file():
            path = path_by_name
        elif (exact_path := Path(template)).is_file():
            path = exact_path
        else:
            return '{{ content }}', "", lambda: True
        mtime = path.stat().st_mtime
        source = path.read_text('utf8')
        return source, str(path), lambda: mtime == path.stat().st_mtime


UNDEFINED_FIELDS = ('hint', 'obj', 'name', 'exception')


class ChainingUndefined(Undefined):
    __slots__ = ()
    _UndefTuple = namedtuple('_UndefTuple', UNDEFINED_FIELDS)  # type: ignore
    _attrs = attrgetter(*[f'_undefined_{field}' for field in UNDEFINED_FIELDS])

    def _replace(self, **kwargs: str) -> Self:
        fields = self._UndefTuple(*self._attrs(self))
        return self.__class__(*fields._replace(**kwargs))

    def __getattr__(self, attr: str) -> Self:
        return self._replace(name=f'{self._undefined_name}.{attr}')


class JekyllTranslator(Extension):
    tags = {'comment'}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.ExprStmt:
        node = jinja2.nodes.ExprStmt(lineno=next(parser.stream).lineno)
        parser.parse_statements(("name:endcomment",), drop_needle=True)
        node.node = jinja2.nodes.Const.from_untrusted(None)
        return node

    # This function is a nightmare horror. I intend for it to be a temporary
    # shim until I'm fully beyond Jekyll.
    def filter_stream(self, stream: TokenStream) -> Iterable[Token]:  # pylint: disable=all
        debug = False
        args = False
        token_type, token_value, lineno = '', '', 1
        state: str | None = None
        block_idx = 0
        block_stack: list[list[str]] = []
        ns_vars = set()

        name_map = {
            'nil': 'None',
            'true': 'True',
            'false': 'False',
        }
        def transform(token: Token) -> Token:
            if token.type == 'name' and token.value in name_map:
                token._replace(value=name_map[token.value])
            return token

        @contextmanager
        def parse_state(new_state: str | None) -> Iterator[None]:
            nonlocal state
            old_state = state
            state = new_state
            yield
            state = old_state

        sublexer = Sublexer(self.environment, stream.name, stream.filename)
        def tokenize(s: str) -> Iterable[Token]:
            nonlocal lineno, state, debug
            return sublexer.tokenize(s, lineno=lineno, state=state, debug=debug)

        def tok(*args: Any, **kwargs: Any) -> Token:
            nonlocal token_type, token_value, lineno
            if args and isinstance(args[0], Token):
                token = args[0]
            elif args:
                token = Token(lineno, token_type, args[0])
            elif kwargs:
                token = Token(lineno, *next(iter(kwargs.items())))
            else:
                token = Token(lineno, token_type, token_value)
            token = transform(token)
            if debug:
                print(
                    f'{stream.filename}:{lineno}'
                    f'- Token {token.type!r} {token.value!r}'
                )
            return current(token)

        def current(token: Token) -> Token:
            nonlocal lineno, token_type, token_value
            lineno, token_type, token_value = token.lineno, token.type, token.value
            return token

        def passthrough() -> Token:
            return tok(next(stream))

        if stream.name:
            yield from tokenize('{% set ns = namespace() %}')
        for token in map(current, stream):
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
                    while (token := current(next(stream))).type != lexer.TOKEN_BLOCK_END:
                        if token.type == 'colon':
                            had_colon = True
                            yield tok(lparen='(')
                        elif token.value == 'forloop':
                            yield tok(name='loop')
                        elif token.value == 'limit':
                            colon = next(stream)
                            if colon.type != lexer.TOKEN_COLON:
                                yield tok()
                                yield tok(colon)
                            count = current(stream.expect(lexer.TOKEN_INTEGER))
                            with parse_state('block'):
                                yield from tokenize(f'| limit({count.value})')
                        else:
                            yield tok()
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
                    yield tok()
                    yield tok(stream.expect(lexer.TOKEN_BLOCK_END))
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
                    yield from tokenize(f'| limit({count.value})')
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
        debug: bool=False,
    ) -> Iterable[Token]:
        for token in self.lexer.tokenize(source, self.name, self.filename, state):
            if debug:
                print(
                    f'{self.filename}:{token.lineno+lineno-1}'
                    f'- Token {token.type!r} {token.value!r}'
                )
            yield Token(token.lineno + lineno - 1, token.type, token.value)


def append(base: str | Path, to: str) -> Path | str:
    if isinstance(base, Path):
        return base / str(to)
    return str(base) + str(to)


def get_date(dt: str | date_only | datetime) -> AutoDate:
    try:
        dt = cast(date_only, dt)
        return AutoDate(dt.isoformat())
    except AttributeError:
        return AutoDate(str(dt))


def date(dt: str | datetime | date_only, fmt: str='auto') -> str:
    try:
        if fmt == 'auto':
            return str(get_date(dt))
        return get_date(dt).datetime.strftime(fmt)
    except TypeError:
        return cast(str, dt)


def size(it: object | None) -> int:
    try:
        return len(cast(Sized, it))
    except TypeError:
        return 0


NAME_PATTERN = r'[\w.]+'
STRING_PATTERN = r'"[^"]*"' + "|" + r"'[^']*'"
ARG_PATTERN = f'{NAME_PATTERN}|{STRING_PATTERN}'

contains_re = re.compile(f'({ARG_PATTERN}) contains ({ARG_PATTERN})')

def where(iterable: Iterable[T], key: str, compare: str) -> Iterable[T]:
    iterable = cast(Mapping[Any, Any], iterable)
    return (item for item in iterable if str(item[key]) == str(compare))


@pass_context
def where_exp(
    context: Context, iterable: Iterable[T], var: str, expression: str,
) -> Iterable[T]:
    python_defs = {'nil': None, 'true': True, 'false': False}
    template_vars = {**context.parent, **context.vars, **python_defs}
    def gen() -> Iterable[T]:
        environment = context.environment
        fixed = contains_re.sub(r'\2 in \1', expression)
        condition = environment.compile_expression(fixed)
        for item in iterable:
            try:
                if condition(**{**template_vars, var: item}):
                    yield item
            except TypeError:
                pass
    return [*gen()]


def sort_natural(iterable: Iterable[Any], sort_type: str) -> Iterable[Any]:
    key: Callable[[Any], Any]
    if sort_type == 'date':
        key = lambda x: get_date(x.date)
    else:
        key = itemgetter(sort_type)
    sorted_it = sorted(iterable, key=key)
    return sorted_it


def number_of_words(s: str) -> int:
    try:
        return len(s.split())
    except (AttributeError, TypeError):
        return 0


def real_slice(iterable: Iterable[T], offset: int, limit: int) -> Sequence[T]:
    try:
        iterable = cast(Sequence[T], iterable)
        return iterable[offset:offset+limit]
    except TypeError:
        return list(iterable)[offset:offset+limit]

def index(iterable: Iterable[T], idx: int) -> T:
    try:
        iterable = cast(Sequence[T], iterable)
        return iterable[idx]
    except TypeError:
        return list(iterable)[idx]


@overload
def debug(context: Context, it: Undefined, *labels: str) -> Undefined: ...
@overload
def debug(context: Context, it: Iterable[T], *labels: str) -> Iterable[T]: ...
@pass_context
def debug(context: Context, it: T, *labels: str) -> T | Undefined | Iterable[T]:
    name = context.name
    label = ' '.join(map(str, labels))
    print(f'DEBUGGING {name}{f" - {label}" if label else ""}')
    if isinstance(it, jinja2.runtime.Undefined):
        print(f'DEBUG {name}: {label} = Undefined')
    elif isinstance(it, Mapping):
        for key, item in {**it}.items():
            print(f'DEBUG {name}: {label}[{key}] = {item!r}')
    elif isinstance(it, Iterable) and type(it) not in (str, bytes):
        def log_all() -> Iterable[T]:
            idx = 0
            for idx, item in enumerate(it, start=1):
                print(f'DEBUG {name}: {label}[{idx-1}] = {item!r}')
                yield item
            print(f'DEBUG {name}: len({label}) = {idx}{"" if idx else f" ({it!r})"}')
        return [*log_all()]
    else:
        print(f'DEBUG {name}: {label} = {it!r}')
    return it


def limit(it: Iterable[T], count: int) -> Iterable[T]:
    return list(it)[:count]


@pass_environment
def first(environment: Environment, it: Iterable[T]) -> T | Undefined:
    try:
        return ifirst(it)
    except ValueError:
        return environment.undefined('No first item, iterable was empty')


@pass_environment
def last(environment: Environment, it: Iterable[T]) -> T | Undefined:
    try:
        return ilast(it)
    except ValueError:
        return environment.undefined('No last item, iterable was empty')


def plus(x: int | Undefined, y: int | Undefined) -> int | Undefined:
    if isinstance(x, Undefined):
        return x
    if isinstance(y, Undefined):
        return y
    return int(x) + int(y)


def minus(x: int | Undefined, y: int | Undefined) -> int | Undefined:
    if isinstance(x, Undefined):
        return x
    if isinstance(y, Undefined):
        return y
    return int(x) - int(y)


def default(x: T | Undefined, default: T) -> T:
    if not isinstance(x, Undefined):
        return x
    return default


@pass_context
def absolute_url(context: Context, path: str) -> UrlPath:
    url = context.resolve('page').url
    if not url:
        print(f'Unable to resolve url in context: {context.name}', file=sys.stderr)
        return context.environment.undefined('url')
    page_url = UrlPath(str(url))
    try:
        path_url = page_url >> path
    except Exception as exc:
        raise Exception(
            f'Error in {url} - page_url = {page_url} - path = {path}'
        ) from exc
    return path_url.absolute()


@pass_context
def relative_url(context: Context, path: str) -> str:
    url = context.resolve('page').url
    if not url:
        print(f'Unable to resolve url in context: {context.name}', file=sys.stderr)
        return path
    page_url = UrlPath(str(url))
    path_url = page_url >> path
    return path_url.relative_to(page_url).path.lstrip('/')


def reverse(it: Iterable[T]) -> Iterable[T]:
    return reversed(list(it))

@pass_context
def dictmap(
    context: Context,
    it: Mapping[K, V],
    filter: str,
    by: Literal['key',  'value'],
    *args: Any, **kwargs: Any,
) -> Mapping[K, Any] | Mapping[Any, V]:
    def func(item: Any) -> Any:
        return context.environment.call_filter(
            filter, item, args, kwargs, context=context
        )

    if by == 'key':
        return {func(k): it[k] for k in it}
    return {k: func(it[k]) for k in it}


@pass_context
def markdownify(context: Context, md: str) -> str:
    parser = cast(
        MarkdownParser,
        context.resolve('pyde').env.markdown_parser
    )
    return parser.parse(md)
