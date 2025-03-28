from pathlib import Path
from textwrap import dedent
from typing import Any, cast

from jinja2 import Template

from pyde.transformer import Transformer


def test_default_transform() -> None:
    path = Path('path/to/input.ext')
    tf = Transformer(path)

    assert tf.source == path
    assert tf.outputs == path


def test_permalink_transform() -> None:
    path = Path('path/to/input.ext')
    tf = Transformer(path, permalink='/new_root/:path/:basename')

    assert tf.outputs == Path('new_root/path/to/input.ext')


def test_markdown_transform() -> None:
    path = Path('path/to/post.md')
    tf = Transformer(path)

    assert tf.transform(
        'Hello *there,* world'
    ) == '<p>Hello <em>there,</em> world</p>'


def test_template_transform() -> None:
    path = Path('path/to/post.txt')
    template = Template('Hello {{ content }}, {{ page.name }}')
    tf = Transformer(
        path, template=template, name='friend'
    )

    assert tf.transform('world') == 'Hello world, friend'


def test_markdown_template_transform() -> None:
    path = Path('path/to/post.md')
    template = Template('<html><body>{{ content }}</body></html>')
    tf = Transformer(path, template=template)

    assert tf.transform(
        'Hello *there,* world'
    ) == '<html><body><p>Hello <em>there,</em> world</p></body></html>'


def test_markdown_template_pipeline() -> None:
    path = Path('path/to/post.md')
    template = Template('<html><body>{{ content }}</body></html>')
    tf = Transformer(path).pipe(template=template)

    assert tf.transform(
        'Hello *there,* world'
    ) == '<html><body><p>Hello <em>there,</em> world</p></body></html>'


def metaprocessor(source: str | bytes | Path, **meta: Any) -> str:
    if isinstance(source, Path):
        content = source.read_text('utf8')
    elif isinstance(source, bytes):
        content = source.decode('utf8')
    else:
        content = source
    return Template(content).render(meta)


def test_metadata_template() -> None:
    path = Path('path/to/post.template')
    content = dedent(
        '''\
            ---
            title: Some Title
            ---
            # {{ title }}

            Hello, world!
        '''
    )
    tf = Transformer(path, metaprocessor=metaprocessor)

    assert tf.transform(content).rstrip() == dedent(
        '''\
            # Some Title

            Hello, world!
        '''
    ).rstrip()


def test_markdown_with_metadata() -> None:
    path = Path('path/to/post.md')
    content = dedent(
        '''\
            ---
            title: Some Title
            ---
            # {{ title }}

            Hello, world!
        '''
    )
    tf = Transformer(path, metaprocessor=metaprocessor)

    assert tf.transform(content).rstrip() == dedent(
        '''\
            <h1>Some Title</h1>
            <p>Hello, world!</p>
        '''
    ).rstrip()


def test_metadata_on_template() -> None:
    path = Path('path/to/post.md')
    content = dedent(
        '''\
            ---
            title: Some Title
            ---
            Hello, world!
        '''
    )
    template_str = dedent(
        '''\
            <html>
                <body>
                    <h1>{{ page.title }}</h1>
                    {{ content }}
                </body>
            </html>
        '''
    )
    template = Template(template_str)
    tf = Transformer(path, template=template)

    assert tf.transform(content).rstrip() == dedent(
        '''\
            <html>
                <body>
                    <h1>Some Title</h1>
                    <p>Hello, world!</p>
                </body>
            </html>
        '''
    ).rstrip()


class FakePath:
    def __init__(self, path: Path | str, content: str=''):
        self._content = content
        self._path = Path(path)

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        return f'FakePath({str(self._path)!r}, content={self._content!r})'

    def _with_path(self, path: Path) -> 'FakePath':
        return FakePath(path, self._content)

    def __eq__(self, other: object) -> bool:
        return self._path == other

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def stem(self) -> str:
        return self._path.stem

    @property
    def suffix(self) -> str:
        return self._path.suffix

    def with_suffix(self, suffix: str) -> 'FakePath':
        return self._with_path(self._path.with_suffix(suffix))

    def match(self, pattern: str) -> bool:
        return self._path.match(pattern)

    def relative_to(self, other: str | Path) -> 'FakePath':
        return self._with_path(self._path.relative_to(other))

    @property
    def parent(self) -> 'FakePath':
        return self._with_path(self._path.parent)

    @property
    def parents(self) -> list['FakePath']:
        return [*map(FakePath, self._path.parents)]

    def __truediv__(self, other: Any) -> 'FakePath':
        return self._with_path(self._path / other)

    def __rtruediv__(self, other: Any) -> 'FakePath':
        return self._with_path(other / self._path)

    def read_bytes(self) -> bytes:
        return self._content.encode('utf8')

    def read_text(self) -> str:
        return self._content

    def write_bytes(self, data: bytes) -> None:
        self._content = data.decode('utf8')

    def write_text(self, data: str) -> None:
        self._content = data


def test_metadata_joined() -> None:
    path = cast(Path, FakePath(
        'path/to/post.md',
        content=dedent(
            '''\
                ---
                title: Some Title
                ---
                Hello, world!
            '''
        ),
    ))
    template_str = dedent(
        '''\
            <html>
                <body>
                    <h1>{{ title }}</h1>
                    {{ content }}
                </body>
            </html>
        '''
    )
    template = Template(template_str)
    tf = Transformer(path, template=template, permalink='/:path/:basename')
    tf.preprocess(Path('.')).transform_from_file(Path('.'))
    assert tf.metadata == {
        'path': Path('/path/to/post'),
        'file': Path('path/to/post.html'),
        'title': 'Some Title',
        'word_count': 2,
        'excerpt': '<p>Hello, world!</p>',
    }
