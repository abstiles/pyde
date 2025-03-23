from pathlib import Path
from textwrap import dedent
from typing import Any

from jinja2 import Template

from pyde.transformer import Transformer


def test_default_transform() -> None:
    path = Path('path/to/input.ext')
    tf = Transformer(path)

    assert tf.source == path
    assert tf.outputs == path


def test_permalink_transform() -> None:
    path = Path('path/to/input.ext')
    tf = Transformer(path, permalink='/new_root/:path/:name')

    assert tf.outputs == Path('new_root/path/to/input.ext')


def test_markdown_transform() -> None:
    path = Path('path/to/post.md')
    tf = Transformer(path, permalink='/new_root/:path/:name')

    assert tf.outputs == Path('new_root/path/to/post.html')

    assert tf.transform(
        'Hello *there,* world'
    ) == '<p>Hello <em>there,</em> world</p>'


def test_template_transform() -> None:
    path = Path('path/to/post.txt')
    template = Template('Hello {{ content }}, {{ name }}')
    tf = Transformer(
        path, permalink='/new_root/:path/:name',
        template=template, name='friend'
    )

    assert tf.outputs == Path('new_root/path/to/post.txt')

    assert tf.transform('world') == 'Hello world, friend'


def test_markdown_template_transform() -> None:
    path = Path('path/to/post.md')
    template = Template('<html><body>{{ content }}</body></html>')
    tf = Transformer(path, permalink='/new_root/:path/:name', template=template)

    assert tf.outputs == Path('new_root/path/to/post.html')

    assert tf.transform(
        'Hello *there,* world'
    ) == '<html><body><p>Hello <em>there,</em> world</p></body></html>'


def test_markdown_template_pipeline() -> None:
    path = Path('path/to/post.md')
    template = Template('<html><body>{{ content }}</body></html>')
    tf = (
        Transformer(path)
            .pipe(template=template)
            .pipe(permalink='/new_root/:path/:name')
    )

    assert tf.outputs == Path('new_root/path/to/post.html')

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
                    <h1>{{ title }}</h1>
                    {{ content }}
                </body>
            </html>
        '''
    )
    template = Template(template_str)
    tf = Transformer(path, metaprocessor=metaprocessor, template=template)

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
