from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, cast

from jinja2 import Template

from pyde.path.filepath import ReadablePath
from pyde.path.url import UrlPath
from pyde.transformer import Transformer
from pyde.path import AnySource, LocalPath, VirtualPath


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

    assert tf.transform_data(
        'Hello *there,* world'
    ) == '<p>Hello <em>there,</em> world</p>'


def test_template_transform() -> None:
    path = Path('path/to/post.txt')
    template = Template('Hello {{ content }}, {{ page.name }}')
    tf = Transformer(
        path, template=template, name='friend'
    )

    assert tf.transform_data('world') == 'Hello world, friend'


def test_markdown_template_transform() -> None:
    path = Path('path/to/post.md')
    template = Template('<html><body>{{ content }}</body></html>')
    tf = Transformer(path, template=template)

    assert tf.transform_data(
        'Hello *there,* world'
    ) == '<html><body><p>Hello <em>there,</em> world</p></body></html>'


def test_markdown_template_pipeline() -> None:
    path = Path('path/to/post.md')
    template = Template('<html><body>{{ content }}</body></html>')
    tf = Transformer(path).pipe(template=template)

    assert tf.transform_data(
        'Hello *there,* world'
    ) == '<html><body><p>Hello <em>there,</em> world</p></body></html>'


def metaprocessor(src_path: AnySource | bytes, **meta: Any) -> str:
    if isinstance(src_path, (Path, ReadablePath)):  # pyright: ignore
        content = src_path.read_text('utf8')
    elif isinstance(src_path, bytes):
        content = src_path.decode('utf8')
    else:
        content = src_path
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

    assert tf.transform_data(content).rstrip() == dedent(
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

    assert tf.transform_data(content).rstrip() == dedent(
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

    assert tf.transform_data(content).rstrip() == dedent(
        '''\
            <html>
                <body>
                    <h1>Some Title</h1>
                    <p>Hello, world!</p>
                </body>
            </html>
        '''
    ).rstrip()


def test_metadata_joined() -> None:
    path = cast(Path, VirtualPath(
        'path/to/post.md',
        content=dedent(
            '''\
                ---
                title: Some Title
                date: 1970-01-01
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
    tf.preprocess(path).transform()
    assert tf.metadata == {
        'date': datetime.fromisoformat('1970-01-01'),
        'path': UrlPath('/path/to/post'),
        'url': UrlPath('/path/to/post'),
        'dir': '/path/to/',
        'file': LocalPath('path/to/post.html'),
        'title': 'Some Title',
        'word_count': 2,
        'excerpt': '<p>Hello, world!</p>',
    }
