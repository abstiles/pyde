from pathlib import Path
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
