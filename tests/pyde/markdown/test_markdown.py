from textwrap import dedent

from pyde.markdown import markdownify


def test_blockquote_attrs() -> None:
    md_text = dedent('''\
        > This is a blockquote
        {: .some-class }
    ''').rstrip()
    expected_html = dedent('''\
        <blockquote class="some-class">
        <p>This is a blockquote</p>
        </blockquote>
    ''').rstrip()
    assert markdownify(md_text) == expected_html


def test_list_attrs() -> None:
    md_text = dedent('''\
        * This is an unordered list
        * With two items
        {: .some-class }
    ''').rstrip()
    expected_html = dedent('''\
        <ul class="some-class">
        <li>This is an unordered list</li>
        <li>With two items</li>
        </ul>
    ''').rstrip()
    assert markdownify(md_text) == expected_html
