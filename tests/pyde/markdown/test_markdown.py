from textwrap import dedent

from pyde.markdown import MarkdownParser


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
    assert MarkdownParser().parse(md_text) == expected_html


def test_blockquote_complex_attrs() -> None:
    md_text = dedent('''\
    > This is a blockquote, paragraph 1.
    >
    > Paragraph 2 should get the next attr.
    > {: .p2attr }
    {: .bq-attr }

    ''').rstrip()
    expected_html = dedent('''\
        <blockquote class="bq-attr">
        <p>This is a blockquote, paragraph 1.</p>
        <p class="p2attr">Paragraph 2 should get the next attr.</p>
        </blockquote>
    ''').rstrip()
    assert MarkdownParser().parse(md_text) == expected_html


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
    assert MarkdownParser().parse(md_text) == expected_html
