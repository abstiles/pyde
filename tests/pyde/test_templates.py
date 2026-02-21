from html import unescape

from pyde import templates

from ..test import parametrize


@parametrize(
    ['simple text', 'simple text'],
    ['text with <em>emphasis</em> here', 'text with emphasis here'],
    ['<p>paragraph 1</p><p>paragraph 2</p>', 'paragraph 1\n\nparagraph 2'],
    ['leading text<p>paragraph</p>', 'leading text\n\nparagraph'],
)
def test_plaintext(html: str, expected: str) -> None:
    assert unescape(templates.plaintext(html)) == expected
