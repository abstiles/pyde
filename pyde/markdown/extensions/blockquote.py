import re

import xml.etree.ElementTree as etree

from markdown import util
from markdown.extensions import Extension
from markdown.blockprocessors import BlockProcessor


class BlockQuoteProcessor(BlockProcessor):
    """
    replaces: https://github.com/Python-Markdown/markdown/blob/f6ca75429562cfa7df333b3529838679e4bfd443/markdown/blockprocessors.py#L277
    """

    RE = re.compile(r'(^|\n)[ ]{0,3}>[ ]?(.*)')

    def test(self, parent, block):
        return bool(self.RE.search(block)
                   ) and not util.nearing_recursion_limit()

    def run(self, parent, blocks):
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            before = block[:m.start()]  # Lines before blockquote
            # Pass lines before blockquote in recursively for parsing first.
            self.parser.parseBlocks(parent, [before])
            # Remove ``> `` from beginning of each line.
            block = '\n'.join(
                [self.clean(line) for line in block[m.start():].split('\n')]
            )
            
        # This is a new blockquote. Create a new parent element.
        quote = etree.SubElement(parent, 'blockquote')
        # Recursively parse block with blockquote as parent.
        # change parser state so blockquotes embedded in lists use p tags
        self.parser.state.set('blockquote')
        self.parser.parseChunk(quote, block)
        self.parser.state.reset()

    def clean(self, line):
        """ Remove ``>`` from beginning of a line. """
        m = self.RE.match(line)
        if line.strip() == '>':
            return ''
        elif m:
            return m.group(2)
        else:
            return line


class BlockQuoteExtension(Extension):
    def extendMarkdown(self, md):
        """Extend the inline and block processor objects."""

        md.parser.blockprocessors.register(
            BlockQuoteProcessor(md.parser), 'quote', 70
        )
        md.registerExtension(self)
