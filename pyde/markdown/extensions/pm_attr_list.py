"""
Attribute List Extension for Python-Markdown
============================================

Adds attribute list syntax. Inspired by
[maruku](http://maruku.rubyforge.org/proposal.html#attribute_lists)'s
feature of the same name.

See <https://Python-Markdown.github.io/extensions/attr_list>
for documentation.

Original code Copyright 2011 [Waylan Limberg](http://achinghead.com/).

All changes Copyright 2011-2014 The Python Markdown Project

License: [BSD](https://opensource.org/licenses/bsd-license.php)

Additional changes to allow list-wide and table-wide attributes 
by Paul Melis, 2022
"""

import xml.etree.ElementTree as ET
from markdown import Extension
from markdown.treeprocessors import Treeprocessor
import re


def _handle_double_quote(s, t):
    k, v = t.split('=', 1)
    return k, v.strip('"')


def _handle_single_quote(s, t):
    k, v = t.split('=', 1)
    return k, v.strip("'")


def _handle_key_value(s, t):
    return t.split('=', 1)


def _handle_word(s, t):
    if t.startswith('.'):
        return '.', t[1:].replace('.', ' ')
    if t.startswith('#'):
        return 'id', t[1:]
    return t, t


_scanner = re.Scanner([
    (r'[^ =]+=".*?"', _handle_double_quote),
    (r"[^ =]+='.*?'", _handle_single_quote),
    (r'[^ =]+=[^ =]+', _handle_key_value),
    (r'[^ =]+', _handle_word),
    (r' ', None)
])


def get_attrs(str):
    """ Parse attribute list and return a list of attribute tuples. """
    return _scanner.scan(str)[0]


def isheader(elem):
    return elem.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

def _dump_tree(level, elem):
    extra = []
    if elem.text:
        extra.append('text=[%s]' % repr(elem.text))
    if elem.tail:
        extra.append('tail=[%s]' % repr(elem.tail))
    e = ''
    if len(extra) > 0:
        e = '{' + ', '.join(extra) + '}'
        
    print('%s%s %s   %s' % (' '*(level*4), elem.tag, e, elem))
    
    if len(elem):
        for child in elem:
            _dump_tree(level+1, child)
    
def dump_tree(doc):        
    for elem in doc.findall('.'):
        _dump_tree(0, elem)

def _gather_parents(parents, parent, elem):
    #print(parent, elem, elem.tag, 'text='+repr(elem.text), 'tail='+repr(elem.tail))
    if parent is not None:
        parents[elem] = parent
    tag = elem.tag
    if tag in ['ol', 'ul', 'dl', 'table', 'blockquote']:
        parent = elem
    for child in elem:
        _gather_parents(parents, parent, child)        
    return parents

def gather_parents(doc):
    parents = {}
    for elem in doc.findall('.'):
        _gather_parents(parents, None, elem)
    return parents
    

class AttrListTreeprocessor(Treeprocessor):

    BASE_RE = r'\{\:?[ ]*([^\}\n ][^\}\n]*)[ ]*\}'
    HEADER_RE = re.compile(r'[ ]+{}[ ]*$'.format(BASE_RE))
    WITHIN_BLOCK_RE = re.compile(r'[ ]+{}[ ]*$'.format(BASE_RE), flags=re.MULTILINE)
    BLOCK_RE = re.compile(r'\n[ ]*{}[ ]*$'.format(BASE_RE))
    INLINE_RE = re.compile(r'^{}'.format(BASE_RE))
    NAME_RE = re.compile(r'[^A-Z_a-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02ff'
                         r'\u0370-\u037d\u037f-\u1fff\u200c-\u200d'
                         r'\u2070-\u218f\u2c00-\u2fef\u3001-\ud7ff'
                         r'\uf900-\ufdcf\ufdf0-\ufffd'
                         r'\:\-\.0-9\u00b7\u0300-\u036f\u203f-\u2040]+')

    def run(self, doc):
        #dump_tree(doc)
        parents = gather_parents(doc)
        for elem in doc.iter():
            parent = parents[elem] if elem in parents else None
            if self.md.is_block_level(elem.tag):
                # Block level: check for attrs on last line of text
                RE = self.BLOCK_RE
                if isheader(elem) or elem.tag in ['dt', 'td', 'th']:
                    # header, def-term, or table cell: check for attrs at end of element
                    RE = self.HEADER_RE
                if parent and parent.tag == 'blockquote':
                    #_dump_tree(0, parent)
                    if len(elem) and elem[-1].tail and (m := self.BLOCK_RE.search(elem[-1].tail)):
                        self.assign_attrs(parent, m.group(1))
                        elem[-1].tail = elem[-1].tail[:m.start()]
                    elif elem.text and (m := self.BLOCK_RE.search(elem.text)):
                        self.assign_attrs(parent, m.group(1))
                        elem.text = elem.text[:m.start()]
                    if elem.text and (m := self.WITHIN_BLOCK_RE.search(elem.text)):
                        self.assign_attrs(elem, m.group(1))
                        elem.text = elem.text[:m.start()]
                if len(elem) and elem.tag == 'li':
                    # special case list items. children may include a ul or ol.
                    pos = None
                    # find the ul or ol position
                    for i, child in enumerate(elem):
                        if child.tag in ['ul', 'ol']:
                            pos = i
                            break
                    if pos is None and elem[-1].tail:
                        # use tail of last child. no ul or ol.
                        m = RE.search(elem[-1].tail)
                        if m:
                            self.assign_attrs(elem, m.group(1), parent)
                            elem[-1].tail = elem[-1].tail[:m.start()]
                    elif pos is not None and pos > 0 and elem[pos-1].tail:
                        # use tail of last child before ul or ol
                        m = RE.search(elem[pos-1].tail)
                        if m:
                            self.assign_attrs(elem, m.group(1), parent)
                            elem[pos-1].tail = elem[pos-1].tail[:m.start()]
                    elif elem.text:
                        # use text. ul is first child.
                        m = RE.search(elem.text)
                        if m:
                            self.assign_attrs(elem, m.group(1), parent)
                            elem.text = elem.text[:m.start()]
                elif len(elem) and elem[-1].tail:
                    # has children. Get from tail of last child
                    m = RE.search(elem[-1].tail)
                    if m:
                        self.assign_attrs(elem, m.group(1), parent)
                        elem[-1].tail = elem[-1].tail[:m.start()]
                        if isheader(elem):
                            # clean up trailing #s
                            elem[-1].tail = elem[-1].tail.rstrip('#').rstrip()
                elif elem.text:
                    # no children. Get from text.
                    m = RE.search(elem.text)
                    if m:
                        self.assign_attrs(elem, m.group(1), parent)
                        elem.text = elem.text[:m.start()]
                        if isheader(elem):
                            # clean up trailing #s
                            elem.text = elem.text.rstrip('#').rstrip()
            else:
                # inline: check for attrs at start of tail
                if elem.tail:
                    m = self.INLINE_RE.match(elem.tail)
                    if m:
                        self.assign_attrs(elem, m.group(1), parent)
                        elem.tail = elem.tail[m.end():]

    def assign_attrs(self, elem, attrs, parent=None):
        """ Assign attrs to element. """
        #print('assign_attrs(%s, %s, %s)' % (elem, attrs, parent))
        apply_elem = elem if parent is None else parent
        for k, v in get_attrs(attrs):
            if k == '.':
                # add to class
                cls = apply_elem.get('class')
                if cls:
                    apply_elem.set('class', '{} {}'.format(cls, v))
                else:
                    apply_elem.set('class', v)
            elif k == '^' and parent is not None:
                # set remaining attributes on parent
                apply_elem = parent
            else:
                # assign attr k with v
                apply_elem.set(self.sanitize_name(k), v)

    def sanitize_name(self, name):
        """
        Sanitize name as 'an XML Name, minus the ":"'
        See https://www.w3.org/TR/REC-xml-names/#NT-NCName
        """
        return self.NAME_RE.sub('_', name)


class PMAttrListExtension(Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(AttrListTreeprocessor(md), 'pm_attr_list', 8)
        md.registerExtension(self)


def makeExtension(**kwargs):  # pragma: no cover
    return PMAttrListExtension(**kwargs)
