from pathlib import Path

from pyde.site import DirTree, SiteFile
from pyde.transformer import CopyTransformer
from ..test import parametrize


TREE_DATA = {
    '/index.html': 'index-data',
    '/posts/something.html': 'post-something-data',
    '/posts/whatever.html': 'post-whatever-data',
    '/posts/extra/something.js': 'extra-something-data',
}

def generate_dirtree(datamap: dict[str, str] = TREE_DATA) -> DirTree:
    return DirTree(
        path_to_sitefile(path, data)
        for path, data in datamap.items()
    )

def path_to_sitefile(path: str, data: str) -> SiteFile:
    relative = Path(path).relative_to('/')
    ret = SiteFile.raw(
        CopyTransformer(relative).preprocess(relative)
    )
    ret.metadata['data'] = data
    return ret


@parametrize(
    *TREE_DATA.keys()
)
def test_dirtree(path: str) -> None:
    dirtree = generate_dirtree()
    assert dirtree[path].metadata.data == TREE_DATA[path]


def test_dirtree_relative_to_path() -> None:
    dirtree = generate_dirtree()
    assert dirtree['/posts'].metadata.files == [
        'something.html', 'whatever.html',
    ]


def test_dirtree_walk() -> None:
    def make_structure(tree: DirTree) -> str:
        level = 0
        lines = []
        for entry in tree.walk():
            if entry.pre_children:
                lines.append('    ' * level + entry.path.name)
                level += 1
            if entry.post_children:
                for (path, file) in entry.contents.items():
                    lines.append(
                        '    ' * level + f'{path} - {file.metadata.data}'
                    )
                level -= 1
        return '\n'.join(lines)
    dirtree = generate_dirtree()
    assert make_structure(dirtree) == '''
    posts
        extra
            something.js - extra-something-data
        something.html - post-something-data
        whatever.html - post-whatever-data
    index.html - index-data'''
