import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from .path.filepath import FilePath

QUESTION = '[^/]'
STAR = r'[^/]*'
INVERTED = r'[^'
DOUBLE_STAR = r'(?:^|(?<=/))[^/]*(/[^/]+)*/?'
END = f'(?:/?{DOUBLE_STAR}|$)'
PART = re.compile(r'''
    (?P<literal>            # Treat the same as regex patterns:
        [^*?\\+.|()^${}[]+  # any character that is not special
        | (?:\\.)+          # or any character preceded by a backslash
        | (?:\[\^)          # or a regex-style inverted character class
        | (?:\[(?!!))       # or a regex-style regular character class.
    )
    | (?P<double_star>  # Double stars match every (sub-)path.
        \*\*/?
    )
    | (?P<star>         # Stars match any single file or directory.
        \*
    )
    | (?P<question>     # Question marks match one character.
        \?
    )
    | (?P<inverted>     # Glob syntax for an inverted character class.
        \[!
    )
    | (?P<regex_char>   # Regex characters to be escaped for globs.
        [+{}^$()|.]
    )
''', re.VERBOSE)


class PathExpression:
    def __init__(self, glob: str):
        self._regex = re.compile(''.join(_split_parts(glob)) + END)

    def __repr__(self) -> str:
        return repr(self._regex)

    def match(self, path: Path | FilePath | str | bytes) -> bool:
        if isinstance(path, bytes):
            # Is this a reasonable thing to do? Maybe not?
            path = path.decode(errors='ignore')
        elif not isinstance(path, str):
            path = str(path)
        return bool(self._regex.match(path))


@lru_cache
def compile(glob: str) -> PathExpression:
    return PathExpression(glob)


def match(glob: str, path: FilePath | Path) -> bool:
    return compile(glob).match(path)


def _split_parts(glob: str) -> Iterable[str]:
    for match in re.finditer(PART, glob):
        if literal := match.group('literal'):
            yield literal
        elif double_star := match.group('double_star'):
            yield DOUBLE_STAR + double_star[2:]
        elif match.group('star'):
            yield STAR
        elif match.group('question'):
            yield QUESTION
        elif match.group('inverted'):
            yield INVERTED
        elif char := match.group('regex_char'):
            yield f'\\{char}'
