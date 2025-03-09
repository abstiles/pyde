"""
Build logic for Pyde
"""

from   collections.abc          import Mapping
from   dataclasses              import dataclass, field
from   pathlib                  import Path
import re
import sys
from   typing                   import Iterable, NewType

import yaml

from   .config                  import Config, SourceFile
from   .markdown                import Markdown
from   .templates               import Template, TemplateError


HTML = NewType('HTML', str)


class Data(Mapping[str, object]):
    def __init__(self, d: dict[str, object]=None, **kwargs: object):
        super().__setattr__('_d', d or {})
        self._d.update(kwargs)

    def __iter__(self) -> Iterable[str]:
        yield from self._d

    def __len__(self) -> int:
        len(self._d)

    def __bool__(self) -> bool:
        return bool(self._d)

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __getitem__(self, key: str) -> object:
        return self._d[key]

    def __getattr__(self, key: str) -> object:
        try:
            return self._d[key]
        except KeyError:
            return None

    def __setitem__(self, key: str, val: object) -> None:
        self._d[key] = val

    def __setattr__(self, key: str, val: object) -> None:
        self[key] = val

    def __delitem__(self, key: str) -> None:
        del self._d[key]

    def __delattr__(self, key: str) -> None:
        del self[key]

    def __repr__(self) -> str:
        return repr(self._d)


@dataclass
class FileData:
    path: Path
    meta: dict[str, object] = field(default_factory=dict)
    has_frontmatter: bool = False
    is_binary: bool = False

    @property
    def type(self) -> str:
        """Get the file's type"""
        return ''.join(self.path.suffixes).lstrip('.')

    @property
    def content(self) -> str | bytes:
        try:
            return get_content(self.path.read_text())
        except UnicodeDecodeError:
            return self.path.read_bytes()

    @classmethod
    def load(cls, file: SourceFile) -> 'FileData':
        is_binary = False
        try:
            frontmatter = get_frontmatter(file.path.read_text())
        except UnicodeDecodeError:
            frontmatter = None
            is_binary = True
        has_frontmatter = frontmatter is not None
        if file.path.suffix == '.md':
            basename = file.path.stem
        else:
            basename = file.path.name
        meta = Data(
            page=Data({
                **file.values,
                **(yaml.safe_load(frontmatter or '') or {}),
            }),
            path=str(file.path.parent),
            basename=basename,
        )
        meta.title = meta.title or meta.basename
        meta.page.url = f'/{get_path(meta)}'
        meta.page.dir = f'{Path(meta.page.url).parent}/'
        # A stupid hack
        if meta.page.links is None:
            meta.page.links = []
        if 'alt' not in meta.page:
            meta.page.alt = None
        return cls(file.path, meta, has_frontmatter, is_binary)


def build_site(src_dir: Path, dest_dir: Path, config: Config) -> None:
    """Build the site"""
    template = Template.from_config(config)
    sources = config.iter_files(src_dir, exclude=[dest_dir])
    files = [*map(FileData.load, sources)]
    site = Data(pages=[file.meta.page for file in files if file.type == "md"],
                url=config.url)
    for file in files:
        dest_type = file.type
        if file.has_frontmatter:
            try:
                content = template.render(file.content, file.meta)
            except TemplateError as exc:
                print(f'Skipping {file.path} due to error: {exc.message}',
                      file=sys.stderr)
                continue
        else:
            content = file.content
        if file.type == 'md':
            dest_type = 'html'
            content = Markdown(content).html
        dest = dest_dir / get_path(file.meta)
        if dest_type == 'html':
            dest = dest.with_suffix('.html')
        dest.parent.mkdir(parents=True, exist_ok=True)
        file.meta.page.dir = f'/{dest.parent.relative_to(dest_dir)}/'
        file.meta.page.content = file.content
        template_name = f'{file.meta["page"].layout}.{dest_type}'
        data = {**file.meta, "site": site, "content": content}
        try:
            dest.write_text(template.apply(template_name, data))
        except TemplateError as exc:
            print(
                f'Unable to process {file.path} due to error in template'
                f' {template_name}: {exc.message}',
                file=sys.stderr
            )


def get_path(meta: Data) -> Path:
    """Generate a path based on file metadata"""
    path = str(meta.permalink or '/:path/:basename')
    def get(match: re.Match[str]) -> str:
        try:
            return str(meta[match.group(1)])
        except KeyError:
            return ''
    return Path(re.sub(r':(\w+)', get, path).lstrip('/'))


def get_frontmatter(source: str) -> str | None:
    """Split a file into the frontmatter and text file components"""
    text = text_file(source)
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 3)
    return text[4:end]


def get_content(source: str) -> str:
    """Split a file into the frontmatter and text file components"""
    text = text_file(source)
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 3)
    return text[end + 5:]


def split_frontmatter(source: str) -> tuple[str | None, str]:
    """Split a file into the frontmatter and text file components"""
    text = text_file(source)
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 3)
    frontmatter = text[4:end]
    text = text[end + 5:]
    return frontmatter, text


def text_file(contents: str) -> str:
    """Make sure this is a valid text file ending in newline"""
    if contents.endswith('\n'):
        return contents
    return contents + '\n'
