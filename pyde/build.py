"""
Build logic for Pyde
"""

from   argparse                 import Namespace
from   dataclasses              import dataclass, field
from   pathlib                  import Path
import re
import sys
from   typing                   import NewType

import yaml

from   .config                  import Config, SourceFile
from   .markdown                import Markdown
from   .templates               import Template, TemplateError


HTML = NewType('HTML', str)


@dataclass
class FileData:
    path: Path
    meta: dict[str, object] = field(default_factory=dict)
    content: str = ''
    has_frontmatter: bool = False

    @property
    def type(self) -> str:
        """Get the file's type"""
        return ''.join(self.path.suffixes).lstrip('.')

    @classmethod
    def load(cls, file: SourceFile) -> 'FileData':
        frontmatter, content = split_frontmatter(file.path.read_text())
        has_frontmatter = frontmatter is not None
        meta = {
            "page": Namespace(**{
                **file.values,
                **(yaml.safe_load(frontmatter or '') or {}), "content": content}
            ),
            "path": str(file.path.parent),
            "basename": file.path.name,
        }
        return cls(file.path, meta, content, has_frontmatter)


def build_site(src_dir: Path, dest_dir: Path, config: Config) -> None:
    """Build the site"""
    template = Template.from_config(config)
    sources = config.iter_files(src_dir, exclude=[dest_dir])
    files = map(FileData.load, sources)
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
            md = Markdown(content)
            content = md.html
            # meta.update(md.meta)
            file.meta['basename'] = file.path.with_suffix('.html').name
            file.meta['title'] = file.meta.get('title', file.meta['basename'])
        dest = dest_dir / get_path(file.meta)
        dest.parent.mkdir(parents=True, exist_ok=True)
        file.meta["page"].dir = dest.parent
        template_name = f'{file.meta["page"].layout}.{dest_type}'
        data = {**file.meta, "content": content}
        try:
            dest.write_text(template.apply(template_name, data))
        except TemplateError as exc:
            print(
                f'Unable to process {file.path} due to error in template'
                f' {template_name}: {exc.message}',
                file=sys.stderr
            )


def get_path(meta: dict[str, object]) -> Path:
    """Generate a path based on file metadata"""
    path = str(meta.get('permalink', '/:path/:basename'))
    def get(match: re.Match[str]) -> str:
        return str(meta.get(match.group(1), ''))
    return Path(re.sub(r':(\w+)', get, path).lstrip('/'))


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
