"""
Build logic for Pyde
"""

import sys
import shutil
import traceback
from itertools import dropwhile

from .config import Config
from .data import Data, FileData
from .markdown import markdownify
from .templates import TemplateManager, TemplateError


def build_site(config: Config) -> None:
    """Build the site"""
    template = TemplateManager.from_config(config)
    files = [*FileData.iter_files(config)]
    site = Data(pages=[file.meta.page for file in files if file.type == "md"],
                url=config.url)
    site.posts = site.pages
    for file in files:
        dest_type = file.type
        dest = config.output_dir / file.meta.file_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not file.has_frontmatter:
            # If it lacks frontmatter, then treat it as a plain file and copy
            # it to where it needs to be.
            shutil.copy(file.path, dest)
            continue
        try:
            content = template.render(file.content, {"site": site, **file.meta})
        except TemplateError as exc:
            print(f'Skipping {file.path} due to error: {exc.message}',
                  file=sys.stderr)
            continue
        if file.type == 'md':
            dest_type = 'html'
            content = markdownify(content)
            dest = dest.with_suffix('.html')
        file.meta.page.dir = f'/{dest.parent.relative_to(config.output_dir)}/'
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
            template_tb = dropwhile(
                lambda l: 'template code' not in l,
                traceback.format_exception(exc.__cause__)
            )
            print(''.join(template_tb), file=sys.stderr)
