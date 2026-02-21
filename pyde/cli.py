"""
CLI interface to pyde
"""

import argparse
import sys
import os
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import cast, Mapping

from pyde.environment import Environment as Pyde
from pyde.markdown import split_frontmatter
from pyde.utils import slugify
from pyde.yaml import YamlType, parse_yaml_dict

from .config import Config


def main() -> int:
    """Entrypoint to the CLI"""
    prog, *args = sys.argv
    parser = argparse.ArgumentParser(prog=Path(prog).name)

    project_args = argparse.ArgumentParser(add_help=False)
    project_args.add_argument(
        '--root', '-r', metavar='DIR', dest='dir', type=directory,
        help='Root directory for sources'
    )

    build_args = argparse.ArgumentParser(add_help=False)
    build_args.add_argument(
        '--serve', action='store_true', help='Serve the site'
    )
    build_args.add_argument(
        '-d', '--destination', metavar='DIR', type=Path, default='_pysite'
    )
    build_args.add_argument(
        '--drafts', action='store_true', help='Render drafts'
    )

    config_settings = argparse.ArgumentParser(add_help=False)
    config_settings.add_argument('-c', '--config', type=Path, default='_config.yml')
    config_settings.add_argument('--drafts-dir', type=directory, default='_drafts')

    subparsers = parser.add_subparsers(help='command')
    subparsers.required = True

    build_parser = subparsers.add_parser(
        'build', help='Build your site', parents=[
            project_args, config_settings, build_args
        ],
    )
    build_parser.set_defaults(func=build)

    build_parser = subparsers.add_parser(
        'watch', help='Build your site and watch for changes',
        parents=[
            project_args, config_settings, build_args
        ],
    )
    build_parser.set_defaults(func=watch)

    config_cmd_parser = subparsers.add_parser(
        'config', parents=[project_args, config_settings, build_args],
        help='Print the current configuration, including all defaults',
    )
    config_cmd_parser.set_defaults(func=print_config)

    generate_cmd_parser = subparsers.add_parser(
        'generate', parents=[project_args, config_settings, build_args],
        help='Generate a basic Pyde skeleton',
    )
    generate_cmd_parser.set_defaults(func=generate)

    draft_cmd_parser = subparsers.add_parser(
        'draft', parents=[config_settings],
        help='Start a new draft',
    )
    draft_cmd_parser.set_defaults(func=draft)
    draft_cmd_parser.add_argument(
        'title', help='Title for the new draft', nargs='?'
    )
    draft_cmd_parser.add_argument(
        '--no-edit', dest='edit', action='store_false',
        help='Do not launch $EDITOR',
    )

    drafts_cmd_parser = subparsers.add_parser(
        'drafts', parents=[config_settings],
        help='List all drafts',
    )
    drafts_cmd_parser.set_defaults(func=list_drafts)

    post_cmd_parser = subparsers.add_parser(
        'post', parents=[project_args, config_settings],
        help='Post a draft',
    )
    post_cmd_parser.add_argument(
        'post', metavar='POST', type=Path,
        help='Draft to post',
    )
    post_cmd_parser.set_defaults(func=post)

    opts = parser.parse_args(args)
    status = opts.func(opts)
    try:
        sys.exit(int(status))
    except ValueError:
        raise RuntimeError(str(status)) from None


def build(opts: argparse.Namespace) -> int:
    """Build the site"""
    config = get_config(opts)
    Pyde(config).build(serve=opts.serve)
    return 0


def watch(opts: argparse.Namespace) -> int:
    """Build the site and watch"""
    config = get_config(opts)
    Pyde(config).watch(serve=opts.serve)
    return 0


def draft(opts: argparse.Namespace) -> int:
    config = get_config(opts)
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    title = opts.title if opts.title else f'Untitled {today}'
    filename = slugify(title) + '.md'
    # Ensure we don't override an existing file by adding a 1-based index if
    # a file with the given title already exists.
    idx = 1
    while (draft_file := config.drafts_dir / filename).exists():
        idx += 1
        filename = slugify(f'{title}') + f'.{idx}.md'
    draft_file.write_text(f'---\ntitle: {title}\ndate: {today}\ntags:\n---\n')
    if opts.edit:
        # And after all, why shouldn't I default this to vim?
        editor = os.environ.get('EDITOR', 'vim')
        os.execlp(editor, editor, str(draft_file))
    return 0


def list_drafts(opts: argparse.Namespace) -> int:
    config = get_config(opts)
    for path, metadata in get_drafts(config.drafts_dir):
        print(path)
        for k, v in metadata.items():
            print(f'    {k}: {v}')
    return 0


def post(opts: argparse.Namespace) -> int:
    config = get_config(opts)
    post: Path
    if (path := opts.post).exists():
        post = path
    elif (path := config.drafts_dir / opts.post).exists():
        post = path
    else:
        print(f'Error: {opts.post} does not exist', file=sys.stderr)
        return 1
    draft_data = get_draft_data(post)
    if not draft_data:
        print(f'Error: unable to parse metadata for {post}', file=sys.stderr)
        return 1
    dest_filename = slugify(str(draft_data['title'])) + '.md'
    dest_path = config.posts.source / dest_filename
    if dest_path.exists():
        print(
            f'Error: destination {str(dest_path)!r} would be overwritten.',
            'Change draft title to avoid overwriting existing post.',
            file=sys.stderr
        )
        return 1
    dest_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        frontmatter, body = split_frontmatter(path.read_text())
        new_date = date.today().isoformat()
        with dest_path.open('w') as f:
            print('---', file=f)
            for line in (frontmatter or '').split('\n'):
                if line.startswith('date: '):
                    print(f'date: {new_date}', file=f)
                else:
                    print(line, file=f)
            print('---', file=f)
            if body:
                print(body, file=f)
    except Exception:
        dest_path.unlink()
        raise
    post.unlink()
    return 0


def get_drafts(dir: Path) -> Iterable[tuple[Path, Mapping[str, YamlType]]]:
    draft_files = dir.glob('*.md')
    for path in draft_files:
        if (metadata := get_draft_data(path)):
            yield path, metadata


def get_draft_data(path: Path) -> Mapping[str, YamlType] | None:
    frontmatter = split_frontmatter(path.read_text())[0]
    if frontmatter is not None:
        return parse_yaml_dict(frontmatter)
    return None


def get_config(opts: argparse.Namespace, raw: bool=False) -> Config:
    config = Config.parse(opts.config, raw)
    try:
        if opts.dir:
            config.root = cast(Path, opts.dir)
    except AttributeError:
        pass
    try:
        if opts.drafts:
            config.drafts = opts.drafts
    except AttributeError:
        pass
    try:
        if opts.destination:
            config.output_dir = cast(Path, opts.destination)
    except AttributeError:
        pass
    try:
        if opts.drafts_dir:
            config.drafts_dir = cast(Path, opts.drafts_dir)
    except AttributeError:
        pass
    return config


def print_config(opts: argparse.Namespace) -> int:
    config = get_config(opts, raw=True)
    print(config.as_yaml())
    return 0


def generate(opts: argparse.Namespace) -> int:
    config = get_config(opts, raw=True)
    if not opts.config.exists():
        opts.config.write_text(config.as_yaml())
    dirs = [
        config.layouts_dir, config.includes_dir, config.plugins_dir,
        config.drafts_dir, config.posts.source
    ]
    for dir in dirs:
        dir.mkdir(exist_ok=True, parents=True)
    return 0


def directory(arg: str) -> Path:
    """Returns a Path if arg is a directory"""
    path = Path(arg)
    if path.is_dir():
        return path
    raise ValueError(f'{arg} is not a directory')


if __name__ == '__main__':
    main()
