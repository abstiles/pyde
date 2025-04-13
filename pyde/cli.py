"""
CLI interface to pyde
"""

import argparse
import sys
from pathlib import Path
from typing import cast

from pyde.environment import Environment as Pyde

from .config import Config


def main() -> int:
    """Entrypoint to the CLI"""
    prog, *args = sys.argv
    parser = argparse.ArgumentParser(prog=Path(prog).name)

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument(
        '--serve', action='store_true', help='Serve the site'
    )

    config_settings = argparse.ArgumentParser(add_help=False)
    config_settings.add_argument(
        '--drafts', action='store_true', help='Render drafts'
    )
    config_settings.add_argument('dir', type=directory, nargs='?')
    config_settings.add_argument('-c', '--config', type=Path, default='_config.yml')
    config_settings.add_argument(
        '-d', '--destination', metavar='DIR', type=Path, default='_pysite')

    subparsers = parser.add_subparsers(help='command')
    subparsers.required = True

    build_parser = subparsers.add_parser(
        'build', help='Build your site', parents=[config_settings, common_args],
    )
    build_parser.set_defaults(func=build)

    build_parser = subparsers.add_parser(
        'watch', help='Build your site and watch for changes',
        parents=[config_settings, common_args],
    )
    build_parser.set_defaults(func=watch)

    config_cmd_parser = subparsers.add_parser(
        'config', parents=[config_settings],
        help='Print the current configuration, including all defaults',
    )
    config_cmd_parser.set_defaults(func=print_config)

    generate_cmd_parser = subparsers.add_parser(
        'generate', parents=[config_settings],
        help='Generate a basic Pyde skeleton',
    )
    generate_cmd_parser.set_defaults(func=generate)

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


def get_config(opts: argparse.Namespace, raw: bool=False) -> Config:
    config = Config.parse(opts.config, raw)
    if opts.dir:
        config.root = cast(Path, opts.dir)
    if opts.drafts:
        config.drafts = True
    if opts.destination:
        config.output_dir = cast(Path, opts.destination)
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
