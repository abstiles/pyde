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
    subparsers = parser.add_subparsers(help='command')
    subparsers.required = True
    build_parser = subparsers.add_parser('build', help='Build your site')
    build_parser.set_defaults(func=build)
    build_parser.add_argument('dir', type=directory, nargs='?')
    build_parser.add_argument('-c', '--config', type=Path, default='_config.yml')
    build_parser.add_argument(
        '-d', '--destination', metavar='DIR', type=Path, default='_pysite')
    build_parser.add_argument('--drafts', action='store_true')
    opts = parser.parse_args(args)

    status = opts.func(opts)
    try:
        sys.exit(int(status))
    except ValueError:
        raise RuntimeError(str(status)) from None


def build(opts: argparse.Namespace) -> int:
    """Build the site"""
    config = Config.parse(opts.config)
    if opts.dir:
        config.root = cast(Path, opts.dir)
    if opts.drafts:
        config.drafts = True
    if opts.destination:
        config.output_dir = cast(Path, opts.destination)
    Pyde(config).build()
    return 0


def directory(arg: str) -> Path:
    """Returns a Path if arg is a directory"""
    path = Path(arg)
    if path.is_dir():
        return path
    raise ValueError(f'{arg} is not a directory')


if __name__ == '__main__':
    main()
