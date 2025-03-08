"""
CLI interface to pyde
"""

import argparse
import sys
from pathlib import Path
from typing import cast

from .config import Config
from .build import build_site


def main() -> int:
    """Entrypoint to the CLI"""
    prog, *args = sys.argv
    parser = argparse.ArgumentParser(prog=Path(prog).name)
    subparsers = parser.add_subparsers(help='command')
    subparsers.required = True
    build_parser = subparsers.add_parser('build', help='Build your site')
    build_parser.set_defaults(func=build)
    build_parser.add_argument('dir', type=directory, default='.', nargs='?')
    build_parser.add_argument('-c', '--config', type=Path, default='_config.yml')
    build_parser.add_argument(
        '-d', '--destination', metavar='DIR', type=Path, default='_pysite')
    opts = parser.parse_args(args)

    status = opts.func(opts)
    try:
        sys.exit(int(status))
    except ValueError:
        raise RuntimeError(str(status)) from None


def build(opts: argparse.Namespace) -> int:
    """Build the site"""
    root = cast(Path, opts.dir)
    dest = cast(Path, opts.destination)
    build_site(root, dest, Config.parse(opts.config))
    return 0


def directory(arg: str) -> Path:
    """Returns a Path if arg is a directory"""
    path = Path(arg)
    if path.is_dir():
        return path
    raise ValueError(f'{arg} is not a directory')


if __name__ == '__main__':
    main()
