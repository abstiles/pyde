import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from pyde.config import Config
from pyde.environment import Environment
from pyde.utils import dict_to_dataclass

TEST_DATA_DIR = Path(__file__).parent / 'test_data'
IN_DIR = Path('input')
OUT_DIR = TEST_DATA_DIR / 'output'
EXPECTED_DIR = TEST_DATA_DIR / 'expected'


@pytest.fixture
def output_dir() -> Iterable[Path]:
    OUT_DIR.mkdir(exist_ok=True)
    yield OUT_DIR
    for child in OUT_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def get_config(**kwargs: Any) -> Config:
    return dict_to_dataclass(
        Config,
        {
            'url': 'https://www.example.com',
            'output_dir': OUT_DIR,
            'permalink': '/:path/:name',
            'defaults': [
                {'values': {'layout': 'default'}},
                {'scope': {'path': 'posts'}, 'values': {'layout': 'post'}},
                {
                    'scope': {'path': '_drafts'},
                    'values': {'permalink': '/drafts/:title'},
                },
            ],
            **kwargs,
        }
    )


def get_env(**kwargs: Any) -> Environment:
    return Environment(get_config(**kwargs), exec_dir=TEST_DATA_DIR / IN_DIR)


LAYOUT_FILES = {
    '_layouts/index.html',
    '_layouts/post.html',
    '_layouts/default.html',
}
INCLUDE_FILES = {'_includes/header.html'}
DRAFT_FILES = {'_drafts/unfinished_post.md'}
SOURCE_FILES = {
    'index.md',
    'js/script.js',
    'posts/post.md',
    'styles/base.css',
}
OUTPUT_FILES = {
    'index.html',
    'js/script.js',
    'posts/post.html',
    'styles/base.css',
}
DRAFT_OUTPUT_FILES = OUTPUT_FILES | {
    'drafts/WIP.html',
}

def test_environment_source_files() -> None:
    env = get_env()
    assert set(map(str, env.source_files())) == SOURCE_FILES

def test_environment_layout_files() -> None:
    env = get_env()
    assert set(map(str, env.layout_files())) == LAYOUT_FILES

def test_environment_include_files() -> None:
    env = get_env()
    assert set(map(str, env.include_files())) == INCLUDE_FILES

def test_environment_draft_files() -> None:
    env = get_env()
    assert set(map(str, env.draft_files())) == DRAFT_FILES

def test_environment_output_files() -> None:
    env = get_env()
    assert set(map(str, env.output_files())) == OUTPUT_FILES

def test_environment_output_drafts() -> None:
    env = get_env(drafts=True)
    assert set(map(str, env.output_files())) == DRAFT_OUTPUT_FILES

def test_build(output_dir: Path) -> None:
    env = get_env(drafts=True)
    env.build(output_dir)
    for file in DRAFT_OUTPUT_FILES:
        expected = EXPECTED_DIR / file
        actual = output_dir / file

        assert actual.exists()
        assert actual.read_text().rstrip() == expected.read_text().rstrip()

def test_build_cleanup(output_dir: Path) -> None:
    dirty_file = output_dir / "inner" / "dirty.txt"
    dirty_file.parent.mkdir(parents=True, exist_ok=True)
    dirty_file.write_text("I shouldn't be here!")

    get_env(drafts=True).build(output_dir)

    assert not dirty_file.exists()
    for parent in dirty_file.relative_to(output_dir).parents:
        if parent != Path('.'):
            assert not (output_dir / parent).exists()
