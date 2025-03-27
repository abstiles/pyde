import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from pyde.config import Config
from pyde.environment import Environment
from pyde.utils import dict_to_dataclass

from ..test import parametrize

TEST_DATA_DIR = Path(__file__).parent / 'test_data'
IN_DIR = Path('input')
OUT_DIR = TEST_DATA_DIR / 'output'
EXPECTED_DIR = TEST_DATA_DIR / 'expected'


@pytest.fixture(autouse=True)
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
            'permalink': '/:path/:basename',
            'defaults': [
                {'values': {'layout': 'default'}},
                {'scope': {'path': '_posts'}, 'values': {'layout': 'post'}},
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
    '_posts/post.md',
    'styles/base.css',
}
RAW_OUTPUTS = {
    'js/script.js',
    'styles/base.css',
}
PAGE_OUTPUTS = {
    'index.html',
}
DRAFT_OUTPUTS = {
    'drafts/WIP.html',
}
POST_OUTPUTS = {
    'posts/post.html',
}
OUTPUT_FILES = RAW_OUTPUTS | PAGE_OUTPUTS | POST_OUTPUTS
DRAFT_OUTPUT_FILES = OUTPUT_FILES | DRAFT_OUTPUTS

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

def test_build() -> None:
    env = get_env(drafts=True)
    env.build()
    for file in DRAFT_OUTPUT_FILES:
        expected = EXPECTED_DIR / file
        actual = OUT_DIR / file

        assert actual.exists()
        assert actual.read_text().rstrip() == expected.read_text().rstrip()

def test_build_cleanup() -> None:
    dirty_file = OUT_DIR / "inner" / "dirty.txt"
    dirty_file.parent.mkdir(parents=True, exist_ok=True)
    dirty_file.write_text("I shouldn't be here!")

    get_env(drafts=True).build()

    assert not dirty_file.exists()
    for parent in dirty_file.relative_to(OUT_DIR).parents:
        if parent != Path('.'):
            assert not (OUT_DIR / parent).exists()

@parametrize(
    ['raw', RAW_OUTPUTS],
    ['pages', PAGE_OUTPUTS],
    ['posts', POST_OUTPUTS | DRAFT_OUTPUTS],
)
def test_site_files(
    type: str, results: set[str]
) -> None:
    env = get_env(drafts=True)
    assert set(
        str(file['file']) for file in getattr(env.site, type)
    ) == results
