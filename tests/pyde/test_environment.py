from pathlib import Path
from typing import Any

from pyde.config import Config
from pyde.environment import Environment

TEST_DATA_DIR = Path(__file__).parent / 'test_data'

def get_config(**kwargs: Any) -> Config:
    return Config(
        root=TEST_DATA_DIR,
        layouts_dir=TEST_DATA_DIR / '_layouts',
        includes_dir=TEST_DATA_DIR / '_includes',
        drafts_dir=TEST_DATA_DIR / '_drafts',
        output_dir=TEST_DATA_DIR / '_site',
        permalink='/:path/:basename',
        defaults=[
            {'values': {'layout': 'default'}},
            {'scope': {'path': 'posts'}, 'values': {'layout': 'post'}},
            {'scope': {'path': '_drafts'}, 'values': {'permalink': '/drafts/:title'}},
        ],
        **kwargs,
    )


def get_env() -> Environment:
    return Environment(get_config(), exec_dir=TEST_DATA_DIR)


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
