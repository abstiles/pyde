import re
import shutil
from collections.abc import Generator, Iterable
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


def get_config(**kwargs: Any) -> Config:
    return dict_to_dataclass(
        Config,
        {
            'config_file': TEST_DATA_DIR / IN_DIR / '_config.yml',
            'url': 'https://www.example.com',
            'include': ['.htaccess'],
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
            'tags': 'tag',
            'paginate': {'template': 'postindex', 'size': 2},
            **kwargs,
        }
    )


def get_env(**kwargs: Any) -> Environment:
    return Environment(get_config(**kwargs))


LAYOUT_FILES = {
    '_layouts/index.html',
    '_layouts/post.html',
    '_layouts/default.html',
    '_layouts/tag.html',
    '_layouts/postindex.html',
}
INCLUDE_FILES = {'_includes/header.html'}
DRAFT_FILES = {'_drafts/unfinished_post.md'}
SOURCE_FILES = {
    'index.md',
    'js/script.js',
    '_posts/post.md',
    '_posts/another-post.md',
    '_posts/third-post.md',
    'styles/base.css',
    '.htaccess',
}
RAW_OUTPUTS = {
    '.htaccess',
    'js/script.js',
    'styles/base.css',
}
PAGE_OUTPUTS = {
    'index.html',
}
META_OUTPUTS = {
    'tag/test.html',
    'tag/tag-with-spaces.html',
    'posts/page.1.html',
    'posts/page.2.html',
    'posts/index.html',
}
DRAFT_OUTPUTS = {
    'drafts/WIP.html',
}
DRAFT_META_OUTPUTS = {
    'drafts/index.html',
}
POST_OUTPUTS = {
    'posts/post.html',
    'posts/another-post.html',
    'posts/third-post.html',
}
OUTPUT_FILES = RAW_OUTPUTS | PAGE_OUTPUTS | POST_OUTPUTS | META_OUTPUTS
DRAFT_OUTPUT_FILES = OUTPUT_FILES | DRAFT_OUTPUTS | DRAFT_META_OUTPUTS


def make_output_dir() -> None:
    OUT_DIR.mkdir(exist_ok=True)


def clean_output_dir() -> None:
    for child in OUT_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


class TestFileSets:
    @pytest.fixture(autouse=True)
    def output_dir(self) -> Iterable[Path]:
        make_output_dir()
        yield OUT_DIR
        clean_output_dir()

    def test_environment_source_files(self) -> None:
        env = get_env()
        assert set(map(str, env.source_files())) == SOURCE_FILES

    def test_environment_layout_files(self) -> None:
        env = get_env()
        assert set(map(str, env.layout_files())) == LAYOUT_FILES

    def test_environment_include_files(self) -> None:
        env = get_env()
        assert set(map(str, env.include_files())) == INCLUDE_FILES

    def test_environment_draft_files(self) -> None:
        env = get_env()
        assert set(map(str, env.draft_files())) == DRAFT_FILES

    def test_environment_output_files(self) -> None:
        env = get_env()
        assert set(map(str, env.output_files())) == OUTPUT_FILES

    def test_environment_output_drafts(self) -> None:
        env = get_env(drafts=True)
        assert set(map(str, env.output_files())) == DRAFT_OUTPUT_FILES

    def test_build_cleanup(self) -> None:
        dirty_file = OUT_DIR / "inner" / "dirty.txt"
        dirty_file.parent.mkdir(parents=True, exist_ok=True)
        dirty_file.write_text("I shouldn't be here!")

        get_env(drafts=True).build()

        assert not dirty_file.exists()
        for parent in dirty_file.relative_to(OUT_DIR).parents:
            if parent != Path('.'):
                assert not (OUT_DIR / parent).exists()


class TestBuild:
    @pytest.fixture(scope="class", autouse=True)
    def build(self) -> Generator[None, None, None]:
        make_output_dir()
        env = get_env(drafts=True)
        env.build()
        yield
        clean_output_dir()

    @parametrize(*DRAFT_OUTPUT_FILES)
    def test_outputs_exist(self, file: str) -> None:
        print(f'Files in {OUT_DIR}:', '\n'.join(map(str, OUT_DIR.rglob('**/*'))))
        output_file = OUT_DIR / file
        assert output_file.exists(), f'{str(output_file)!r} not found'

    @parametrize(*DRAFT_OUTPUT_FILES)
    def test_outputs_match_contents(self, file: str) -> None:
        actual = (OUT_DIR / file).read_text().rstrip()
        expected = (EXPECTED_DIR / file).read_text().rstrip()
        # Drafts shouldn't have a real publish date, so all they'll have is a
        # date that comes from the filesystem ctime/mtime/birthtime, the
        # accuracy of which is not an interesting part of the test. Therefore
        # this test replaces the specific date with Xs.
        assert re.sub(
            r'"date">\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \+0000',
            '"date">XXXX-XX-XX XX:XX:XX +0000',
            actual,
        ) == expected


@parametrize(
    ['raw', RAW_OUTPUTS],
    ['pages', PAGE_OUTPUTS],
    ['posts', POST_OUTPUTS | DRAFT_OUTPUTS],
    ['meta', META_OUTPUTS | DRAFT_META_OUTPUTS ],
)
def test_site_files(
    type: str, results: set[str]
) -> None:
    env = get_env(drafts=True)
    assert set(
        str(file['file']) for file in getattr(env.site, type)
    ) == results

@parametrize(
    ['markdown', {'posts/post.html'}],
    ['test', {'posts/post.html', 'posts/another-post.html'}],
    ['tag with spaces', {'posts/post.html', 'posts/another-post.html'}],
)
def test_tags(
    tag: str, posts: set[str]
) -> None:
    env = get_env(drafts=True)
    assert set(
        str(post['file']) for post in env.site.tags[tag]
    ) == posts
