from pathlib import Path
from unittest.mock import Mock

import pytest

from pyde.config import Config
from pyde.environment import Environment
from pyde.markdown import MarkdownConfig
from pyde.plugins import PluginManager
from pyde.templates import TemplateManager
from pyde.transformer import TransformerRegistration
from ..test import parametrize, NoParam


TEST_DATA_DIR = Path(__file__).parent / 'test_data'
IN_DIR = Path('input')


def get_config() -> Config:
    return Config(
        config_file=TEST_DATA_DIR / IN_DIR / '_config.yml',
    )


class TestPluginManager:
    @pytest.fixture(scope="class")
    def plugin_manager(self) -> PluginManager:
        config = get_config()
        template_manager = Mock(TemplateManager)
        env = Mock(Environment, config=config, template_manager=template_manager)
        manager = PluginManager(TEST_DATA_DIR / IN_DIR / '_plugins')
        manager.import_plugins(env)
        return manager

    def test_plugin_import(self, plugin_manager: PluginManager) -> None:
        assert 'user_plugins' in plugin_manager.plugin_modules

    def test_plugins_discovered(self, plugin_manager: PluginManager) -> None:
        user_plugins = plugin_manager.plugin_modules['user_plugins']
        assert set(plugin.__name__ for plugin in user_plugins.plugins) == {
            'MyMarkdownPlugin',
            'MyTemplatePlugin',
            'MyTransformerPlugin',
            'MyEnvironmentPlugin',
        }

    @parametrize(
        ('MyMarkdownPlugin', MarkdownConfig),
        ('MyTemplatePlugin', TemplateManager),
        ('MyTransformerPlugin', list[TransformerRegistration]),
        ('MyEnvironmentPlugin', Environment),
    )
    def test_plugins_invoked(
        self, plugin_manager: NoParam[PluginManager], name: str, type: type
    ) -> None:
        user_plugins = plugin_manager.plugin_modules['user_plugins']
        plugin_map = {
            plugin.__name__: plugin for plugin in user_plugins.plugins
        }
        invoked_with = plugin_map[name].value
        try:
            assert isinstance(invoked_with, type)
        except TypeError: # It's the parameterized list
            assert isinstance(invoked_with[0], type.__args__[0])  # type: ignore
