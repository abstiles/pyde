import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, Never, Self, TypeGuard

from pyde.templates import TemplateManager

from .config import Config
from .markdown.handler import MarkdownConfig
from .path import LocalPath
from .transformer import Transformer, TransformerRegistration

if TYPE_CHECKING:
    # This is a circular dependency only needed for type annotations.
    from .environment import Environment


class PydePlugin(ABC):
    registered: ClassVar[list[type[Self]]] = []

    def __init__(self, _: Config, /): ...

    @classmethod
    def register(cls, plugin: type[Self]) -> None:
        cls.registered.append(plugin)

    def __init_subclass__(cls) -> None:
        if cls.__base__ is PydePlugin:
            PydePlugin.register(cls)
            cls.registered = []
        else:
            cls.register(cls)

    @abstractmethod
    def configure(self, it: Never, /) -> None: ...


class EnvironmentPlugin(PydePlugin):
    @abstractmethod
    def configure(self, environment: 'Environment', /) -> None: ...


class TransformerPlugin(PydePlugin):
    @abstractmethod
    def configure(self, transformers: list[TransformerRegistration], /) -> None: ...


class TemplatePlugin(PydePlugin):
    @abstractmethod
    def configure(self, template_manager: TemplateManager, /) -> None: ...


class MarkdownPlugin(PydePlugin):
    @abstractmethod
    def configure(self, config: MarkdownConfig, /) -> None: ...


@dataclass
class PluginModule:
    module: ModuleType

    @property
    def plugins(self) -> Iterable[type[PydePlugin]]:
        classes = dict(inspect.getmembers(self.module, inspect.isclass))
        return filter(self.is_pyde_plugin, classes.values())

    def is_pyde_plugin(self, plugin_class: type) -> TypeGuard[type[PydePlugin]]:
        return (
            plugin_class.__module__ == self.module.__name__
            and issubclass(plugin_class, PydePlugin)
        )

    @classmethod
    def load(cls, name: str) -> 'PluginModule':
        return cls(import_module(name))


class PluginManager:
    def __init__(self, plugins_dir: LocalPath | Path):
        self.plugins_dir = plugins_dir
        self.plugin_modules: dict[str, PluginModule] = {}

    def import_plugins(self, env: 'Environment') -> None:
        self._import_plugins_dir()
        self._apply_markdown_plugins(env)
        self._apply_template_plugins(env)
        self._apply_transformer_plugins(env)
        self._apply_environment_plugins(env)

    def _apply_markdown_plugins(self, env: 'Environment') -> None:
        if not MarkdownPlugin.registered:
            return
        md_config = MarkdownConfig()
        for plugin in MarkdownPlugin.registered:
            plugin(env.config).configure(md_config)
        parser = md_config.make_parser()
        env.markdown_parser = parser

    def _apply_template_plugins(self, env: 'Environment') -> None:
        for plugin in TemplatePlugin.registered:
            plugin(env.config).configure(env.template_manager)

    def _apply_transformer_plugins(self, env: 'Environment') -> None:
        for plugin in TransformerPlugin.registered:
            plugin(env.config).configure(Transformer.registered)

    def _apply_environment_plugins(self, env: 'Environment') -> None:
        for plugin in EnvironmentPlugin.registered:
            plugin(env.config).configure(env)

    def _import_plugins_dir(self) -> None:
        if not self.plugins_dir.is_dir():
            return
        # Save the path to restore afterward, preventing plugins from tampering
        # with it.
        save_path = [*sys.path]
        try:
            sys.path.insert(0, str(self.plugins_dir.absolute()))
            for plugin_file in self.plugins_dir.glob('*.py'):
                name = self._module_name(plugin_file)
                module = PluginModule.load(name)
                self.plugin_modules[name] = module
        finally:
            sys.path = save_path

    def _module_name(self, path: Path) -> str:
        path = path.relative_to(self.plugins_dir)
        if path.stem == '__init__':
            path = path.parent
        return '.'.join(path.with_suffix('').parts)
