from dataclasses import dataclass

from pyde import (
    Environment,
    EnvironmentPlugin,
    MarkdownConfig,
    MarkdownPlugin,
    TemplateManager,
    TemplatePlugin,
    TransformerPlugin,
    TransformerRegistration,
)


class MyMarkdownPlugin(MarkdownPlugin):
    value: MarkdownConfig | None = None
    def configure(self, config: MarkdownConfig) -> None:
        self.__class__.value = config


@dataclass
class MyTemplatePlugin(TemplatePlugin):
    value: TemplateManager | None = None
    def configure(self, template_manager: TemplateManager) -> None:
        self.__class__.value = template_manager


@dataclass
class MyTransformerPlugin(TransformerPlugin):
    value: list[TransformerRegistration] | None = None
    def configure(self, transformers: list[TransformerRegistration]) -> None:
        self.__class__.value = transformers


@dataclass
class MyEnvironmentPlugin(EnvironmentPlugin):
    value: Environment | None = None
    def configure(self, environment: Environment) -> None:
        self.__class__.value = environment
