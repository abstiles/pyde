"""Handler for templated result files"""

from pathlib import Path
from typing import Callable

import jinja2
from jinja2 import Template as JinjaTemplate
from jinja2 import BaseLoader
from jinja2 import Environment, select_autoescape

from .config import Config


class Template:
    def __init__(self, templates_dir: Path=Path('_layouts')):
        self.env = Environment(
            loader=TemplateLoader(templates_dir),
            autoescape=select_autoescape(
                enabled_extensions=(),
                default_for_string=False,
            )
        )

    @classmethod
    def from_config(cls, config: Config) -> 'Template':
        return cls(config.layouts_dir)

    def get_template(self, name: str) -> JinjaTemplate:
        return self.env.get_template(name)

    def apply(self, template: str, data: dict[str, object]) -> str:
        try:
            return self.get_template(template).render(data)
        except jinja2.exceptions.TemplateError as exc:
            raise TemplateError("Invalid template syntax", exc.message) from exc

    def render(self, source: str, data: dict[str, object]) -> str:
        try:
            return JinjaTemplate(source).render(data)
        except jinja2.exceptions.TemplateError as exc:
            raise TemplateError("Invalid template syntax", exc.message) from exc


class TemplateError(ValueError):
    @property
    def message(self) -> str:
        return ' - '.join(self.args)


class TemplateLoader(BaseLoader):
    """Main loader for templates"""
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str | None, Callable[[], bool] | None]:
        path = self.templates_dir / template
        if path.is_file():
            mtime = path.stat().st_mtime
            source = path.read_text()
            return source, str(path), lambda: mtime == path.stat().st_mtime
        return '{{ content }}', "", lambda: True
