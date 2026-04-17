"""Recipe — multi-course training pipeline definition.

A Recipe chains Courses. Each Course is a Kit + Kitchen + Gate.
Users write YAML, carl runs it end-to-end.

Example carl-recipe.yaml:
    name: my-custom-agent
    base: your-org/your-model
    courses:
      - kit: coding-agent
        steps: 100
        gate:
          metric: task_completion
          threshold: 0.8
      - kit: reasoning
        steps: 50
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["Recipe", "CourseSpec", "load_recipe"]


@dataclass
class CourseSpec:
    """One training stage in a recipe."""
    kit: str                          # kit ID from registry
    steps: int = 100
    gate: dict = field(default_factory=dict)  # {metric, threshold}
    num_generations: int = 4
    learning_rate: float = 5e-6
    temperature: float = 1.2


@dataclass
class Recipe:
    """Full training pipeline — list of courses."""
    name: str
    base: str = ""
    hub_namespace: str = ""
    courses: list[CourseSpec] = field(default_factory=list)
    source: str = ""                  # where to gather data from
    count: int = 50                   # samples per course

    @property
    def hub_id(self) -> str:
        ns = self.hub_namespace or ""
        slug = self.name.lower().replace(" ", "-").replace("_", "-")
        base_tag = self.base.split("/")[-1].lower().replace("-", "")[:10] if self.base else "base"
        prefix = self._resolve_naming_prefix()
        model_id = f"{prefix}-{base_tag}-{slug}" if prefix else f"{base_tag}-{slug}"
        return f"{ns}/{model_id}" if ns else model_id

    @staticmethod
    def _resolve_naming_prefix() -> str:
        """Resolve naming prefix from settings if available."""
        try:
            from carl_studio.settings import CARLSettings
            settings = CARLSettings.load()
            return settings.naming_prefix
        except Exception:
            return ""


def load_recipe(path: str | Path) -> Recipe:
    """Load recipe from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recipe not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Recipe must be a YAML mapping, got {type(data)}")

    courses = []
    for c in data.get("courses", []):
        if not isinstance(c, dict):
            raise ValueError(f"Each course must be a YAML mapping, got {type(c)}")
        courses.append(CourseSpec(
            kit=str(c.get("kit", "coding-agent")),
            steps=int(c.get("steps", 100)),
            gate=c.get("gate", {}) or {},
            num_generations=int(c.get("num_generations", 4)),
            learning_rate=float(c.get("learning_rate", 5e-6)),
            temperature=float(c.get("temperature", 1.2)),
        ))

    # Default: single course with coding-agent kit
    if not courses:
        courses = [CourseSpec(kit="coding-agent")]

    return Recipe(
        name=str(data.get("name", path.stem)),
        base=str(data.get("base", "") or ""),
        hub_namespace=str(data.get("hub_namespace", "") or ""),
        courses=courses,
        source=str(data.get("source", "") or ""),
        count=int(data.get("count", 50)),
    )
