"""Problem templates for synthetic training data generation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib
import importlib.util
import pkgutil

__all__ = ["TestTier", "ProblemTemplate", "ALL_TEMPLATES", "discover_templates"]


@dataclass
class TestTier:
    name: str
    description: str
    check: str


@dataclass
class ProblemTemplate:
    id: str
    name: str
    dims: list[str]
    description: str
    file_skeleton: dict[str, str]
    bug_pattern: str
    tiers: list[TestTier]
    variation_axes: list[str]

    def as_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.id.replace("-", "_"),
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files from the codebase to base the problem on",
                        },
                        "variation": {
                            "type": "string",
                            "description": "What makes this instance unique",
                        },
                    },
                    "required": ["target_files"],
                },
            },
        }


def discover_templates() -> list[ProblemTemplate]:
    templates: list[ProblemTemplate] = []
    pkg_path = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
        if module_name.startswith("_"):
            continue
        mod = importlib.import_module(f"carl_studio.data.templates.{module_name}")
        if hasattr(mod, "TEMPLATES"):
            templates.extend(mod.TEMPLATES)
    user_dir = Path.home() / ".carl" / "templates"
    if user_dir.is_dir():
        for py_file in user_dir.glob("*.py"):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                if hasattr(mod, "TEMPLATES"):
                    templates.extend(mod.TEMPLATES)
    return templates


ALL_TEMPLATES: list[ProblemTemplate] = discover_templates()
