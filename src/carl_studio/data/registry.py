"""Data source registry — loads sources from YAML, resolves adapters.

Usage:
    registry = DataRegistry.load()
    sources = registry.filter(domain=Domain.SWE, min_difficulty=0.2)
    adapter = registry.get_adapter("nemotron-cascade-swe")
    samples = adapter.load_and_adapt()
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from carl_studio.data.adapters.base import DataAdapter
from carl_studio.data.types import DataSource, Domain, Modality


_SOURCES_YAML = Path(__file__).parent / "sources.yaml"


class DataRegistry:
    """Registry of known data sources. Loaded from YAML, extensible at runtime."""

    def __init__(self, sources: list[DataSource]) -> None:
        self._sources = {s.name: s for s in sources}

    @classmethod
    def load(cls, path: Path | None = None) -> DataRegistry:
        """Load registry from YAML file."""
        path = path or _SOURCES_YAML
        with open(path) as f:
            data = yaml.safe_load(f)

        sources = []
        for entry in data.get("sources", []):
            sources.append(DataSource(
                name=entry["name"],
                repo_id=entry["repo_id"],
                adapter=entry["adapter"],
                domain=Domain(entry["domain"]),
                modality=Modality(entry["modality"]),
                description=entry.get("description", ""),
                license=entry.get("license", "unknown"),
                size=entry.get("size"),
                difficulty_range=tuple(entry["difficulty_range"]) if entry.get("difficulty_range") else None,
                default_split=entry.get("default_split", "train"),
                tags=entry.get("tags", []),
            ))

        return cls(sources)

    def register(self, source: DataSource) -> None:
        """Register a source at runtime (not persisted to YAML)."""
        self._sources[source.name] = source

    def get(self, name: str) -> DataSource:
        """Get a source by name."""
        if name not in self._sources:
            available = ", ".join(sorted(self._sources.keys()))
            raise KeyError(f"Unknown source '{name}'. Available: {available}")
        return self._sources[name]

    def list_sources(self) -> list[DataSource]:
        """List all registered sources."""
        return list(self._sources.values())

    def filter(
        self,
        domain: Domain | None = None,
        modality: Modality | None = None,
        tags: list[str] | None = None,
    ) -> list[DataSource]:
        """Filter sources by dimension."""
        results = list(self._sources.values())
        if domain:
            results = [s for s in results if s.domain == domain]
        if modality:
            results = [s for s in results if s.modality == modality]
        if tags:
            tag_set = set(tags)
            results = [s for s in results if tag_set.issubset(set(s.tags))]
        return results

    def get_adapter(self, name: str) -> DataAdapter:
        """Instantiate the adapter for a named source."""
        source = self.get(name)
        return _resolve_adapter(source)


def _resolve_adapter(source: DataSource) -> DataAdapter:
    """Import adapter class by module path and instantiate."""
    module_path, class_name = source.adapter.rsplit(".", 1)
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(source)
