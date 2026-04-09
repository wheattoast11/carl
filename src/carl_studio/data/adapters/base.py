"""Base adapter — the contract all dataset adapters implement.

An adapter transforms one source-specific dataset into an iterable
of UnifiedSample records. The registry loads adapters by module path
from data_sources.yaml.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from carl_studio.data.types import DataSource, UnifiedSample


class DataAdapter(ABC):
    """Transform a source-specific dataset into UnifiedSample records.

    Subclass and implement adapt() to support a new data source.
    The registry instantiates adapters by module path.
    """

    def __init__(self, source: DataSource) -> None:
        self.source = source

    @abstractmethod
    def adapt(self, raw: list[dict[str, Any]]) -> Iterator[UnifiedSample]:
        """Transform raw records (from HF datasets) into UnifiedSample.

        Args:
            raw: List of dicts from datasets.load_dataset().

        Yields:
            UnifiedSample for each valid record. Skip invalid records.
        """
        ...

    def load_and_adapt(self, split: str | None = None) -> list[UnifiedSample]:
        """Load from HuggingFace and adapt in one step."""
        from datasets import load_dataset

        ds = load_dataset(
            self.source.repo_id,
            split=split or self.source.default_split,
        )
        return list(self.adapt([dict(row) for row in ds]))
