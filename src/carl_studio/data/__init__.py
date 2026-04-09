"""CARL data pipeline — unified dataset types, adapters, and curriculum.

The type system IS the formalism. Domain, Modality, and DifficultyTier
encode the bifurcated distinctions. DataSource and UnifiedSample are the
primitive abstractions. Adapters transform source-specific formats into
the universal schema. The curriculum engine filters by difficulty.
"""

from carl_studio.data.types import (
    DataSource,
    DifficultyTier,
    Domain,
    EnvironmentSpec,
    FileRef,
    Modality,
    UnifiedSample,
)

__all__ = [
    "DataSource",
    "DifficultyTier",
    "Domain",
    "EnvironmentSpec",
    "FileRef",
    "Modality",
    "UnifiedSample",
]
