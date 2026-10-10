"""Encoder data splits, pilot settings and independent representation acceptance."""

from __future__ import annotations

import math
from typing import Literal

from carl_core.data_handles import DataRef
from carl_core.encoder_settings import EncoderSettings
from pydantic import Field, model_validator

from .types import Contract, SemanticInput


class EncoderExample(Contract):
    """Episode-bound contrastive example with explicit feedback provenance."""

    id: str
    episode_id: str = Field(min_length=1)
    original_sources: tuple[str, ...] = Field(min_length=1)
    observed_at: str
    feedback_ref: str = Field(min_length=1)
    query: SemanticInput
    positive: SemanticInput
    negatives: tuple[SemanticInput, ...] = Field(min_length=1)
    relation: Literal["supports", "contradicts", "corresponds", "corrects"]
    action_label: str = Field(min_length=1)
    task_family: str = Field(default="general", min_length=1)
    source_artifacts: dict[str, DataRef] = Field(default_factory=dict)

    @model_validator(mode="after")
    def explicit_negatives(self) -> EncoderExample:
        if any(n.content_id == self.positive.content_id for n in self.negatives):
            raise ValueError("A positive cannot be an explicit negative")
        for value in (self.query, self.positive, *self.negatives):
            for part in value.parts:
                if part.source_ref and (
                    part.source_ref not in self.source_artifacts
                    or self.source_artifacts[part.source_ref].sha256 != part.source_sha256
                ):
                    raise ValueError("Media source artifact is unbound")
        return self


def validate_splits(groups: dict[str, list[EncoderExample]], cutoff: str) -> None:
    """Reject episode/source leakage before derived views and future context."""
    if set(groups) != {"train", "validation", "test"} or any(not rows for rows in groups.values()):
        raise ValueError("Nonempty train, validation and test groups required")
    from datetime import datetime

    boundary = datetime.fromisoformat(cutoff)
    if boundary.tzinfo is None:
        raise ValueError("Temporal cutoff requires a timezone")
    seen_episodes: set[str] = set()
    seen_sources: set[str] = set()
    seen_content: set[str] = set()
    for rows in groups.values():
        episodes = {row.episode_id for row in rows}
        sources = {source for row in rows for source in row.original_sources}
        content = {
            part.content_id for row in rows for part in (row.query, row.positive, *row.negatives)
        }
        if seen_episodes & episodes or seen_sources & sources or seen_content & content:
            raise ValueError("Encoder split lineage overlap")
        if len({row.id for row in rows}) != len(rows):
            raise ValueError("Duplicate encoder example identity")
        for row in rows:
            observed = datetime.fromisoformat(row.observed_at)
            if observed.tzinfo is None:
                raise ValueError("Episode time requires a timezone")
            if observed > boundary or any(
                ref.created_at > boundary for ref in row.source_artifacts.values()
            ):
                raise ValueError("Future-context leakage")
        seen_episodes |= episodes
        seen_sources |= sources
        seen_content |= content


class RepresentationMeasurement(Contract):
    """Embedding-specific held-out evidence, independent of token Phi."""

    population: str
    processing: str
    tasks: str
    budget: str
    action_accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    dimensions: int
    finite: bool
    positive_negative_margin: float = Field(allow_inf_nan=False)
    variance: float = Field(ge=0, allow_inf_nan=False)
    slices: dict[str, float]
    policies: dict[str, bool]
    resources_passed: bool
    updated_parameters: tuple[str, ...]

    @model_validator(mode="after")
    def finite_slices(self) -> RepresentationMeasurement:
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.slices.values()):
            raise ValueError("Invalid retention measurement")
        return self


def representation_acceptance(
    baseline: RepresentationMeasurement,
    candidate: RepresentationMeasurement,
    required_slices: set[str],
    required_policies: set[str],
    *,
    threshold: float = 0.80,
    improvement: float = 0.05,
    retention_loss: float = 0.02,
) -> dict[str, object]:
    """Missing measurements are inconclusive; task progress cannot hide collapse."""
    reasons: list[str] = []
    if any(
        getattr(baseline, key) != getattr(candidate, key)
        for key in ("population", "processing", "tasks", "budget")
    ):
        return {"status": "inconclusive", "reasons": ["incomparable measurements"]}
    if (
        not required_slices
        or not required_slices <= baseline.slices.keys()
        or not required_slices <= candidate.slices.keys()
    ):
        return {"status": "inconclusive", "reasons": ["missing retention slices"]}
    if not required_policies <= candidate.policies.keys():
        return {"status": "inconclusive", "reasons": ["missing policy measurements"]}
    if candidate.dimensions != 768 or not candidate.finite:
        reasons.append("numerical admission failed")
    if candidate.variance <= 1e-12:
        reasons.append("collapsed representations")
    if candidate.positive_negative_margin <= 0:
        reasons.append("insufficient positive-negative discrimination")
    if not candidate.updated_parameters:
        reasons.append("unchanged candidate")
    if (
        candidate.action_accuracy < threshold
        or candidate.action_accuracy - baseline.action_accuracy <= improvement
    ):
        reasons.append("insufficient task improvement")
    if any(
        baseline.slices[key] - candidate.slices[key] > retention_loss for key in required_slices
    ):
        reasons.append("retention regression")
    if not candidate.resources_passed or not all(
        candidate.policies[key] for key in required_policies
    ):
        reasons.append("policy or resource limit failure")
    return {"status": "rejected" if reasons else "accepted", "reasons": reasons}


__all__ = [
    "EncoderExample",
    "EncoderSettings",
    "RepresentationMeasurement",
    "representation_acceptance",
    "validate_splits",
]
