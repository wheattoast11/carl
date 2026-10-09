"""Shared encoder, interpretation, and representation contracts."""

from __future__ import annotations

import math
from typing import Literal

from carl_core.hashing import content_hash
from pydantic import BaseModel, ConfigDict, Field, model_validator

Modality = Literal["text", "structured", "image", "audio", "video"]
RUNGS = (128, 256, 512, 768)
GEMMA_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


class Contract(BaseModel):
    """Immutable, closed wire contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class EncoderBinding(Contract):
    """Space identity excludes execution and retrieval view identities."""

    model: str = "google/embeddinggemma-2"
    revision: str = GEMMA_REVISION
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_dimension: int = Field(default=768, ge=1, le=16384)
    dimensions: tuple[int, ...] = RUNGS
    modalities: tuple[Modality, ...] = ("text", "structured", "image", "audio", "video")
    recipes: tuple[str, ...] = ("SearchQuery", "Document", "SentenceSimilarity", "Classification")
    adaptation: tuple[str, ...] = ("frozen_heads", "adapter")

    @model_validator(mode="after")
    def validate_capabilities(self) -> EncoderBinding:
        if (
            not self.dimensions
            or self.native_dimension not in self.dimensions
            or tuple(sorted(set(self.dimensions))) != self.dimensions
            or any(d < 1 or d > self.native_dimension for d in self.dimensions)
        ):
            raise ValueError("Invalid encoder dimension capabilities")
        return self

    @property
    def space_id(self) -> str:
        return content_hash(
            self.model_dump(
                mode="json",
                include={
                    "model",
                    "revision",
                    "artifact_sha256",
                    "processor_sha256",
                    "native_dimension",
                },
            )
        )


class ExecutionBinding(Contract):
    """Actual interpreter, dependency, processor and trainable-module identities."""

    interpreter: str
    interpreter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependencies: dict[str, str]
    processor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trainable_modules: tuple[str, ...] = ()
    device: str = "cpu"
    dtype: str = "float32"

    @property
    def execution_id(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class SemanticPart(Contract):
    """Ordered source part; media uses vault references rather than payloads."""

    modality: Modality
    text: str | None = Field(default=None, max_length=65536)
    source_ref: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_schema: str | None = None
    start_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_part(self) -> SemanticPart:
        if self.modality in {"text", "structured"}:
            if not self.text:
                raise ValueError("Text and structured recipes require content")
            if self.modality == "structured" and not self.source_schema:
                raise ValueError("Structured input requires its source schema")
        elif self.text is not None or not self.source_ref or not self.source_sha256:
            raise ValueError("Media requires a source reference and hash")
        if self.start_s is not None and self.modality not in {"audio", "video"}:
            raise ValueError("Only audio and video have time windows")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("A segment requires both window endpoints")
        if self.start_s is not None and self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("Invalid source window")
        return self


class SemanticInput(Contract):
    """One semantic event, preserving interleaving and source relationships."""

    event_id: str = Field(min_length=1)
    parts: tuple[SemanticPart, ...] = Field(min_length=1, max_length=64)
    recipe: str = "Document"
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    audio_sample_rate: Literal[16000] = 16000
    audio_channels: Literal[1] = 1
    video_fps: Literal[1] = 1
    vision_budget: Literal["model_default"] = "model_default"

    @property
    def content_id(self) -> str:
        return content_hash(self.model_dump(mode="json", exclude={"event_id"}))


class RepresentationRef(Contract):
    """A full carrier artifact with its event and derivation bindings."""

    event_id: str
    content_id: str
    space_id: str
    execution_id: str
    recipe: str
    artifact: dict[str, object]
    generation: str
    dimensions: int = 768


class Carrier(Contract):
    """Raw full representation; normalized views do not overwrite its tail."""

    values: tuple[float, ...] = Field(min_length=1, max_length=16384)
    dimensions: tuple[int, ...] = RUNGS

    @model_validator(mode="after")
    def admit(self) -> Carrier:
        if any(not math.isfinite(v) for v in self.values):
            raise ValueError("Nonfinite representation")
        if (
            not self.dimensions
            or max(self.dimensions) != len(self.values)
            or tuple(sorted(set(self.dimensions))) != self.dimensions
            or any(d < 1 for d in self.dimensions)
        ):
            raise ValueError("Invalid carrier dimensions")
        for dimension in self.dimensions:
            if math.hypot(*self.values[:dimension]) <= 1e-12:
                raise ValueError("Collapsed scoring prefix")
        return self

    def prefix(self, dimension: int) -> tuple[float, ...]:
        if dimension not in self.dimensions:
            raise ValueError("Unsupported resolution")
        return self.values[:dimension]

    def scoring(self, dimension: int) -> tuple[float, ...]:
        raw = self.prefix(dimension)
        norm = math.hypot(*raw)
        return tuple(v / norm for v in raw)

    def score(self, other: Carrier, dimension: int) -> float:
        return sum(a * b for a, b in zip(self.scoring(dimension), other.scoring(dimension)))


class Interpretation(Contract):
    """Similarity, speaker confirmation and action outcome remain independent."""

    id: str
    event_id: str
    utterance_ref: str
    context_ref: str
    goal_ref: str
    proposal_ref: str
    revision: int = Field(default=1, ge=1)
    supersedes: str | None = None
    feedback: Literal["unconfirmed", "confirmed", "rejected"] = "unconfirmed"
    feedback_ref: str | None = None
    confirmed_artifact: str | None = None
    outcome_ref: str | None = None
    action_success: bool | None = None
    similarity: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    compatible_observations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_confirmation(self) -> Interpretation:
        if self.feedback != "unconfirmed" and not self.feedback_ref:
            raise ValueError("Explicit feedback reference is required")
        if self.feedback == "confirmed" and not self.confirmed_artifact:
            raise ValueError("Confirmation requires an artifact")
        return self

    @property
    def commit_key(self) -> str:
        if self.feedback != "confirmed":
            raise ValueError("Only confirmed interpretations can consolidate")
        return content_hash(
            {
                k: getattr(self, k)
                for k in ("event_id", "context_ref", "goal_ref", "revision", "confirmed_artifact")
            }
        )


class RecallResult(Contract):
    """Lexical and semantic measurements with explicit retrieval mode."""

    source_ref: str
    lexical_score: float
    semantic_score: float | None = None
    resolution: int | None = None
    mode: Literal["lexical", "semantic"]
