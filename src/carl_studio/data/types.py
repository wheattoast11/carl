"""Unified dataset types for the CARL data pipeline.

The type system encodes three bifurcated distinctions:
  1. Domain:     what kind of problem (swe, code, math, tool_calling, knowledge)
  2. Modality:   what sensory channels (text, code, vision, multimodal)
  3. Difficulty:  where on the learnable frontier (trivial → frontier)

UnifiedSample is the universal record. Every dataset adapter transforms
source-specific formats into this type. The curriculum engine filters by
difficulty tier. The cascade matches stage to difficulty.

Design principle: new sources are added by writing an adapter + YAML entry.
The type system is stable — it doesn't change when new sources appear.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums — the bifurcated distinctions
# ---------------------------------------------------------------------------

class Domain(str, Enum):
    """What kind of problem. Filterable first-class dimension."""
    SWE = "swe"                    # Real-repo software engineering (GitHub issues)
    COMPETITIVE = "competitive"    # Competitive programming (algorithmic)
    TOOL_CALLING = "tool_calling"  # Tool use in sandbox environment
    MATH = "math"                  # Mathematical reasoning
    INSTRUCTION = "instruction"    # General instruction following
    KNOWLEDGE = "knowledge"        # Domain-specific knowledge tasks


class Modality(str, Enum):
    """What sensory channels. Filterable first-class dimension."""
    TEXT = "text"
    CODE = "code"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class DifficultyTier(str, Enum):
    """Where on the learnable frontier.

    Tiers derived from sampled pass rate (Nemotron dynamic filtering):
      TRIVIAL:  >90% — no gradient (all-pass groups)
      EASY:     70-90% — warm-up, Stage A curriculum
      MEDIUM:   40-70% — sweet spot, maximum GRPO advantage variance
      HARD:     15-40% — emergence territory, requires multi-step reasoning
      FRONTIER: <15% — aspirational, seed for future capability
    """
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    FRONTIER = "frontier"

    @classmethod
    def from_pass_rate(cls, rate: float) -> DifficultyTier:
        """Classify difficulty from sampled pass rate (0.0–1.0)."""
        if rate > 0.9:
            return cls.TRIVIAL
        elif rate > 0.7:
            return cls.EASY
        elif rate > 0.4:
            return cls.MEDIUM
        elif rate > 0.15:
            return cls.HARD
        else:
            return cls.FRONTIER


# ---------------------------------------------------------------------------
# Structural types
# ---------------------------------------------------------------------------

class FileRef(BaseModel):
    """A file reference — path + content."""
    path: str
    content: str


class EnvironmentSpec(BaseModel):
    """How to set up the sandbox for this sample.

    For simple tasks: just initial_files.
    For SWE tasks: docker_image + repo + commit.
    """
    initial_files: list[FileRef] = Field(default_factory=list)
    docker_image: str | None = None
    repo: str | None = None
    commit: str | None = None
    setup_commands: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    """How to verify the solution.

    Adapters normalize source-specific verification into this type.
    The curriculum engine uses these to compute pass rates.
    """
    test_commands: list[str] = Field(default_factory=list)
    test_code: str | None = None
    expected_output: str | None = None
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    golden_patch: str | None = None


# ---------------------------------------------------------------------------
# UnifiedSample — the universal record
# ---------------------------------------------------------------------------

class UnifiedSample(BaseModel):
    """Universal dataset record.

    Every source-specific dataset is transformed into this type by its adapter.
    The type is stable — new sources don't change the schema, they implement
    a new adapter that maps their fields into these.

    Filterable by: source, domain, modality, difficulty_tier.
    """
    id: str = Field(description="Unique sample ID (source-specific)")
    prompt: list[dict[str, Any]] = Field(description="Chat-format messages [{role, content}]")
    problem_statement: str = Field(description="Raw problem text (human-readable)")
    domain: Domain
    modality: Modality

    # Verification (optional — not all sources have ground truth)
    verification: Verification = Field(default_factory=Verification)
    golden_solution: str | None = Field(default=None, description="Ground-truth solution (patch/code)")

    # Difficulty (computed by curriculum engine or provided by source)
    difficulty_tier: DifficultyTier | None = None
    difficulty_score: float | None = Field(default=None, ge=0.0, le=1.0, description="Sampled pass rate")

    # Environment (for sandbox-based tasks)
    environment: EnvironmentSpec | None = None
    context_files: list[FileRef] = Field(default_factory=list, description="Relevant repo files")

    # Provenance
    source: str = Field(description="Dataset origin (e.g. 'nemotron-cascade', 'r2e-gym')")
    max_context_tokens: int = Field(default=32768, description="Max prompt length tier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Source-specific overflow")


# ---------------------------------------------------------------------------
# DataSource — a registered data provider
# ---------------------------------------------------------------------------

class DataSource(BaseModel):
    """A registered data source in the CARL data registry.

    Defined in YAML (data_sources.yaml) or programmatically.
    Each source has an adapter that transforms its format into UnifiedSample.
    """
    name: str = Field(description="Human-readable name")
    repo_id: str = Field(description="HuggingFace dataset repo ID")
    adapter: str = Field(description="Python module path for adapter class")
    domain: Domain
    modality: Modality
    description: str = Field(default="")
    license: str = Field(default="unknown")
    size: int | None = Field(default=None, description="Approximate sample count")
    difficulty_range: tuple[float, float] | None = Field(
        default=None, description="Expected pass rate range (low, high)"
    )
    default_split: str = Field(default="train")
    tags: list[str] = Field(default_factory=list)
