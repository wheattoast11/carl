"""Prepared experiment inputs and independent model acceptance."""

from __future__ import annotations

import math
from typing import Literal

from carl_core.constants import SIGMA
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from carl_studio.experiment.types import Artifact, Hypothesis


class RewardBinding(BaseModel):
    """A trusted reward callable and its explicit training weight."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    reference: str = "verification"
    weight: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    stage: Literal["A", "B"] = "A"


class PolicyCheck(BaseModel):
    """A measurable output policy, distinct from execution permissions."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    threshold: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)


class TrainingGoal(BaseModel):
    """Task progress, policy checks and full-logit coherence acceptance."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    description: str = Field(default="Improve held-out task success", min_length=1)
    primary_metric: str = "task_success_rate"
    evaluator: str = "verification"
    direction: Literal["higher", "lower"] = "higher"
    threshold: float = Field(default=0.5, allow_inf_nan=False)
    min_delta: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    rewards: list[RewardBinding] = Field(default_factory=lambda: [RewardBinding()])
    policies: list[PolicyCheck] = Field(default_factory=lambda: list[PolicyCheck]())
    coherence_phi_floor: float = Field(default=SIGMA, ge=0.0, le=1.0)
    discontinuity_min: float = Field(default=0.3, ge=0.0, le=1.0)
    discontinuity_max: float = Field(default=0.7, ge=0.0, le=1.0)
    max_eval_samples: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def validate_goal(self) -> TrainingGoal:
        if not self.rewards:
            raise ValueError("At least one task reward is required")
        if self.discontinuity_min > self.discontinuity_max:
            raise ValueError("Invalid coherence band")
        if len({p.id for p in self.policies}) != len(self.policies):
            raise ValueError("Policy IDs must be unique")
        return self


class SourceBinding(BaseModel):
    """Exact local source bytes used by a prepared experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["config", "model", "resume", "train_data", "eval_data", "callable", "artifact"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReadinessIssue(BaseModel):
    """A concrete preparation requirement and its recovery action."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str
    action: str


class TrainingPreparation(BaseModel):
    """Owner-readable prepared inputs, without starting a training job."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    plan_id: str
    project_root: str
    config: dict[str, object]
    goal: TrainingGoal
    hypothesis: Hypothesis
    sources: list[SourceBinding] = Field(default_factory=lambda: list[SourceBinding]())
    artifacts: list[Artifact] = Field(default_factory=lambda: list[Artifact]())
    train_samples: int = 0
    eval_samples: int = 0
    eval_sample_ids: list[str] = Field(default_factory=list)
    issues: list[ReadinessIssue] = Field(default_factory=lambda: list[ReadinessIssue]())
    variants: dict[str, str] = Field(default_factory=dict)
    capabilities: dict[str, object] = Field(default_factory=dict)
    effects: list[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.issues and self.train_samples > 0 and self.eval_samples > 0


class EvaluationMeasurement(BaseModel):
    """Comparable measurements over a bound held-out population."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    checkpoint: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_ids: list[str]
    primary_metric: str
    primary_value: float = Field(allow_inf_nan=False)
    metrics: dict[str, float] = Field(default_factory=dict)
    coherence: dict[str, float] | None = None
    coherence_source: Literal["full_logits", "unavailable"] = "unavailable"
    policy_results: dict[str, float] = Field(default_factory=dict)
    generation: dict[str, object] = Field(default_factory=dict)

    @field_validator("metrics", "coherence", "policy_results")
    @classmethod
    def finite_values(cls, values: dict[str, float] | None) -> dict[str, float] | None:
        if values is not None and any(not math.isfinite(v) for v in values.values()):
            raise ValueError("Measurements must be finite")
        return values


class TrainingAcceptance(BaseModel):
    """Candidate acceptance does not imply publication or execution success."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["accepted", "rejected", "inconclusive"]
    reasons: list[str]
    baseline: EvaluationMeasurement
    candidate: EvaluationMeasurement
    goal_delta: float
    policy_passed: bool
    coherence_passed: bool
