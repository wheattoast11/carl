"""
Training run state models.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from carl_studio.experiment.types import Artifact
from carl_studio.types.config import TrainingConfig
from carl_studio.types.preparation import TrainingAcceptance


class RunPhase(str, Enum):
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    PROVISIONING = "provisioning"
    TRAINING = "training"
    OBSERVING = "observing"
    CHECKPOINTING = "checkpointing"
    PUSHING = "pushing"
    COMPLETE = "complete"
    FAILED = "failed"
    PAUSED = "paused"


class CoherenceHealth(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    TRANSITION = "transition"  # Coherence transition detected


class TrainingRun(BaseModel):
    """Represents the live state of a training run."""

    id: str = Field(description="Unique run identifier")
    config: TrainingConfig
    phase: RunPhase = RunPhase.INITIALIZING
    current_step: int = Field(default=0, ge=0)
    total_steps: int = Field(default=0, ge=0)
    phi_mean: float = 0.0
    discontinuity_density: float = 0.0
    cloud_quality: float = 0.0
    coherence_health: CoherenceHealth = CoherenceHealth.HEALTHY
    loss: float = 0.0
    reward_mean: float = 0.0
    error_message: Optional[str] = Field(
        default=None, description="Error details if phase == FAILED"
    )
    hub_job_id: Optional[str] = Field(default=None, description="HuggingFace Jobs job ID")
    checkpoint_steps: List[int] = Field(
        default_factory=list, description="Steps at which checkpoints were saved"
    )
    checkpoint: str | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    acceptance: TrainingAcceptance | None = None
    representation_acceptance: dict[str, object] | None = None
    optimizer_phase: Literal["pending", "complete", "stopped", "failed"] | None = None
    evaluation_phase: Literal["pending", "complete", "failed"] | None = None
    activation_phase: Literal["pending", "complete", "failed", "not_requested"] | None = None
    completion_custody: dict[str, str] = Field(default_factory=dict)
    resource_usage: dict[str, float] = Field(default_factory=dict)
