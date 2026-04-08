"""
Training run state models.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from carl_studio.types.config import TrainingConfig


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
    error_message: Optional[str] = Field(default=None, description="Error details if phase == FAILED")
    hub_job_id: Optional[str] = Field(default=None, description="HuggingFace Jobs job ID")
    checkpoint_steps: List[int] = Field(default_factory=list, description="Steps at which checkpoints were saved")
