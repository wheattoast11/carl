"""Env-setup rolling state — JSON-serializable for resume.

State is append-only across questions: each answered question mutates
exactly one field, leaving prior answers intact. Serialized at
``~/.carl/last_env_state.json`` between sessions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EnvState(BaseModel):
    """Accumulated answers from a ``carl env`` session.

    All fields optional; the render path handles "not answered yet" by
    substituting defaults from ``CARLSettings`` or emitting TODO markers.
    """

    mode: str | None = Field(
        default=None,
        description="'train' | 'infer' | 'both' — top-level routing decision",
    )
    method: str | None = Field(
        default=None,
        description="'sft' | 'grpo' | 'dpo' | 'cascade' — when mode includes train",
    )
    dataset: str | None = Field(
        default=None,
        description="HuggingFace repo id OR path to local JSONL",
    )
    dataset_kind: str | None = Field(
        default=None,
        description="'hf' | 'local' | 'synth' — how to resolve the dataset field",
    )
    compute: str | None = Field(
        default=None,
        description="'local' | 'cloud' | preset name (e.g., 'a100-largex4')",
    )
    base_model: str | None = Field(
        default=None,
        description="HF model id; when None, renderer uses CARLSettings.default_model",
    )
    # v0.14 expanded question set
    reward: str | None = Field(
        default=None,
        description="'static' | 'phase_adaptive' | 'custom' | 'none' — GRPO reward shape",
    )
    cascade_stages: int | None = Field(
        default=None,
        description="Number of cascade stages (1-3) when method=cascade",
    )
    eval_gate: str | None = Field(
        default=None,
        description="'none' | 'metric' | 'crystallization' — eval admission policy",
    )
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json_path(self, path: Path) -> None:
        """Write state to ``path`` as JSON for resume."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json_path(cls, path: Path) -> "EnvState":
        """Load state from disk. Fresh state when file missing/bad."""
        if not path.is_file():
            return cls()
        try:
            return cls.model_validate_json(path.read_text())
        except Exception:
            return cls()

    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when enough fields are populated to render a valid config.

        Gate shape:
        * infer mode → just mode required.
        * train/both → mode + method + dataset + compute. The expanded
          v0.14 fields (reward, cascade_stages, eval_gate) have
          sensible defaults in the renderer, so they're not required
          for completeness — users can skip them.
        """
        if self.mode in (None, ""):
            return False
        if self.mode == "infer":
            return True  # inference-only doesn't need method/dataset
        # Train or both: need method + dataset + compute
        return all([self.method, self.dataset, self.compute])

    def as_context(self) -> dict[str, Any]:
        """Dict view for question-function composition."""
        return self.model_dump()


__all__ = ["EnvState"]
