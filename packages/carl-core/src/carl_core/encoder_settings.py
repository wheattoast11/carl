"""Bounded encoder configuration available without worker extras."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .encoder import Contract, ExecutionBinding


class EncoderSettings(Contract):
    """Bounded local encoder learning settings; capture grants are separate."""

    mode: Literal["frozen_heads", "adapter"] = "frozen_heads"
    interpreter: str
    execution: ExecutionBinding | None = None
    rank: int = Field(default=8, ge=1, le=64)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0, ge=0, lt=1, allow_inf_nan=False)
    temperature: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    optimizer_steps: int = Field(default=128, ge=1, le=128)
    microbatch: int = Field(default=4, ge=1, le=4)
    accumulation: int = Field(default=4, ge=1, le=4)
    processed_tokens: int = Field(default=2048, ge=1, le=2048)
    seed: int = 42
    memory_gib: float = Field(default=12, gt=0, le=12, allow_inf_nan=False)
    runtime_s: float = Field(default=900, gt=0, le=900, allow_inf_nan=False)
    validation_dataset: str
    cutoff: str
    activate_workspace: str | None = None
    action_evaluator: str | None = None
    required_slices: tuple[str, ...] = ("text:128", "text:256", "text:512", "text:768")
    required_policies: tuple[str, ...] = ()
    embedding_cache: str | None = None
    head_layout: Literal["shared", "per_rung"] = "shared"
    relation_weight: float = Field(default=1, ge=0, allow_inf_nan=False)
    balanced_sampling: bool = True
    validation_every_steps: int = Field(default=16, ge=1)
    early_stop_patience: int = Field(default=3, ge=1)
    checkpoint_every_steps: int = Field(default=16, ge=1)
    encode_batch_size: int = Field(default=4, ge=1, le=4)
