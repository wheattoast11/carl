"""
Training configuration types for carl-studio.

All configuration surfaces are Pydantic BaseModel with Field validators.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ComputeTarget(str, Enum):
    """Hardware target for training jobs."""

    L4X1 = "l4x1"
    L4X4 = "l4x4"
    A10G_LARGE = "a10g-large"
    A10G_LARGEX2 = "a10g-largex2"
    A10G_LARGEX4 = "a10g-largex4"
    A100_LARGE = "a100-large"
    A100_LARGEX2 = "a100-largex2"
    A100_LARGEX4 = "a100-largex4"
    A100_LARGEX8 = "a100-largex8"
    L40SX1 = "l40sx1"
    L40SX4 = "l40sx4"
    L40SX8 = "l40sx8"
    LOCAL = "local"


COMPUTE_TARGET_ALIASES: dict[str, str] = {
    "a10g": ComputeTarget.A10G_LARGE.value,
    "a100": ComputeTarget.A100_LARGE.value,
}


def normalize_compute_target(value: str) -> str:
    """Normalize common compute target aliases to canonical enum values."""
    normalized = value.strip().lower()
    return COMPUTE_TARGET_ALIASES.get(normalized, normalized)


class TrainingMethod(str, Enum):
    """Training algorithm."""

    SFT = "sft"
    GRPO = "grpo"
    DPO = "dpo"
    KTO = "kto"
    ORPO = "orpo"


class ScaleWeightProfile(str, Enum):
    """
    How to weight multi-scale coherence in the observation pipeline.

    UNIFORM:   All scales weighted equally.
    FINE_FIRST: Fine scales (j=0,1,2) weighted higher -- correct convergence order.
    COARSE_FIRST: Coarse scales weighted higher -- hallucination risk detector.
    ADAPTIVE:  Weights shift based on training phase.
    """

    UNIFORM = "uniform"
    FINE_FIRST = "fine_first"
    COARSE_FIRST = "coarse_first"
    ADAPTIVE = "adaptive"


class LoraConfig(BaseModel):
    """LoRA adapter configuration."""

    r: int = Field(default=64, ge=1, le=512, description="LoRA rank")
    alpha: int = Field(default=128, ge=1, le=1024, description="LoRA alpha")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Which modules to apply LoRA to",
    )
    bias: str = Field(default="none", description="Bias handling: none, all, lora_only")

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        allowed = {"none", "all", "lora_only"}
        if v not in allowed:
            raise ValueError(f"bias must be one of {allowed}, got {v!r}")
        return v


class QuantizationConfig(BaseModel):
    """Quantization settings for model loading."""

    load_in_8bit: bool = Field(default=True, description="Load model in 8-bit quantization")
    load_in_4bit: bool = Field(
        default=False, description="Load model in 4-bit quantization (QLoRA)"
    )
    bnb_4bit_compute_dtype: str = Field(default="bfloat16", description="Compute dtype for 4-bit")
    bnb_4bit_quant_type: str = Field(default="nf4", description="4-bit quant type: nf4 or fp4")

    @model_validator(mode="after")
    def validate_exclusive_quant(self) -> "QuantizationConfig":
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot enable both 8-bit and 4-bit quantization simultaneously")
        return self


class CascadeStage(BaseModel):
    """One stage of a GRPO cascade (Nemotron Cascade 2 methodology)."""

    name: str = Field(description="Stage identifier (e.g. 'format_only', 'tool_selection')")
    steps: int = Field(ge=1, description="Number of training steps in this stage")
    active_rewards: List[str] = Field(description="Reward function names active in this stage")
    learning_rate: Optional[float] = Field(
        default=None, ge=0.0, description="Override LR for this stage"
    )


class ObservationConfig(BaseModel):
    """Configuration for the coherence observation pipeline."""

    enabled: bool = Field(default=True, description="Enable coherence probing")
    probe_every: int = Field(default=1, ge=1, description="Probe every N steps")
    observe_every: int = Field(default=50, ge=1, description="Send to observer every N snapshots")
    scale_weight_profile: ScaleWeightProfile = Field(
        default=ScaleWeightProfile.FINE_FIRST,
        description="Multi-scale weighting strategy",
    )
    discontinuity_alert_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Discontinuity score above which to raise an alert",
    )
    window_size: int = Field(default=100, ge=10, description="Snapshot buffer size for observer")


class TrainingConfig(BaseModel):
    """Full training configuration for a carl-studio run."""

    # Identity
    run_name: str = Field(description="Human-readable run name")
    base_model: str = Field(description="HuggingFace model ID (e.g. 'Tesslate/OmniCoder-9B')")
    output_repo: str = Field(
        description="HuggingFace repo to push results (e.g. 'wheattoast11/OmniCoder-9B-Zero-Phase1')"
    )

    # Method
    method: TrainingMethod = Field(description="Training algorithm")
    compute_target: ComputeTarget = Field(default=ComputeTarget.L4X1, description="Hardware target")

    # Data
    dataset_repo: str = Field(description="HuggingFace dataset repo")
    dataset_split: str = Field(default="train", description="Dataset split to use")
    eval_dataset_repo: Optional[str] = Field(default=None, description="Eval dataset (if separate)")
    eval_split: str = Field(default="test", description="Eval dataset split")

    # Hyperparameters
    num_train_epochs: int = Field(default=3, ge=1, le=100)
    max_steps: int = Field(default=-1, description="-1 for epoch-based training")
    per_device_train_batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=256)
    learning_rate: float = Field(default=2e-5, gt=0.0, le=1.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    max_length: int = Field(default=512, ge=1, le=131072, description="Max sequence length")
    lr_scheduler_type: str = Field(default="cosine")
    bf16: bool = Field(default=True)
    seed: int = Field(default=42)

    # GRPO-specific
    num_generations: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Rollouts per prompt (GRPO). 8 recommended for reward diversity.",
    )
    max_completion_length: int = Field(
        default=512,
        ge=1,
        le=131072,
        description="Max tokens for GRPO completions. Must fit actual output (128+ for coords, 512 for tool-calls).",
    )
    beta: float = Field(default=0.0, ge=0.0, description="KL penalty coefficient")
    cascade_stages: Optional[List[CascadeStage]] = Field(
        default=None, description="GRPO cascade stage definitions"
    )
    disable_thinking: bool = Field(
        default=True,
        description="Disable Qwen3.5 thinking mode (generates <think>...</think> prefix). Required for coordinate/short-output tasks.",
    )

    # VLM (Vision Language Model)
    vlm_mode: bool = Field(
        default=False,
        description="Use AutoModelForImageTextToText instead of AutoModelForCausalLM. Required for image inputs.",
    )
    grpo_temperature: float = Field(
        default=2.0,
        ge=0.1,
        le=5.0,
        description="Generation temperature for GRPO. 2.0 needed to break mode collapse at high Phi.",
    )
    grpo_top_k: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Top-k for GRPO generation. Safety net against garbage at high temperature.",
    )

    # Adapter
    lora: LoraConfig = Field(default_factory=LoraConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

    # Observation
    observation: ObservationConfig = Field(default_factory=ObservationConfig)

    # Hub
    hub_strategy: str = Field(default="every_save", description="When to push to hub")
    push_to_hub: bool = Field(default=True)
    hub_private: bool = Field(default=False)
    timeout: str = Field(default="3h", description="Job timeout string (e.g. '3h', '6h')")

    # Tokenizer
    tokenizer_source: Optional[str] = Field(
        default=None,
        description="Override tokenizer source if base model repo is missing chat template",
    )

    # Extra
    extra_args: Dict[str, str] = Field(
        default_factory=dict,
        description="Passthrough arguments for the training script",
    )

    @field_validator("lr_scheduler_type")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        allowed = {
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        }
        if v not in allowed:
            raise ValueError(f"lr_scheduler_type must be one of {allowed}, got {v!r}")
        return v

    @field_validator("hub_strategy")
    @classmethod
    def validate_hub_strategy(cls, v: str) -> str:
        allowed = {"end", "every_save", "checkpoint", "all_checkpoints"}
        if v not in allowed:
            raise ValueError(f"hub_strategy must be one of {allowed}, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_cascade_method(self) -> "TrainingConfig":
        if self.cascade_stages and self.method != TrainingMethod.GRPO:
            raise ValueError("cascade_stages only valid with GRPO training method")
        return self
