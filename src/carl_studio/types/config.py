"""
Training configuration types for carl-studio.

All configuration surfaces are Pydantic BaseModel with Field validators.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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
    DPO = "dpo"  # Not yet implemented
    KTO = "kto"  # Not yet implemented
    ORPO = "orpo"  # Not yet implemented


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


class CascadeConfig(BaseModel):
    """Cascade gating configuration for GRPO + CARL runs.

    The cascade gates CARL reward activation behind Stage B. Two gate
    modes are supported:

    - ``metric`` (default): watches a scalar reward signal and fires on a
      running-percentile threshold. Self-calibrating but gameable by
      reward volume alone.
    - ``crystallization``: watches the CoherenceTrace field for
      discontinuity events (delta_phi > DEFECT_THRESHOLD). Recommended
      for GRPO + CARLReward runs — it is a structural property of the
      output distribution and cannot be gamed by volume.

    Defaults are chosen so existing fixed-cascade behavior is preserved
    when ``cascade`` is omitted from ``carl.yaml``.
    """

    carl_start: int = Field(
        default=50,
        ge=0,
        description="Step at which CARL (Stage B) activates in fixed mode",
    )
    warmup_steps: int = Field(
        default=10,
        ge=1,
        description="Linear-warmup window for the CARL weight after carl_start",
    )
    gate_mode: Literal["metric", "crystallization"] = Field(
        default="metric",
        description=(
            "Cascade gate mode. 'metric' watches a scalar reward signal; "
            "'crystallization' watches CoherenceTrace phase transitions "
            "(recommended for GRPO + CARLReward)."
        ),
    )
    n_crystallizations_required: int = Field(
        default=3,
        ge=1,
        description="Crystallization events required to fire the gate",
    )
    crystallization_window: int = Field(
        default=10,
        ge=1,
        description="Rolling window length for crystallization counting",
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
    base_model: str = Field(description="HuggingFace model ID (e.g. 'your-org/your-model')")
    output_repo: str = Field(
        description="HuggingFace repo to push results (e.g. 'your-org/your-model-phase1')"
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
    deterministic: bool = Field(
        default=False,
        description=(
            "Pin every randomness source (transformers.set_seed, torch.manual_seed, "
            "np.random.seed, CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED, "
            "torch.use_deterministic_algorithms) for bit-identical runs. When "
            "supported by the installed TRL version, also sets full_determinism=True "
            "on SFTConfig/GRPOConfig."
        ),
    )
    reward_class: Literal["static", "phase_adaptive"] = Field(
        default="static",
        description=(
            "Which CARL reward composite to use. 'static' -> CARLReward with "
            "constant 50/30/20 weights. 'phase_adaptive' -> PhaseAdaptiveCARLReward "
            "which shifts weights based on the detected Kuramoto-R phase."
        ),
    )

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
    cascade: CascadeConfig = Field(
        default_factory=CascadeConfig,
        description=(
            "Cascade gating for GRPO + CARL. See CascadeConfig — controls "
            "carl_start, warmup, gate_mode ('metric' vs 'crystallization'), "
            "and crystallization window/threshold."
        ),
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
    fast_inference: bool = Field(
        default=False,
        description="Enable vLLM fast inference for Unsloth (GRPO)",
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

    # Observability (passed through to TRL training_args)
    report_to: Literal["none", "trackio", "wandb", "mlflow", "tensorboard"] = Field(
        default="none",
        description=(
            "TRL metrics reporter. 'none' keeps training offline; 'trackio' is "
            "carl's preferred local dashboard. Passed through verbatim to "
            "SFTConfig/GRPOConfig as report_to."
        ),
    )
    trace_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Directory for CoherenceTraceCallback JSONL artifacts. "
            "Defaults to ~/.carl/runs/<run_id>/traces/ when unset."
        ),
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

    @model_validator(mode="before")
    @classmethod
    def _auto_select_crystallization_gate(cls, data: Any) -> Any:
        """Smart default: phase_adaptive implies crystallization gate.

        When ``reward_class == "phase_adaptive"`` AND the user did not
        explicitly configure ``cascade.gate_mode`` in the YAML/dict, default
        the gate to ``"crystallization"``. The ``CascadeRewardManager``
        docstring already describes crystallization gating as "recommended
        for GRPO + CARLReward runs" — so coupling makes the recommendation
        operational instead of advisory.

        Lives in a ``mode="before"`` validator (rather than ``mode="after"``)
        because we need to inspect the RAW input dict to distinguish "user
        omitted cascade.gate_mode" from "user explicitly asked for metric".
        By the time an ``after`` validator runs, every field has been
        defaulted and that distinction is lost.
        """
        if not isinstance(data, dict):
            return data
        # Explicit cast so pyright knows what we're indexing into.
        data_dict: dict[str, Any] = data  # type: ignore[assignment]
        if data_dict.get("reward_class") != "phase_adaptive":
            return data_dict
        cascade_raw: Any = data_dict.get("cascade")
        # Only inject when the cascade key is missing OR it's a dict that
        # does not explicitly pin gate_mode. If the user passed a
        # pre-built CascadeConfig instance we respect it verbatim.
        if cascade_raw is None:
            data_dict["cascade"] = {"gate_mode": "crystallization"}
        elif isinstance(cascade_raw, dict) and "gate_mode" not in cascade_raw:
            # Copy so we don't mutate caller-owned dicts.
            new_cascade: dict[str, Any] = dict(cascade_raw)  # type: ignore[arg-type]
            new_cascade["gate_mode"] = "crystallization"
            data_dict["cascade"] = new_cascade
        return data_dict

    @model_validator(mode="after")
    def validate_cascade_method(self) -> "TrainingConfig":
        if self.cascade_stages and self.method != TrainingMethod.GRPO:
            raise ValueError("cascade_stages only valid with GRPO training method")
        return self
