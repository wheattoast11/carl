"""Project configuration management for carl-studio.

A CARLProject wraps TrainingConfig with project-level metadata:
model identity, hardware target, data sources, integrations, and stack definition.
Persisted as carl.yaml in the working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field



class StackConfig(BaseModel):
    """User's environment and tooling context."""

    tools: list[str] = Field(default_factory=list, description="Tool protocols: mcp, a2a, langraph, ...")
    frameworks: list[str] = Field(default_factory=list, description="Frameworks: react, python, fastapi, ...")
    repos: list[str] = Field(default_factory=list, description="Linked codebases (URLs or local paths)")
    use_case: str = Field(default="", description="Natural language description of the training goal")


class CARLProject(BaseModel):
    """Top-level project configuration. Serialized to carl.yaml."""

    # Identity
    name: str = Field(default="my-carl-project", description="Project name")
    description: str = Field(default="", description="What this project trains")

    # Model
    base_model: str = Field(default="", description="HF model ID (e.g. your-org/your-model)")
    resume_from: str | None = Field(
        default=None,
        description="Existing LoRA adapter checkpoint to resume from (HF repo or local path)",
    )
    output_repo: str = Field(default="", description="HF repo for trained model")

    # Training framework (which adapter to route through)
    adapter: str = Field(
        default="trl",
        description=(
            "Training framework adapter: trl, unsloth, axolotl, tinker, atropos, slime. "
            "This is the name registered in carl_studio.adapters — NOT a compute substrate."
        ),
    )

    # Hardware
    compute_target: str = Field(default="local", description="Compute target (l4x1, l40sx1, a100-largex8, local)")
    compute_backend: str = Field(
        default="local",
        description="Compute orchestration: local, hf_jobs, runpod, prime, ssh",
    )

    # Data
    dataset_repo: str = Field(default="", description="HF dataset repo or local path")
    eval_dataset_repo: str | None = Field(default=None, description="Eval dataset (if separate)")

    # Training
    method: str = Field(default="grpo", description="Training method: sft, grpo, dpo, kto, orpo")
    max_steps: int = Field(default=300, description="Training steps")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    carl_enabled: bool = Field(default=True, description="Enable CARL coherence rewards")

    # Integrations
    hub_token_env: str = Field(default="HF_TOKEN", description="Env var name for HF token")
    tracking_url: str | None = Field(default=None, description="Trackio dashboard URL")

    # Stack
    stack: StackConfig = Field(default_factory=StackConfig)

    def to_training_config(self) -> dict[str, Any]:
        """Convert to TrainingConfig-compatible dict."""
        return {
            "run_name": self.name,
            "base_model": self.base_model,
            "output_repo": self.output_repo,
            "method": self.method,
            "adapter": self.adapter,
            "compute_target": self.compute_target,
            "compute_backend": self.compute_backend,
            "dataset_repo": self.dataset_repo,
            "eval_dataset_repo": self.eval_dataset_repo,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
        }


def load_project(path: Path | str = "carl.yaml") -> CARLProject:
    """Load project from YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Project file not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    return CARLProject(**raw)


def save_project(project: CARLProject, path: Path | str = "carl.yaml") -> None:
    """Save project to YAML file."""
    p = Path(path)
    data = project.model_dump(exclude_defaults=False)
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
