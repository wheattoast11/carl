"""Trainer skill — start a CARL training run."""
from __future__ import annotations

from typing import Any

from carl_studio.skills.base import BaseSkill, SkillResult


class TrainerSkill(BaseSkill):
    name = "trainer"
    badge = "Trainer Badge"
    description = "Start a CARL training run. badge_earned=True when job submits successfully."
    requires_tier = "free"

    def execute(  # type: ignore[override]
        self,
        *,
        base_model: str,
        method: str = "grpo",
        compute: str = "l40sx1",
        max_steps: int = 100,
        dataset: str = "",
        output_repo: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        try:
            import anyio

            from carl_studio.training.trainer import CARLTrainer
            from carl_studio.types.config import TrainingConfig

            config_kwargs: dict[str, Any] = dict(
                base_model=base_model,
                method=method,
                compute_target=compute,
                max_steps=max_steps,
            )
            if dataset:
                config_kwargs["dataset_repo"] = dataset
            if output_repo:
                config_kwargs["output_repo"] = output_repo

            config = TrainingConfig(**config_kwargs)
            trainer = CARLTrainer(config)
            run = anyio.run(trainer.train)
            job_id: str = getattr(run, "hub_job_id", None) or getattr(run, "id", "") or ""
            return SkillResult(
                skill_name=self.name,
                success=True,
                badge_earned=bool(job_id),
                metrics={
                    "run_id": getattr(run, "id", ""),
                    "phase": getattr(getattr(run, "phase", None), "value", ""),
                },
                message=f"Run submitted: {job_id}",
                artifact={
                    "run_id": getattr(run, "id", ""),
                    "hub_job_id": getattr(run, "hub_job_id", ""),
                },
            )
        except ImportError:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message="Install carl-studio[training] for training",
            )
        except Exception as exc:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=str(exc),
            )
