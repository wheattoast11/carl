"""Deployer skill — push a trained model to HuggingFace Hub with CARL metadata."""
from __future__ import annotations

from typing import Any

from carl_studio.skills.base import BaseSkill, SkillResult


class DeployerSkill(BaseSkill):
    name = "deployer"
    badge = "Deployment Badge"
    description = (
        "Push a trained model to HuggingFace Hub with CARL metadata. "
        "badge_earned=True on successful push."
    )
    requires_tier = "free"

    def execute(  # type: ignore[override]
        self,
        *,
        model_path: str,
        repo_id: str,
        base_model: str = "",
        method: str = "grpo",
        dataset: str = "",
        private: bool = False,
        **kwargs: Any,
    ) -> SkillResult:
        try:
            import anyio

            from carl_studio.hub.models import push_with_metadata

            url: str = anyio.run(
                push_with_metadata,
                model_path,
                repo_id,
                base_model,
                method,
                dataset,
                None,
                private,
            )
            return SkillResult(
                skill_name=self.name,
                success=True,
                badge_earned=True,
                metrics={"url": url, "repo": repo_id},
                message=f"Deployed to {url}",
                artifact={"url": url, "repo_id": repo_id},
            )
        except ImportError:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message="Install carl-studio[hf] for hub push",
            )
        except Exception as exc:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=str(exc),
            )
