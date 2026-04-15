"""Grader skill — run the CARL eval gate on a checkpoint."""
from __future__ import annotations

from typing import Any

from carl_studio.skills.base import BaseSkill, SkillResult


class GraderSkill(BaseSkill):
    name = "grader"
    badge = "Grader Badge"
    description = "Evaluate a checkpoint with the CARL eval gate. badge_earned=True on PASS."
    requires_tier = "free"

    def execute(  # type: ignore[override]
        self,
        *,
        checkpoint: str,
        base_model: str = "Tesslate/OmniCoder-9B",
        phase: str = "auto",
        threshold: float = 0.30,
        **kwargs: Any,
    ) -> SkillResult:
        try:
            from carl_studio.eval.runner import EvalConfig, EvalRunner

            config = EvalConfig(
                checkpoint=checkpoint,
                base_model=base_model,
                phase=phase,
                threshold=threshold,
            )
            runner = EvalRunner(config)
            report = runner.run()
            return SkillResult(
                skill_name=self.name,
                success=True,
                badge_earned=report.passed,
                metrics=report.metrics,
                message=(
                    f"Phase {report.phase}: "
                    f"{'PASS' if report.passed else 'FAIL'} "
                    f"({report.primary_metric}={report.primary_value:.2%})"
                ),
                artifact={
                    "checkpoint": checkpoint,
                    "phase": report.phase,
                    "passed": report.passed,
                },
            )
        except ImportError:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message="Install carl-studio[training] for eval",
            )
        except Exception as exc:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=str(exc),
            )
