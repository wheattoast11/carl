"""Synthesizer skill — generate graded RL training samples from a codebase source."""
from __future__ import annotations

from typing import Any

from carl_studio.skills.base import BaseSkill, SkillResult


class SynthesizerSkill(BaseSkill):
    name = "synthesizer"
    badge = "Data Badge"
    description = (
        "Synthesize graded RL training samples from a codebase source. "
        "badge_earned=True when count>0 valid samples generated."
    )
    requires_tier = "free"

    def execute(  # type: ignore[override]
        self,
        *,
        source: str,
        count: int = 10,
        output: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        try:
            from carl_studio.console import get_console
            from carl_studio.learn.synthesize import SynthesizeConfig, SynthesizePipeline

            config = SynthesizeConfig(source=source, count=count, output=output)
            result = SynthesizePipeline(config, get_console()).run()
            generated: int = getattr(result, "generated", count)
            return SkillResult(
                skill_name=self.name,
                success=True,
                badge_earned=generated > 0,
                metrics={"generated": generated, "source": source},
                message=f"Generated {generated} samples from {source}",
                artifact={"output_path": output or "", "count": generated},
            )
        except Exception as exc:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=str(exc),
            )
