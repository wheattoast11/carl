"""Observer skill — reads training frames and assesses coherence health."""
from __future__ import annotations

from typing import Any

from carl_studio.skills.base import BaseSkill, SkillResult


class ObserverSkill(BaseSkill):
    name = "observer"
    badge = "Observer Badge"
    description = (
        "Observe training coherence dynamics. Reads from Trackio or local JSONL log. "
        "Returns health assessment and Phi trajectory summary."
    )
    requires_tier = "free"

    def execute(  # type: ignore[override]
        self,
        *,
        url: str = "",
        file: str = "",
        run: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        """Load frames and compute health. badge_earned=True when health is GREEN."""
        try:
            from carl_studio.observe.data_source import (
                FileSource,
                TrackioSource,
                normalize_trackio_space,
            )

            if file:
                src: FileSource | TrackioSource = FileSource(file)
            elif url:
                space = normalize_trackio_space(url)
                src = TrackioSource(space=space, run=run)
            else:
                return SkillResult(
                    skill_name=self.name,
                    success=False,
                    message="Provide url= or file=",
                )

            frames = src.poll()
            if not frames:
                return SkillResult(
                    skill_name=self.name,
                    success=False,
                    message="No data found",
                )

            phis = [f.phi for f in frames]
            rewards = [f.reward_mean for f in frames]
            phi_mean = sum(phis) / len(phis)
            zero_frac = sum(1 for r in rewards if abs(r) < 1e-6) / max(len(rewards), 1)

            if phi_mean > 0.1 and zero_frac < 0.2:
                health = "GREEN"
            elif zero_frac < 0.5:
                health = "YELLOW"
            else:
                health = "RED"

            return SkillResult(
                skill_name=self.name,
                success=True,
                badge_earned=(health == "GREEN"),
                metrics={
                    "phi_mean": phi_mean,
                    "n_frames": len(frames),
                    "zero_frac": zero_frac,
                    "health": health,
                },
                message=f"Health: {health} | Phi mean: {phi_mean:.4f} | {len(frames)} steps",
                artifact={"frames": len(frames), "health": health},
            )
        except Exception as exc:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=str(exc),
            )
