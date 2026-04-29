"""AnticipatoryReward: realizability-gap reward shaping (v0.19.2).

Thin studio wrapper around carl_core.realizability_gap. Closes the
realizability loop named in design doc §4.5 / §10: trajectories whose
forecast at time t matched outcome at t+τ get high reward; mismatched
forecasts get penalized.

Pure numpy — no torch dep. The PAID `[anticipatory]` extra is reserved
for the learned subspace-prior MLP head + torch-based fractal
forecaster wiring (deferred to v0.19.2.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from carl_core import (
    CoherenceForecast,
    realizability_chain,
    realizability_gap,
)


@dataclass(frozen=True, eq=False)
class AnticipatoryRewardComponents:
    """Per-call metrics from AnticipatoryReward."""

    realizability_gap: float
    scale: float
    baseline: float
    horizon: int | None


@dataclass(frozen=True, eq=False)
class AnticipatoryReward:
    """Reward = baseline - scale * realizability_gap(forecast, observed).

    A vanishing realizability gap is a constructive proof that the forecast
    was sound (Kleene/Curry-Howard). Reward shaping pushes the policy
    toward forecasts whose realization matches the prediction.

    Attributes:
        scale: Multiplier on the gap (default 1.0). Larger ⇒ harsher
            penalty on mismatch.
        baseline: Constant reward floor when gap == 0 (default 0.0).
        clip_min: Optional lower bound on the returned reward (None ⇒ no clip).
        clip_max: Optional upper bound on the returned reward.
    """

    scale: float = 1.0
    baseline: float = 0.0
    clip_min: float | None = None
    clip_max: float | None = None

    def __call__(
        self,
        forecast: CoherenceForecast,
        observed_phi: NDArray[np.float64] | Sequence[float],
        *,
        horizon: int | None = None,
    ) -> tuple[float, AnticipatoryRewardComponents]:
        """Compute reward + metadata for one (forecast, observed) pair."""
        gap = realizability_gap(forecast, observed_phi, horizon=horizon)
        reward = self.baseline - self.scale * gap
        if self.clip_min is not None:
            reward = max(reward, self.clip_min)
        if self.clip_max is not None:
            reward = min(reward, self.clip_max)
        return float(reward), AnticipatoryRewardComponents(
            realizability_gap=gap,
            scale=self.scale,
            baseline=self.baseline,
            horizon=horizon,
        )

    def chain_reward(
        self,
        forecasts: Sequence[CoherenceForecast],
        observations: Sequence[NDArray[np.float64] | Sequence[float]],
        horizons: Sequence[int | None],
    ) -> tuple[float, AnticipatoryRewardComponents]:
        """Composed reward over a sequence: baseline - scale * Σ R(τ_i).

        By the data processing inequality, the chained gap upper-bounds
        the joint realizability cost, so this is a conservative reward
        — undershoots rather than overshoots true forecast quality.
        """
        total_gap = realizability_chain(forecasts, observations, horizons)
        reward = self.baseline - self.scale * total_gap
        if self.clip_min is not None:
            reward = max(reward, self.clip_min)
        if self.clip_max is not None:
            reward = min(reward, self.clip_max)
        return float(reward), AnticipatoryRewardComponents(
            realizability_gap=total_gap,
            scale=self.scale,
            baseline=self.baseline,
            horizon=None,
        )


def make_anticipatory_reward(
    *,
    scale: float = 1.0,
    baseline: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> AnticipatoryReward:
    """Factory matching the project's reward-construction convention."""
    return AnticipatoryReward(
        scale=scale, baseline=baseline, clip_min=clip_min, clip_max=clip_max
    )


__all__ = [
    "AnticipatoryReward",
    "AnticipatoryRewardComponents",
    "make_anticipatory_reward",
]
