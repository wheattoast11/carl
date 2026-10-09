"""Terminal privacy penalties outside cascade masks and reward weights."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from carl_studio.eval.runner import PrivacyBoundaryGate
from carl_studio.training.rewards.base import extract_text


def privacy_rewards(completions: list[Any], **kwargs: Any) -> list[float]:
    """Audit per-completion private identifiers and optional event trajectories."""
    keys = kwargs.get("private_keys")
    if keys is None:
        return [0.0] * len(completions)
    if not isinstance(keys, list) or len(cast(list[Any], keys)) != len(completions):
        raise ValueError("private_keys must have one identifier list per completion")
    trajectories = kwargs.get("trajectory")
    if trajectories is not None and (
        not isinstance(trajectories, list) or len(cast(list[Any], trajectories)) != len(completions)
    ):
        raise ValueError("trajectory must have one event list per completion")
    key_rows = cast(list[Any], keys)
    trajectory_rows = cast(list[Any] | None, trajectories)
    gate = PrivacyBoundaryGate()
    scores: list[float] = []
    for index, completion in enumerate(completions):
        identifiers: Any = key_rows[index]
        if not isinstance(identifiers, (list, set, tuple)) or any(
            not isinstance(key, str) or not key for key in cast(list[Any], identifiers)
        ):
            raise ValueError("Private draft identifiers must be nonempty strings")
        events: Any = trajectory_rows[index] if trajectory_rows is not None else []
        if not isinstance(events, list):
            raise TypeError("A trajectory must be a list of events")
        public = {"phase": "public", "completion": extract_text(completion)}
        scores.append(
            gate.terminal_reward(
                [*cast(list[dict[str, Any]], events), public], set(cast(list[str], identifiers))
            )
        )
    return scores


def suppress_private_rewards(function: Callable[..., list[float]]) -> Callable[..., list[float]]:
    """Zero task and coherence contributions for completions with privacy violations."""

    def score(completions: list[Any], **kwargs: Any) -> list[float]:
        penalties = privacy_rewards(completions, **kwargs)
        values = function(completions=completions, **kwargs)
        if len(values) != len(penalties):
            raise ValueError("Reward population mismatch")
        return [0.0 if penalty else value for value, penalty in zip(values, penalties)]

    score.__name__ = getattr(function, "__name__", "reward")
    return score
