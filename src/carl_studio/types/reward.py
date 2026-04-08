"""
Reward function protocol for carl-studio training loops.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardFunction(Protocol):
    """
    Structural typing protocol for reward functions.

    Any class with name, weight, and compute() is a valid RewardFunction.
    """

    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    def compute(self, completions: list, **kwargs: object) -> list[float]: ...
