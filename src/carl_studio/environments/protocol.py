"""Environment protocol for CARL training environments.

All CARL environments implement this protocol. TRL's environment_factory
auto-discovers public methods as tools. This protocol adds:
  - EnvironmentSpec: declarative metadata (lane, tools, reward type)
  - EnvironmentLane: MECE categories (code, query, retrieval, routing, infra, visual)
  - BaseEnvironment: common state tracking + turn history for CARL analysis
  - Validation: check TRL compatibility before submission

The protocol is deliberately minimal — complexity lives in reward functions
and dataset routing, not in environment classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EnvironmentLane(str, Enum):
    """MECE environment categories, partitioned by substrate + verification.

    Every agent use case maps to exactly one lane. Hybrid use cases
    (e.g., "write SQL then visualize") are compositions handled by
    MultiEnvironmentFactory.
    """
    CODE = "code"            # Filesystem + interpreter
    QUERY = "query"          # Database (SQL, dataframe)
    RETRIEVAL = "retrieval"  # Document corpus, search
    ROUTING = "routing"      # Conversation state machine
    INFRA = "infra"          # System state (containers, configs)
    VISUAL = "visual"        # Rendered UI (browser, desktop)


@dataclass(frozen=True)
class EnvironmentSpec:
    """Declarative environment metadata.

    Used by the training pipeline to configure rewards, dataset columns,
    multi-environment routing, and per-lane cascade gating.
    """
    lane: EnvironmentLane
    name: str                                    # e.g. "python-sandbox"
    tools: tuple[str, ...]                       # Tool method names
    max_turns: int = 10
    reward_type: str = "binary"                  # "binary", "continuous", "categorical"
    multimodal: bool = False
    system_prompt: str = ""
    dataset_columns: tuple[str, ...] = ()        # Required beyond 'prompt'


class BaseEnvironment(ABC):
    """Abstract base for CARL training environments.

    Subclasses implement tools as public methods with type hints and
    Google-style docstrings (Args: section required for TRL discovery).

    IMPORTANT: __init__ must take NO arguments (TRL constraint).
    Use class-level config or environment variables for external deps.
    """

    spec: EnvironmentSpec  # Subclasses MUST set this as a class attribute

    def __init__(self) -> None:
        self.reward: float = 0.0
        self.done: bool = False
        self._turn_count: int = 0
        self._history: list[dict[str, Any]] = []

    @abstractmethod
    def reset(self, **kwargs: Any) -> str | None:
        """Reset environment state. Receives dataset columns as kwargs.

        Returns:
            Initial observation string, or None.
        """
        self.reward = 0.0
        self.done = False
        self._turn_count = 0
        self._history = []
        return None

    def _record_turn(self, tool_name: str, args: dict[str, Any], result: str) -> None:
        """Track interaction for CARL phase transition analysis."""
        self._turn_count += 1
        self._history.append({
            "turn": self._turn_count,
            "tool": tool_name,
            "args_keys": list(args.keys()),
            "result_length": len(result),
            "reward_at_turn": self.reward,
        })

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def __del__(self) -> None:
        """Override for cleanup. Wrap in try/except for shutdown safety."""
        pass
