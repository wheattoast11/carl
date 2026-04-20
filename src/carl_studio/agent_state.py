"""Composed collaborators for CARLAgent.

Extracted from ``chat_agent.py`` in the v0.7 Phase-1 god-class decomposition
(ticket R2-001 / F-PYTH-001). These dataclasses own the state they describe;
:class:`CARLAgent` composes them rather than carrying the fields directly.

Backward-compat is preserved on :class:`CARLAgent` via ``@property`` shims
that delegate to these collaborators — every test that reaches for
``agent._total_cost_usd``, ``agent._messages``, ``agent._turn_count``, or
``agent.cost_summary`` continues to pass without edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostLedger:
    """Per-agent cost accumulation. One instance per :class:`CARLAgent` lifetime.

    All mutation routes through :meth:`add_turn` so partial/failed turns,
    one-shot auxiliary calls, and clean turns share a single accounting
    surface. Inputs are defensively clamped to non-negative values to keep
    a single bad usage payload from corrupting the ledger.
    """

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_turn(
        self,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Accumulate one turn's cost/token usage.

        Negative inputs are clamped to zero — the Anthropic SDK occasionally
        returns ``None``/missing usage fields that callers default to ``0``;
        a typo elsewhere must not be able to drive the ledger negative.
        """
        self.total_cost_usd += max(0.0, float(cost_usd))
        self.total_input_tokens += max(0, int(input_tokens))
        self.total_output_tokens += max(0, int(output_tokens))

    def reset(self) -> None:
        """Zero the ledger. Used by session ``load()`` before rehydration."""
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def as_dict(self) -> dict[str, float | int]:
        """Serialize for :pyattr:`CARLAgent.cost_summary` and session save."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


@dataclass
class SessionState:
    """Message history + turn accounting. One instance per chat session.

    ``session_theme`` records the lane the user started in —
    ``""`` = not yet classified, ``"move:explore"`` / ``"move:train"`` /
    ``"move:eval"`` / ``"move:sticky"`` / ``"free-form"`` = canonical moves.
    """

    messages: list[dict[str, Any]] = field(
        default_factory=lambda: [],  # type: list[dict[str, Any]]
    )
    turn_count: int = 0
    session_theme: str = ""

    def append_message(self, msg: dict[str, Any]) -> None:
        """Append a single message dict (role + content) to the history."""
        self.messages.append(msg)

    def increment_turn(self) -> None:
        """Bump ``turn_count`` by one. Called at the top of every ``chat()`` turn."""
        self.turn_count += 1

    def is_fresh(self) -> bool:
        """True when no turn has completed yet — drives the greeting gate.

        The greeting prompt is emitted on the FIRST call after the turn
        counter was incremented (``turn_count == 1``). A resumed session's
        first call has ``turn_count >= 2`` and skips the greeting. A
        ``turn_count == 0`` snapshot (pre-increment, fresh session) is
        also considered fresh so external gates see the right state.
        """
        return self.turn_count <= 1

    def reset(self) -> None:
        """Zero the session state. Used by session ``load()`` before rehydration."""
        self.messages.clear()
        self.turn_count = 0
        self.session_theme = ""
