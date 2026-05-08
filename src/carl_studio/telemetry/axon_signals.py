"""AXON signal-type constants.

Mirrors carl.camp's vendored ``src/lib/axon-signal-types.ts``. The 5 v0.10
top-level signals named in the carl-studio CLAUDE.md "Mental model: AXON
signal vocabulary" section. The remaining 15 commerce / lifecycle signals
listed in the carl.camp allowlist are emitted by carl.camp itself
(billing webhooks, agent-card register, etc.) — not by the studio.

Drift-detection: any rename here MUST land on carl.camp's TS module in
the same PR; the ``/api/axon/ingest`` route validates ``signal_type``
against an allowlist.
"""

from __future__ import annotations

SKILL_TRAINING_STARTED: str = "skill.training_started"
"""Phase transition into learning. Emitted on ``ActionType.TRAINING_STEP``."""

SKILL_CRYSTALLIZED: str = "skill.crystallized"
"""Reward crystallized, artifact learnable. Emitted on ``ActionType.CHECKPOINT``."""

COHERENCE_UPDATE: str = "coherence.update"
"""Internal-consistency observability. Emitted as a SECONDARY signal when a
step's ``kuramoto_r`` / ``phi`` field is populated, regardless of the
primary action type."""

INTERACTION_CREATED: str = "interaction.created"
"""Per-episode lifecycle marker. Emitted on ``ActionType.LLM_REPLY`` /
``ActionType.REWARD``."""

ACTION_DISPATCHED: str = "action.dispatched"
"""Fine-grained trajectory reconstruction. Emitted on ``ActionType.EXTERNAL``."""


# Allowlist for runtime validation — keep in sync with the constants above.
AXON_SIGNAL_ALLOWLIST: frozenset[str] = frozenset(
    {
        SKILL_TRAINING_STARTED,
        SKILL_CRYSTALLIZED,
        COHERENCE_UPDATE,
        INTERACTION_CREATED,
        ACTION_DISPATCHED,
    }
)


__all__ = [
    "ACTION_DISPATCHED",
    "AXON_SIGNAL_ALLOWLIST",
    "COHERENCE_UPDATE",
    "INTERACTION_CREATED",
    "SKILL_CRYSTALLIZED",
    "SKILL_TRAINING_STARTED",
]
