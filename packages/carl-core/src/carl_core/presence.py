"""Presence — a composition helper for queryable self-awareness.

This module is a **thin composition** of existing carl-core primitives
(InteractionChain · kuramoto_r · compute_phi) plus one light coherence
snapshot. It is NOT a new primitive — it is an API seam that lets a
running agent call a single function and get back a typed description
of its own coherence state.

The v0.10 peer review (``docs/v10_master_plan.md`` Finding P2-1)
explicitly downgraded an earlier "carl-sense as new primitive" proposal
to "thin composition". This module is the result.

Public surface:
    :class:`PresenceReport` — the typed report dataclass.
    :func:`compose_presence_report` — build a report from a chain.

Downstream consumers (e.g., ``carl_studio.mcp.server``) register this
function as the ``carl.presence.self`` MCP tool so agents can
introspect their own coherence via a tool call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Callable


@dataclass(frozen=True)
class PresenceReport:
    """A queryable snapshot of an agent's own coherence state.

    Fields:

    * ``R`` — recent-window Kuramoto order parameter, in ``[0, 1]``.
    * ``psi`` — recent-window mean phase (radians), in ``[-π, π]``. May
      be 0.0 when no phase data is available; check ``has_phase``.
    * ``has_phase`` — whether the ``psi`` field carries meaningful data.
    * ``window_size`` — number of steps that contributed to the
      sample. 0 means cold-start; callers should treat as "no data".
    * ``crystallization`` — in ``[0, 1]``, equal to ``R`` — high R
      corresponds to stable recurrence per the BITC paper's
      ``tau = 1 - crystallization`` coupling.
    * ``constructive`` — Deutsch-Marletto invariant: can the system
      transform again? True when the chain has steps and the most
      recent step is ``success=True``.
    * ``recent_action_types`` — the last N ``ActionType`` values on
      the chain (default N = 8), most-recent last. Empty list if no
      data.
    * ``step_count`` — total steps on the chain.
    * ``note`` — human-readable summary suitable for an agent's
      self-narration.
    """

    R: float
    psi: float
    has_phase: bool
    window_size: int
    crystallization: float
    constructive: bool
    recent_action_types: list[str] = field(default_factory=list)
    step_count: int = 0
    note: str = ""


def _sample_window(chain: Any, window: int) -> tuple[list[float], list[str], list[bool]]:
    """Return ``(kuramoto_samples, action_types, successes)`` for the tail window."""

    if chain is None or not hasattr(chain, "steps"):
        return [], [], []
    steps = list(chain.steps)[-window:]
    ks = [
        float(s.kuramoto_r)
        for s in steps
        if getattr(s, "kuramoto_r", None) is not None and isfinite(float(s.kuramoto_r))
    ]
    # Action type may be an enum or a string — normalize to the string value.
    atypes: list[str] = []
    for s in steps:
        raw = getattr(s, "action", None)
        if raw is None:
            continue
        atypes.append(getattr(raw, "value", str(raw)))
    successes = [bool(getattr(s, "success", True)) for s in steps]
    return ks, atypes, successes


def compose_presence_report(
    chain: Any,
    *,
    window: int = 8,
    phase: float | None = None,
) -> PresenceReport:
    """Build a :class:`PresenceReport` from the tail of an InteractionChain.

    Args:
        chain: An :class:`InteractionChain` (duck-typed; must have
            ``.steps`` iterable of :class:`Step`). ``None`` is tolerated
            and returns a cold-start report.
        window: Number of most-recent steps to sample. Default 8.
        phase: Optional externally-computed mean phase (radians) when
            the caller has access to the phase field directly. When
            ``None`` the report sets ``has_phase=False`` and ``psi=0.0``.

    Returns:
        A frozen ``PresenceReport``. Safe to serialize; all fields are
        primitives.
    """

    ks, atypes, successes = _sample_window(chain, window)
    if not ks:
        return PresenceReport(
            R=1.0,
            psi=phase if phase is not None else 0.0,
            has_phase=phase is not None,
            window_size=0,
            crystallization=1.0,
            constructive=False,
            recent_action_types=atypes,
            step_count=len(list(chain.steps)) if chain is not None and hasattr(chain, "steps") else 0,
            note="cold-start: no coherence samples in window",
        )
    R = sum(ks) / len(ks)
    # Constructive iff the chain has a most-recent step AND it succeeded.
    constructive = bool(successes and successes[-1])
    step_count = len(list(chain.steps)) if hasattr(chain, "steps") else 0
    note = f"R={R:.3f} over {len(ks)} samples; {'stable' if R >= 0.7 else 'diffuse' if R < 0.3 else 'liquid'}"
    return PresenceReport(
        R=R,
        psi=phase if phase is not None else 0.0,
        has_phase=phase is not None,
        window_size=len(ks),
        crystallization=R,
        constructive=constructive,
        recent_action_types=atypes,
        step_count=step_count,
        note=note,
    )


# ---------------------------------------------------------------------------
# Default endogenous probe — v0.11 Fano V7 realization helper
# ---------------------------------------------------------------------------
#
# A minimal "no-training-context" coherence probe that estimates R from the
# chain's own recent-step success rate. Use when you don't have a training-
# loop phi signal but still want auto-attach + coherence-gated routing to
# operate on a meaningful signal instead of all-1.0 defaults.
#
# The estimate is intentionally coarse — R := success_rate over the recent
# window of the SAME action type. A session that's mostly succeeding gets
# a high R; a session cascading into failure sees R drop. This gives the
# coherence gate a floor signal without any external probe.


def success_rate_probe(chain: Any, *, window: int = 8) -> Callable[..., dict[str, float]]:
    """Return a probe closure that estimates kuramoto_r from chain success rate.

    Usage::

        from carl_core.presence import success_rate_probe
        from carl_core.interaction import InteractionChain

        chain = InteractionChain()
        chain.register_coherence_probe(success_rate_probe(chain))

    The returned probe reads the tail window of the chain's steps (those
    with the SAME action type as the step currently being recorded) and
    returns ``{"kuramoto_r": success_rate}`` where success_rate is the
    fraction of successful recent steps. Bounded by ``[0.0, 1.0]``.

    This is an endogenous probe — it consults only the chain's own
    history, no external state. Fano V3 endogenous-measurability PASS.
    """

    def _probe(*, action: Any, name: str, input: Any, output: Any) -> dict[str, float]:
        if chain is None or not hasattr(chain, "steps"):
            return {"kuramoto_r": 1.0}
        same_type = [s for s in list(chain.steps)[-window:] if s.action == action]
        if not same_type:
            return {"kuramoto_r": 1.0}  # cold-start: no history yet
        success_count = sum(1 for s in same_type if bool(getattr(s, "success", True)))
        rate = success_count / len(same_type)
        return {"kuramoto_r": float(rate)}

    return _probe


__all__ = [
    "PresenceReport",
    "compose_presence_report",
    "success_rate_probe",
]
