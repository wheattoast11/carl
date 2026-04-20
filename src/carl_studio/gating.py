"""Shared primitives for allow/deny gates across carl-studio.

Two decorators currently gate call-sites: :func:`carl_studio.consent.consent_gate`
(privacy / network-egress) and :func:`carl_studio.tier.tier_gate` (monetized
feature access). They remain distinct public APIs, but under the hood they
now share:

* :class:`GatingPredicate` — a ``Protocol`` any gate's decision object
  can implement. Both :class:`carl_studio.consent.ConsentPredicate` and
  :class:`carl_studio.tier.TierPredicate` satisfy it.
* ``carl.gate.*`` — a shared structured error-code namespace. Individual
  gates keep their domain-specific exception classes; they surface a
  ``gate_code`` attribute in their error ``context`` so operators can
  filter "any gate denial" uniformly without collapsing the taxonomy.
* :func:`emit_gate_event` — a single structured logging path that appends
  one :class:`~carl_core.interaction.Step` per gate check to the active
  :class:`~carl_core.interaction.InteractionChain`. Both consent and tier
  produce identical event shapes so downstream consumers (audit,
  training signal, replay) can treat them polymorphically.

This module intentionally contains no business logic — it's just the
shape primitives that keep the two gates aligned.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from carl_core.interaction import InteractionChain


@runtime_checkable
class GatingPredicate(Protocol):
    """A predicate used by a gate to decide allow/deny at a call-site.

    Implementations MUST be pure w.r.t. the state they observe: call
    :meth:`check` twice in a row with unchanged state and expect the same
    answer. Side-effects (logging, persistence) belong to the gate that
    runs the predicate, not the predicate itself.
    """

    @property
    def name(self) -> str:  # stable identifier for event logging (e.g., "consent:telemetry")
        ...

    def check(self) -> tuple[bool, str]:
        """Return ``(allowed, reason)``. ``reason`` is human-readable when denied."""
        ...


# carl.gate.* is the shared structured error-code namespace. Each gate
# raises its own domain-specific exception class (ConsentError, TierGateError)
# with these codes surfaced via the error ``context`` dict so operators
# can filter on ``context["gate_code"]`` without collapsing the taxonomy.
GATE_CONSENT_DENIED = "carl.gate.consent_denied"
GATE_TIER_INSUFFICIENT = "carl.gate.tier_insufficient"


def emit_gate_event(
    *,
    predicate_name: str,
    allowed: bool,
    reason: str,
    chain: InteractionChain | None = None,
) -> None:
    """Record a structured gate event on the provided chain.

    No-op when *chain* is ``None``. The shape is intentionally minimal so
    callers on both sides (consent, tier) produce identical event
    payloads when inspected downstream:

    * ``action`` → :attr:`~carl_core.interaction.ActionType.GATE_CHECK`
    * ``name`` → predicate name (``"consent:telemetry"``, ``"tier:mcp.serve"``)
    * ``input`` → ``{"predicate": <name>, "allowed": <bool>}``
    * ``output`` → ``{"reason": <human-readable>}``
    * ``success`` → whether the gate allowed the call
    """
    if chain is None:
        return
    # Local import: carl_core.interaction is a light primitive, but we
    # keep the import inside the function so ``import carl_studio.gating``
    # stays cheap at module load time.
    from carl_core.interaction import ActionType, Step

    chain.append(
        Step(
            action=ActionType.GATE_CHECK,
            name=predicate_name,
            input={"predicate": predicate_name, "allowed": allowed},
            output={"reason": reason},
            success=allowed,
        )
    )


__all__ = [
    "GATE_CONSENT_DENIED",
    "GATE_TIER_INSUFFICIENT",
    "GatingPredicate",
    "emit_gate_event",
]
