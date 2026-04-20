"""Shared primitives for allow/deny gates across carl-studio.

Two decorators currently gate call-sites: :func:`carl_studio.consent.consent_gate`
(privacy / network-egress) and :func:`carl_studio.tier.tier_gate` (monetized
feature access). They remain distinct public APIs, but under the hood they
now share:

* :class:`GatingPredicate` — a ``Protocol`` any gate's decision object
  can implement. Both :class:`carl_studio.consent.ConsentPredicate` and
  :class:`carl_studio.tier.TierPredicate` satisfy it.
* :class:`BaseGate` — a generic helper that owns the check → emit → raise
  loop shared by both gates. See :meth:`BaseGate.enforce` (used by
  :func:`consent_gate`) and :meth:`BaseGate.for_predicate` (used by
  :func:`tier_gate`).
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

import functools
import inspect
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
GATE_COHERENCE_INSUFFICIENT = "carl.gate.coherence_insufficient"


def emit_gate_event(
    *,
    predicate_name: str,
    allowed: bool,
    reason: str,
    gate_code: str | None = None,
    chain: InteractionChain | None = None,
) -> None:
    """Record a structured gate event on the provided chain.

    No-op when *chain* is ``None``. Shape:

    * ``action`` → :attr:`~carl_core.interaction.ActionType.GATE_CHECK`
    * ``name`` → predicate name (``"consent:telemetry"``, ``"tier:mcp.serve"``,
      ``"coherence:min_R=0.5"``)
    * ``input`` → ``{"predicate": <name>, "allowed": <bool>}``
    * ``output`` → ``{"reason": <human-readable>, "gate_code": <code | None>}``
    * ``success`` → whether the gate allowed the call

    ``gate_code`` is optional for back-compat with pre-v0.10 callers; when
    present it lands in the output dict for downstream filtering.
    """
    if chain is None:
        return
    from carl_core.interaction import ActionType, Step

    output: dict[str, Any] = {"reason": reason}
    if gate_code is not None:
        output["gate_code"] = gate_code
    chain.append(
        Step(
            action=ActionType.GATE_CHECK,
            name=predicate_name,
            input={"predicate": predicate_name, "allowed": allowed},
            output=output,
            success=allowed,
        )
    )


P = TypeVar("P", bound=GatingPredicate)


class BaseGate(Generic[P]):
    """Generic helper that owns the shared gate loop.

    Both :func:`carl_studio.consent.consent_gate` and
    :func:`carl_studio.tier.tier_gate` share the same four-step shape:

    1. Run ``predicate.check()`` → ``(allowed, detail)``.
    2. Emit a ``GATE_CHECK`` step on the active
       :class:`~carl_core.interaction.InteractionChain` (if any).
    3. On deny, attach a ``gate_code`` (plus any extra context) to the
       domain-specific error via an ``error.context`` dict.
    4. Raise.

    :class:`BaseGate` extracts those four steps without forcing the two
    public APIs to share a signature — consent is a *call-site gate*
    (``consent_gate("flag")``) and tier is a *decorator factory*
    (``@tier_gate(Tier.PAID, feature=...)``). The two public surfaces
    stay distinct; the loop is shared.

    Type parameter ``P`` binds to a concrete predicate type so
    :meth:`_attach_gate_context` can receive structured predicate state
    without losing static typing at the call site.
    """

    # ------------------------------------------------------------------
    # Low-level primitives (shared by every gate surface)
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_predicate(
        predicate: P, *, chain: InteractionChain | None
    ) -> tuple[bool, str]:
        """Run ``predicate.check()`` and emit the structured event.

        Returns ``(allowed, detail)`` — callers decide whether to raise.
        The event is emitted whether the predicate allows or denies so
        downstream audit consumers see every check, not just failures.
        """
        allowed, detail = predicate.check()
        emit_gate_event(
            predicate_name=predicate.name,
            allowed=allowed,
            reason=detail,
            chain=chain,
        )
        return allowed, detail

    @staticmethod
    def _attach_gate_context(
        error: Exception,
        *,
        predicate: P,
        gate_code: str,
        detail: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Attach ``gate_code`` (and predicate-specific fields) to ``error.context``.

        :class:`carl_core.errors.CARLError` subclasses (e.g.
        :class:`~carl_studio.consent.ConsentError`) already own a
        ``context`` dict, so we *merge* — never replace — to preserve any
        fields the caller already set at construction time.

        :class:`~carl_core.tier.TierGateError` is a plain ``Exception`` in
        the primitive layer (so ``carl-core`` stays dependency-free); it
        has no native ``context`` attribute. We layer one on via
        ``setattr`` at the carl-studio boundary. The narrow pyright
        ignore matches the existing pattern that lived in ``tier.py``
        before v0.8.
        """
        payload: dict[str, Any] = {
            "gate_code": gate_code,
            "gate_name": predicate.name,
            "gate_detail": detail,
        }
        if extra:
            payload.update(extra)

        existing = getattr(error, "context", None)
        if isinstance(existing, dict):
            # Merge: preserve keys the caller already wrote (e.g. ``flag``
            # on ConsentError). New gate-level keys win only where absent.
            merged: dict[str, Any] = dict(existing)  # type: ignore[arg-type]
            for key, value in payload.items():
                merged.setdefault(key, value)
            try:
                error.context = merged  # pyright: ignore[reportAttributeAccessIssue]
            except AttributeError:  # pragma: no cover — slotted class without context
                setattr(error, "context", merged)
        else:
            error.context = payload  # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    def _emit_failure_event(
        predicate: P,
        chain: InteractionChain | None,
        gate_code: str,  # noqa: ARG004 — reserved for future structured routing
        detail: str,
    ) -> None:
        """Emit a single deny-path event.

        Exposed as a dedicated helper so non-``_wrap_predicate`` call-sites
        (e.g. :func:`carl_studio.consent.consent_gate`'s unknown-flag
        branch, which short-circuits before constructing a predicate view
        of the bad input) can still produce the shared event shape.
        """
        emit_gate_event(
            predicate_name=predicate.name,
            allowed=False,
            reason=detail,
            chain=chain,
        )

    # ------------------------------------------------------------------
    # High-level surfaces (used by the two public gates)
    # ------------------------------------------------------------------

    @classmethod
    def enforce(
        cls,
        predicate: P,
        *,
        chain: InteractionChain | None,
        error_factory: Callable[[str], Exception],
        gate_code: str,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        """Run the predicate; raise the error_factory's exception on deny.

        Used by *call-site gates* (``consent_gate("flag")``) that aren't
        wrapping a function. The decorator variant is
        :meth:`for_predicate`.
        """
        allowed, detail = cls._wrap_predicate(predicate, chain=chain)
        if allowed:
            return
        error = error_factory(detail)
        cls._attach_gate_context(
            error,
            predicate=predicate,
            gate_code=gate_code,
            detail=detail,
            extra=extra_context,
        )
        raise error

    @classmethod
    def for_predicate(
        cls,
        *,
        predicate_factory: Callable[..., P],
        error_type: type[Exception],
        gate_code: str,
    ) -> Callable[..., Callable[..., Any]]:
        """Return a decorator factory for gating a function behind ``predicate``.

        Usage (the pattern used by ``tier_gate``)::

            my_gate = BaseGate.for_predicate(
                predicate_factory=MyPredicate,
                error_type=MyError,
                gate_code=GATE_MY_DENIED,
            )

            @my_gate(required_tier, feature="x")
            def guarded(): ...

        The returned callable accepts the same ``*args, **kwargs`` the
        downstream gate wants to pass through to ``predicate_factory``.
        It produces a decorator that:

        * Pops a ``_gate_chain`` sentinel from the wrapped call's kwargs
          (so callers can thread a chain through without changing the
          wrapped function's signature).
        * Runs the predicate.
        * On deny, builds ``error_type(...)`` via a best-effort
          constructor (supports both the ``CARLError(message, *, code,
          context)`` shape and the plain-``Exception(*args)`` shape used
          by ``TierGateError``).
        * On allow, calls the wrapped function and returns its result.

        Sync and ``async def`` functions are both supported: the returned
        wrapper preserves coroutine-ness via :func:`inspect.iscoroutinefunction`.
        """

        def factory(
            *factory_args: Any, **factory_kwargs: Any
        ) -> Callable[..., Any]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                if inspect.iscoroutinefunction(func):

                    @functools.wraps(func)
                    async def async_wrapper(
                        *args: Any, **kwargs: Any
                    ) -> Any:
                        chain: InteractionChain | None = kwargs.pop(
                            "_gate_chain", None
                        )
                        predicate = predicate_factory(
                            *factory_args, **factory_kwargs
                        )
                        allowed, detail = cls._wrap_predicate(
                            predicate, chain=chain
                        )
                        if not allowed:
                            err = _build_gate_error(
                                error_type, predicate, detail
                            )
                            cls._attach_gate_context(
                                err,
                                predicate=predicate,
                                gate_code=gate_code,
                                detail=detail,
                                extra=_predicate_context(predicate),
                            )
                            raise err
                        return await func(*args, **kwargs)

                    return async_wrapper

                @functools.wraps(func)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    chain: InteractionChain | None = kwargs.pop(
                        "_gate_chain", None
                    )
                    predicate = predicate_factory(
                        *factory_args, **factory_kwargs
                    )
                    allowed, detail = cls._wrap_predicate(
                        predicate, chain=chain
                    )
                    if not allowed:
                        err = _build_gate_error(
                            error_type, predicate, detail
                        )
                        cls._attach_gate_context(
                            err,
                            predicate=predicate,
                            gate_code=gate_code,
                            detail=detail,
                            extra=_predicate_context(predicate),
                        )
                        raise err
                    return func(*args, **kwargs)

                return wrapper

            return decorator

        return factory


def _predicate_context(predicate: GatingPredicate) -> dict[str, Any]:
    """Extract predicate-specific context for the error ``context`` dict.

    Best-effort: reads common predicate fields (``feature``, ``required``,
    plus a resolved ``current`` via ``_effective()`` when available) and
    returns them as a dict. Enum values are unwrapped via ``.value`` so
    the payload stays JSON-safe (matches the pre-v0.8 ``TierGateError``
    context shape exactly: ``{"feature": str, "required": str,
    "current": str}``).
    """
    ctx: dict[str, Any] = {}
    feature = getattr(predicate, "feature", None)
    if feature is not None:
        ctx["feature"] = feature
    required = getattr(predicate, "required", None)
    if required is not None:
        ctx["required"] = getattr(required, "value", required)
    effective_fn = getattr(predicate, "_effective", None)
    if callable(effective_fn):
        try:
            current = effective_fn()
        except Exception:
            current = None
        if current is not None:
            ctx["current"] = getattr(current, "value", current)
    return ctx


def _build_gate_error(
    error_type: type[Exception],
    predicate: GatingPredicate,
    detail: str,
) -> Exception:
    """Construct an error of ``error_type`` carrying predicate context.

    Two construction shapes are supported:

    * ``TierGateError(feature, required, current)`` — the primitive-layer
      shape used by ``carl_core.tier``. We look for ``feature`` /
      ``required`` / ``current`` attributes on the predicate (via
      ``getattr`` with ``None`` fallback) and, when all three are
      present, invoke that signature. This preserves the shipped
      ``TierGateError`` fields (``err.feature``, ``err.required``,
      ``err.current``) untouched.
    * ``CARLError(message, *, code, context)`` — the standard carl-core
      error shape. Used as the fallback.
    """
    feature = getattr(predicate, "feature", None)
    required = getattr(predicate, "required", None)
    # ``current`` is not on our predicates directly; derive from
    # ``_effective`` when available to preserve the legacy signature.
    current = None
    effective_fn = getattr(predicate, "_effective", None)
    if callable(effective_fn):
        try:
            current = effective_fn()
        except Exception:
            current = None

    if feature is not None and required is not None and current is not None:
        try:
            return error_type(feature, required, current)  # type: ignore[call-arg]
        except TypeError:
            pass

    # Fall back to the CARLError signature.
    try:
        return error_type(detail)
    except TypeError:
        return error_type()


# ---------------------------------------------------------------------------
# CoherenceGate — predicate that consults Kuramoto R on the active chain.
# ---------------------------------------------------------------------------
#
# The IRE paper's "G" (coherence-gated routing) is the fourth tuple element.
# v0.8 shipped BaseGate for consent + tier, both of which gate on discrete
# predicates. Neither consults the coherence field. v0.10 closes that loop:
# CoherenceGate is a GatingPredicate that reads the recent window's Kuramoto
# order parameter R and denies when R is below threshold.
#
# This is opt-in — existing call sites gated only on consent/tier continue to
# work unchanged. Wrap a function with ``@coherence_gate(min_R=0.5)`` to add
# coherence-based admission on top of any other gates already in place.


class CoherenceError(Exception):
    """Raised when coherence is below the configured threshold.

    Carries ``code = "carl.gate.coherence_insufficient"`` in a ``context``
    dict and the same ``gate_code`` key the other gate errors use, so
    operators can filter on ``carl.gate.*`` uniformly.
    """

    def __init__(self, message: str, *, current_R: float, required_R: float) -> None:
        super().__init__(message)
        self.code = GATE_COHERENCE_INSUFFICIENT
        self.current_R = current_R
        self.required_R = required_R
        self.context = {
            "gate_code": GATE_COHERENCE_INSUFFICIENT,
            "gate_name": "coherence",
            "current_R": current_R,
            "required_R": required_R,
        }


@dataclass(frozen=True)
class CoherenceSnapshot:
    """The minimal read from the active chain that CoherenceGate needs."""

    R: float
    window_size: int
    # Fano V4 contrastive: when the probe returns a constant (e.g., 1.0
    # always), R carries no information even though window_size > 0.
    # ``variance`` is 0.0 when unknown (no samples or single sample) and
    # mirrors sample variance otherwise; ``is_degenerate`` rolls this up
    # alongside no-data for callers that want to downweight or disable
    # coherence-gated admission.
    variance: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.window_size > 0

    @property
    def is_degenerate(self) -> bool:
        """True when the snapshot carries no discriminative signal.

        Degenerate iff ``window_size == 0`` OR the sample variance is
        below ``1e-6`` AND ``window_size >= 2`` (i.e., multiple identical
        samples — a constant-returning probe). A one-sample window is
        NOT flagged degenerate since a single measurement is unknowable
        as degenerate or not.
        """
        if self.window_size == 0:
            return True
        if self.window_size >= 2 and self.variance < 1e-6:
            return True
        return False


def read_chain_coherence(chain: Any, window: int = 16) -> CoherenceSnapshot:
    """Sample the tail of a chain's kuramoto_r fields.

    Returns ``CoherenceSnapshot(R=1.0, window_size=0)`` when the chain is
    None, has no steps, or none of its steps carry ``kuramoto_r`` — a
    "no data, allow" default that prevents false-denies during cold-start
    or when coherence auto-attach is disabled.
    """

    if chain is None or not hasattr(chain, "steps"):
        return CoherenceSnapshot(R=1.0, window_size=0, variance=0.0)
    steps = list(chain.steps)[-window:]
    samples = [
        s.kuramoto_r for s in steps if getattr(s, "kuramoto_r", None) is not None
    ]
    if not samples:
        return CoherenceSnapshot(R=1.0, window_size=0, variance=0.0)
    mean = sum(samples) / len(samples)
    if len(samples) >= 2:
        variance = sum((v - mean) ** 2 for v in samples) / len(samples)
    else:
        variance = 0.0
    return CoherenceSnapshot(R=mean, window_size=len(samples), variance=variance)


class CoherenceGatePredicate:
    """A :class:`GatingPredicate` that denies when recent Kuramoto R < threshold."""

    def __init__(
        self,
        *,
        min_R: float,
        chain: Any = None,
        window: int = 16,
    ) -> None:
        if not 0.0 <= min_R <= 1.0:
            raise ValueError(f"min_R must be in [0, 1]; got {min_R}")
        self._min_R = min_R
        self._chain = chain
        self._window = window
        self._last_snapshot: CoherenceSnapshot | None = None

    @property
    def name(self) -> str:
        return f"coherence:min_R={self._min_R}"

    @property
    def min_R(self) -> float:
        return self._min_R

    @property
    def last_snapshot(self) -> CoherenceSnapshot | None:
        return self._last_snapshot

    def check(self) -> tuple[bool, str]:
        snap = read_chain_coherence(self._chain, self._window)
        self._last_snapshot = snap
        if not snap.has_data:
            return True, "no coherence data"
        if snap.R < self._min_R:
            return (
                False,
                f"coherence R={snap.R:.3f} below required {self._min_R:.3f} "
                f"(window={snap.window_size})",
            )
        return True, f"coherence R={snap.R:.3f} >= {self._min_R:.3f}"


def coherence_gate(
    *,
    min_R: float,
    window: int = 16,
    feature: str | None = None,
) -> "Callable[[Callable[..., Any]], Callable[..., Any]]":
    """Decorator form. Usage::

        @coherence_gate(min_R=0.5, feature="training.update")
        def step_policy_update(*args, _gate_chain=my_chain, **kwargs): ...

    The chain is threaded via the ``_gate_chain`` kwarg (same convention
    as ``consent_gate`` and ``tier_gate``). On deny, raises
    :class:`CoherenceError` with full context. Safe to stack with other
    gates — each raises its own domain-specific error with a
    ``carl.gate.*`` code.
    """

    def _decorate(func: "Callable[..., Any]") -> "Callable[..., Any]":
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            chain = kwargs.pop("_gate_chain", None)
            predicate = CoherenceGatePredicate(
                min_R=min_R, chain=chain, window=window
            )
            allowed, reason = predicate.check()
            snap = predicate.last_snapshot or CoherenceSnapshot(R=1.0, window_size=0)
            if allowed:
                emit_gate_event(
                    predicate_name=predicate.name,
                    allowed=True,
                    reason=reason,
                    gate_code=None,
                    chain=chain,
                )
                return func(*args, **kwargs)
            emit_gate_event(
                predicate_name=predicate.name,
                allowed=False,
                reason=reason,
                gate_code=GATE_COHERENCE_INSUFFICIENT,
                chain=chain,
            )
            feat = feature or func.__name__
            raise CoherenceError(
                f"{feat}: {reason}",
                current_R=snap.R,
                required_R=min_R,
            )

        return wrapper

    return _decorate


__all__ = [
    "GATE_CONSENT_DENIED",
    "GATE_TIER_INSUFFICIENT",
    "GATE_COHERENCE_INSUFFICIENT",
    "BaseGate",
    "GatingPredicate",
    "CoherenceError",
    "CoherenceSnapshot",
    "CoherenceGatePredicate",
    "coherence_gate",
    "read_chain_coherence",
    "emit_gate_event",
]
