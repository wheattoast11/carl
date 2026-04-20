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


__all__ = [
    "GATE_CONSENT_DENIED",
    "GATE_TIER_INSUFFICIENT",
    "BaseGate",
    "GatingPredicate",
    "emit_gate_event",
]
