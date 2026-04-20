"""Composed retry + circuit-breaker primitive.

`BreakAndRetryStrategy` fuses :class:`~carl_core.retry.RetryPolicy` with
:class:`~carl_core.retry.CircuitBreaker` behind a single call site. The two
primitives already compose correctly when called by hand — this module just
makes the pairing explicit, so call sites stop re-implementing the wrapping
logic with subtle semantic drift.

Contract
--------
* If ``breaker`` is provided and is currently ``OPEN``, ``run()`` raises
  :class:`CircuitOpenError` (``code="carl.resilience.circuit_open"``) without
  ever calling ``fn``. Fast-fail by design — that's the point of a breaker.
* Otherwise ``run()`` delegates to :func:`carl_core.retry.retry` with the
  supplied ``retry_policy``. A successful return records a success on the
  breaker (always closes + resets the failure counter, even if the breaker
  was never opened). A raised exception records a failure on the breaker
  (which bumps the consecutive-failure counter; the breaker opens on its
  own threshold), then re-raises the original exception unchanged.
* ``on_open`` fires exactly once per transition — when a ``run()`` call
  records a failure that trips the breaker from non-OPEN to OPEN. It never
  fires for a run that was rejected because the breaker was already open
  (that's what :class:`CircuitOpenError` is for).
* :meth:`run_async` mirrors :meth:`run` using
  :func:`carl_core.retry.async_retry`.

Design notes
------------
* The breaker's own context-manager protocol owns the
  success/failure bookkeeping. We use ``with breaker:`` rather than calling
  hypothetical ``record_success``/``record_failure`` methods so the strategy
  stays honest to whatever tracking semantics the breaker instance was
  constructed with (notably ``tracked_exceptions``).
* A ``None`` breaker is a legitimate configuration — it collapses the
  strategy to "retry with no fast-fail gate". Call sites that only want
  retries should still prefer ``retry()`` directly; passing ``breaker=None``
  here exists so callers can opt in/out of the breaker dynamically without
  branching on the strategy.
* The strategy is stateless — it owns no attributes that change during a
  run. All state lives on the injected breaker. Two concurrent ``run()``
  calls on the same strategy are as safe as the underlying breaker
  (which is thread-safe — see ``CircuitBreaker._lock``).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from carl_core.errors import CARLError
from carl_core.retry import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    async_retry,
    retry,
)

T = TypeVar("T")


class CircuitOpenError(CARLError):
    """Raised by :class:`BreakAndRetryStrategy` when the breaker is OPEN.

    Distinct from the breaker's own ``code="carl.circuit_open"`` so callers
    can route strategy-level fast-fail separately from anywhere else the
    raw breaker context manager is entered.
    """

    code = "carl.resilience.circuit_open"


@dataclass(kw_only=True)
class BreakAndRetryStrategy:
    """Composed retry + breaker strategy. See module docstring for the contract.

    Parameters
    ----------
    retry_policy
        Retry policy forwarded to :func:`carl_core.retry.retry` /
        :func:`carl_core.retry.async_retry`.
    breaker
        Optional circuit breaker. When ``None`` the strategy degrades to a
        plain retry wrapper.
    on_open
        Optional zero-arg callback, invoked exactly once when a failing
        ``run()`` transitions the breaker from non-OPEN to OPEN. Callback
        exceptions are swallowed so they never mask the underlying error.
    """

    retry_policy: RetryPolicy
    breaker: CircuitBreaker | None = None
    on_open: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        # RetryPolicy validates max_attempts lazily inside retry(); we front-
        # load the check so the strategy itself fails fast when misconfigured.
        if self.retry_policy.max_attempts < 1:
            raise ValueError(
                f"retry_policy.max_attempts must be >= 1, "
                f"got {self.retry_policy.max_attempts}",
            )

    # -- public API --------------------------------------------------------

    def run(self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Synchronously invoke ``fn(*args, **kwargs)`` under the strategy.

        See module docstring for the exception contract.
        """
        self._assert_closed_or_halfopen()
        try:
            result = retry(
                lambda: fn(*args, **kwargs),
                policy=self.retry_policy,
            )
        except BaseException as exc:
            self._record_failure(exc)
            raise
        self._record_success()
        return result

    async def run_async(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: object,
        **kwargs: object,
    ) -> T:
        """Asynchronous :meth:`run` — same contract, awaits ``fn``."""
        self._assert_closed_or_halfopen()
        try:
            result = await async_retry(
                lambda: fn(*args, **kwargs),
                policy=self.retry_policy,
            )
        except BaseException as exc:
            self._record_failure(exc)
            raise
        self._record_success()
        return result

    # -- internals ---------------------------------------------------------

    def _assert_closed_or_halfopen(self) -> None:
        """Raise :class:`CircuitOpenError` iff the breaker is currently OPEN.

        Reading ``breaker.state`` advances a pending HALF-OPEN transition
        (the breaker does this under its own lock), so a long-elapsed
        OPEN breaker auto-heals to HALF-OPEN here — exactly the behaviour
        the raw breaker context manager has.
        """
        br = self.breaker
        if br is None:
            return
        if br.state == CircuitState.OPEN:
            raise CircuitOpenError(
                "circuit breaker is open; refusing to call fn",
                context={
                    "reset_s": br.reset_s,
                    "failure_threshold": br.failure_threshold,
                },
            )

    def _record_success(self) -> None:
        """Drive the breaker's success path (closes + resets counter)."""
        br = self.breaker
        if br is None:
            return
        # Using the context manager here keeps us faithful to whatever
        # success semantics CircuitBreaker.__exit__ implements, rather than
        # poking private attributes. Entering while CLOSED/HALF-OPEN never
        # raises — the OPEN guard was already applied at run() entry.
        with br:
            pass

    def _record_failure(self, exc: BaseException) -> None:
        """Drive the breaker's failure path and fire on_open on transition."""
        br = self.breaker
        if br is None:
            return
        was_open = br.state == CircuitState.OPEN
        try:
            with br:
                raise exc
        except BaseException:
            # Breaker has consumed the exception into its __exit__ and will
            # have re-raised it — we don't care, the caller's raise re-
            # raises the original. Intentionally swallow here.
            pass
        now_open = br.state == CircuitState.OPEN
        if self.on_open is not None and now_open and not was_open:
            try:
                self.on_open()
            except BaseException:
                # on_open is telemetry — never let it mask the real error.
                pass


__all__ = [
    "BreakAndRetryStrategy",
    "CircuitOpenError",
]
