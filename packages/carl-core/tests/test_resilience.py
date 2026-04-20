"""Tests for carl_core.resilience — BreakAndRetryStrategy contract tests.

Tests exercise the composition of RetryPolicy + CircuitBreaker behind a
single entrypoint. A FakeClock drives the breaker so we can deterministically
observe OPEN → HALF-OPEN transitions without sleeping.
"""
from __future__ import annotations

import asyncio

import pytest

from carl_core.resilience import BreakAndRetryStrategy, CircuitOpenError
from carl_core.retry import CircuitBreaker, CircuitState, RetryPolicy


class _FakeClock:
    """Deterministic monotonic clock shared with the breaker."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def _fast_policy(
    *,
    max_attempts: int = 3,
    retryable: tuple[type[BaseException], ...] = (IOError, ConnectionError, TimeoutError),
) -> RetryPolicy:
    """Retry policy with zero-delay backoff — tests never touch ``time.sleep``."""
    return RetryPolicy(
        max_attempts=max_attempts,
        backoff_base=0.0,
        max_delay=0.0,
        jitter=False,
        retryable=retryable,
    )


def _breaker(
    *,
    failure_threshold: int = 3,
    reset_s: float = 60.0,
    clock: _FakeClock | None = None,
    tracked: tuple[type[BaseException], ...] = (Exception,),
) -> CircuitBreaker:
    """CircuitBreaker with an injected clock for deterministic reset timing."""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        reset_s=reset_s,
        clock=clock or _FakeClock(),
        tracked_exceptions=tracked,
    )


# ---- constructor / config -------------------------------------------------


def test_strategy_requires_keyword_args() -> None:
    # Positional arguments should be rejected — retry_policy is kw-only.
    with pytest.raises(TypeError):
        BreakAndRetryStrategy(_fast_policy())  # pyright: ignore[reportCallIssue]


def test_strategy_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        BreakAndRetryStrategy(retry_policy=RetryPolicy(max_attempts=0))


# ---- run() success path ----------------------------------------------------


def test_run_returns_fn_result_on_success() -> None:
    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy())

    def fn(x: int, *, y: int) -> int:
        return x + y

    assert strategy.run(fn, 2, y=3) == 5


def test_run_records_success_closes_breaker() -> None:
    br = _breaker()
    # Pre-open the breaker indirectly by racking up failures.
    for _ in range(3):
        try:
            with br:
                raise ConnectionError("boom")
        except ConnectionError:
            pass
    assert br.state == CircuitState.OPEN

    # Give the breaker room to half-open so run() is allowed through.
    clock = br.clock
    assert isinstance(clock, _FakeClock)
    clock.tick(br.reset_s + 1)
    assert br.state == CircuitState.HALF_OPEN

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(), breaker=br)
    assert strategy.run(lambda: "ok") == "ok"
    # Success after HALF-OPEN must close the breaker and clear the counter.
    assert br.state == CircuitState.CLOSED


# ---- run() retry + failure paths -------------------------------------------


def test_run_retries_on_retryable_then_succeeds() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(max_attempts=5))
    assert strategy.run(fn) == "ok"
    assert calls["n"] == 3


def test_run_exhausts_retries_and_reraises() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise ConnectionError("boom")

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(max_attempts=3))
    with pytest.raises(ConnectionError, match="boom"):
        strategy.run(fn)
    assert calls["n"] == 3


def test_run_non_retryable_propagates_without_retry() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise ValueError("bug")  # not in retryable tuple

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(max_attempts=5))
    with pytest.raises(ValueError, match="bug"):
        strategy.run(fn)
    assert calls["n"] == 1  # no retry on non-retryable


def test_run_records_failure_on_breaker() -> None:
    br = _breaker(
        failure_threshold=5,
        tracked=(ConnectionError, TimeoutError, IOError),
    )
    strategy = BreakAndRetryStrategy(
        retry_policy=_fast_policy(max_attempts=2),
        breaker=br,
    )

    def fn() -> None:
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        strategy.run(fn)
    # One run with tracked failure → one consecutive failure on breaker.
    assert br._consecutive_failures == 1  # pyright: ignore[reportPrivateUsage]
    assert br.state == CircuitState.CLOSED


def test_run_untracked_failure_does_not_bump_breaker() -> None:
    # Breaker only counts ConnectionError; a retry-policy-non-retryable
    # like ValueError propagates but does NOT feed the breaker either,
    # because the breaker's tracked_exceptions narrows its own gate.
    br = _breaker(
        failure_threshold=3,
        tracked=(ConnectionError,),
    )
    strategy = BreakAndRetryStrategy(
        retry_policy=_fast_policy(max_attempts=2),
        breaker=br,
    )
    with pytest.raises(ValueError):
        strategy.run(lambda: (_ for _ in ()).throw(ValueError("bug")))
    assert br._consecutive_failures == 0  # pyright: ignore[reportPrivateUsage]


# ---- run() circuit-open fast-fail ------------------------------------------


def test_run_raises_circuit_open_when_breaker_open() -> None:
    clock = _FakeClock()
    br = _breaker(failure_threshold=2, clock=clock)
    # Open the breaker by hand.
    for _ in range(2):
        try:
            with br:
                raise ConnectionError("boom")
        except ConnectionError:
            pass
    assert br.state == CircuitState.OPEN

    called = {"n": 0}

    def fn() -> None:
        called["n"] += 1

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(), breaker=br)
    with pytest.raises(CircuitOpenError) as exc_info:
        strategy.run(fn)
    assert exc_info.value.code == "carl.resilience.circuit_open"
    assert called["n"] == 0  # fn never invoked
    # Context carries observability fields.
    assert exc_info.value.context["failure_threshold"] == 2


def test_run_no_breaker_is_plain_retry() -> None:
    # None breaker collapses the strategy to a retry wrapper.
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient")
        return "ok"

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(), breaker=None)
    assert strategy.run(fn) == "ok"
    assert calls["n"] == 2


# ---- on_open callback ------------------------------------------------------


def test_on_open_fires_when_breaker_trips() -> None:
    br = _breaker(
        failure_threshold=2,
        tracked=(ConnectionError,),
    )
    fired = {"n": 0}

    def on_open() -> None:
        fired["n"] += 1

    strategy = BreakAndRetryStrategy(
        retry_policy=_fast_policy(max_attempts=1),
        breaker=br,
        on_open=on_open,
    )

    def fn() -> None:
        raise ConnectionError("boom")

    # First failure — breaker CLOSED → CLOSED (one tick from threshold).
    with pytest.raises(ConnectionError):
        strategy.run(fn)
    assert fired["n"] == 0
    # Second failure — breaker CLOSED → OPEN; on_open fires.
    with pytest.raises(ConnectionError):
        strategy.run(fn)
    assert fired["n"] == 1
    # Third call — breaker was already OPEN, strategy short-circuits with
    # CircuitOpenError. on_open must NOT fire again (not a transition).
    with pytest.raises(CircuitOpenError):
        strategy.run(fn)
    assert fired["n"] == 1


def test_on_open_callback_exception_does_not_mask_fn_error() -> None:
    br = _breaker(failure_threshold=1, tracked=(ConnectionError,))

    def broken_callback() -> None:
        raise RuntimeError("callback bug")

    strategy = BreakAndRetryStrategy(
        retry_policy=_fast_policy(max_attempts=1),
        breaker=br,
        on_open=broken_callback,
    )

    def fn() -> None:
        raise ConnectionError("boom")

    # Callback raises internally but the strategy re-raises the original.
    with pytest.raises(ConnectionError, match="boom"):
        strategy.run(fn)
    # Breaker still recorded the transition.
    assert br.state == CircuitState.OPEN


# ---- run_async parity ------------------------------------------------------


def test_run_async_returns_fn_result_on_success() -> None:
    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy())

    async def fn(x: int, *, y: int) -> int:
        return x + y

    result = asyncio.run(strategy.run_async(fn, 2, y=3))
    assert result == 5


def test_run_async_retries_on_retryable_then_succeeds() -> None:
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(max_attempts=5))
    result = asyncio.run(strategy.run_async(fn))
    assert result == "ok"
    assert calls["n"] == 3


def test_run_async_exhausts_retries_and_reraises() -> None:
    async def fn() -> None:
        raise ConnectionError("boom")

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(max_attempts=2))
    with pytest.raises(ConnectionError, match="boom"):
        asyncio.run(strategy.run_async(fn))


def test_run_async_raises_circuit_open_when_breaker_open() -> None:
    br = _breaker(failure_threshold=1)
    try:
        with br:
            raise ConnectionError("boom")
    except ConnectionError:
        pass
    assert br.state == CircuitState.OPEN

    called = {"n": 0}

    async def fn() -> None:
        called["n"] += 1

    strategy = BreakAndRetryStrategy(retry_policy=_fast_policy(), breaker=br)
    with pytest.raises(CircuitOpenError):
        asyncio.run(strategy.run_async(fn))
    assert called["n"] == 0


def test_run_async_records_success_and_failure() -> None:
    br = _breaker(
        failure_threshold=5,
        tracked=(ConnectionError, TimeoutError, IOError),
    )
    strategy = BreakAndRetryStrategy(
        retry_policy=_fast_policy(max_attempts=1),
        breaker=br,
    )

    async def bad() -> None:
        raise ConnectionError("boom")

    async def good() -> str:
        return "ok"

    with pytest.raises(ConnectionError):
        asyncio.run(strategy.run_async(bad))
    assert br._consecutive_failures == 1  # pyright: ignore[reportPrivateUsage]

    assert asyncio.run(strategy.run_async(good)) == "ok"
    # Success closes the breaker + resets counter.
    assert br._consecutive_failures == 0  # pyright: ignore[reportPrivateUsage]
    assert br.state == CircuitState.CLOSED
