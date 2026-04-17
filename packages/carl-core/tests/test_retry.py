"""Tests for carl_core.retry — deterministic via injectable sleep/clock."""
from __future__ import annotations

import asyncio
import random

import pytest

from carl_core.errors import CARLError, CARLTimeoutError
from carl_core.retry import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    async_retry,
    poll,
    retry,
)


class _FakeClock:
    """Deterministic monotonic clock. `tick(n)` advances time by n seconds."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class _RecordedSleep:
    """Captures sleep(seconds) calls without actually sleeping."""

    def __init__(self, clock: _FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self.clock = clock

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.clock is not None:
            self.clock.tick(seconds)


# ---- retry() -------------------------------------------------------------


def test_retry_success_first_attempt_no_sleep() -> None:
    sleep = _RecordedSleep()
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "ok"

    result = retry(fn, sleep=sleep)
    assert result == "ok"
    assert calls["n"] == 1
    assert sleep.calls == []


def test_retry_success_on_attempt_2_sleeps_once() -> None:
    sleep = _RecordedSleep()
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("boom")
        return "ok"

    policy = RetryPolicy(max_attempts=3, backoff_base=2.0, jitter=False)
    result = retry(fn, policy=policy, sleep=sleep)
    assert result == "ok"
    assert calls["n"] == 2
    assert len(sleep.calls) == 1
    assert sleep.calls[0] == 2.0


def test_retry_exhausted_raises_last_exception() -> None:
    sleep = _RecordedSleep()
    errors = [ConnectionError("a"), ConnectionError("b"), ConnectionError("c")]
    idx = {"i": 0}

    def fn() -> None:
        i = idx["i"]
        idx["i"] += 1
        raise errors[i]

    policy = RetryPolicy(max_attempts=3, backoff_base=1.0, jitter=False)
    with pytest.raises(ConnectionError) as exc_info:
        retry(fn, policy=policy, sleep=sleep)
    assert str(exc_info.value) == "c"
    # Sleeps happen before attempts 2 and 3, so 2 sleeps total.
    assert len(sleep.calls) == 2


def test_retry_non_retryable_raises_immediately() -> None:
    sleep = _RecordedSleep()
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise ValueError("not retryable")

    policy = RetryPolicy(
        max_attempts=5, retryable=(ConnectionError,), jitter=False
    )
    with pytest.raises(ValueError, match="not retryable"):
        retry(fn, policy=policy, sleep=sleep)
    assert calls["n"] == 1
    assert sleep.calls == []


def test_on_retry_callback_fires_with_attempt_exc_delay() -> None:
    sleep = _RecordedSleep()
    observed: list[tuple[int, str, float]] = []

    def cb(attempt: int, exc: BaseException, delay: float) -> None:
        observed.append((attempt, str(exc), delay))

    errors = [ConnectionError("e1"), ConnectionError("e2")]
    idx = {"i": 0}

    def fn() -> str:
        i = idx["i"]
        idx["i"] += 1
        if i < 2:
            raise errors[i]
        return "ok"

    policy = RetryPolicy(
        max_attempts=3, backoff_base=1.0, jitter=False, on_retry=cb
    )
    assert retry(fn, policy=policy, sleep=sleep) == "ok"
    assert len(observed) == 2
    assert observed[0] == (1, "e1", 1.0)
    assert observed[1] == (2, "e2", 2.0)


# ---- RetryPolicy.delay_for -----------------------------------------------


def test_delay_for_base_no_jitter() -> None:
    p = RetryPolicy(backoff_base=2.0, jitter=False, max_delay=1000.0)
    assert p.delay_for(1) == 2.0


def test_delay_for_monotonic_increasing() -> None:
    p = RetryPolicy(backoff_base=2.0, jitter=False, max_delay=1000.0)
    d1 = p.delay_for(1)
    d2 = p.delay_for(2)
    d3 = p.delay_for(3)
    d4 = p.delay_for(4)
    assert d1 < d2 < d3 < d4
    assert d1 == 2.0
    assert d2 == 4.0
    assert d3 == 8.0
    assert d4 == 16.0


def test_max_delay_caps_delay() -> None:
    p = RetryPolicy(backoff_base=2.0, jitter=False, max_delay=5.0)
    assert p.delay_for(1) == 2.0
    assert p.delay_for(2) == 4.0
    assert p.delay_for(3) == 5.0  # capped
    assert p.delay_for(10) == 5.0  # capped
    assert p.delay_for(100) == 5.0  # capped


def test_jitter_adds_bounded_randomness() -> None:
    p = RetryPolicy(backoff_base=2.0, jitter=True, max_delay=1000.0)
    rng = random.Random(42)
    samples = [p.delay_for(2, rng=rng) for _ in range(200)]
    base = 4.0
    # jitter adds 0..0.5*base
    assert all(base <= s <= base + base * 0.5 for s in samples)
    # With seeded rng, there is variance across samples.
    assert len(set(samples)) > 1


def test_delay_for_rejects_zero_attempt() -> None:
    p = RetryPolicy()
    with pytest.raises(ValueError):
        p.delay_for(0)


def test_retry_rejects_zero_max_attempts() -> None:
    policy = RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        retry(lambda: "never", policy=policy, sleep=_RecordedSleep())


# ---- async_retry --------------------------------------------------------


def test_async_retry_success_path() -> None:
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("boom")
        return "ok"

    # jitter=False and backoff_base small so the asyncio.sleep is fast.
    policy = RetryPolicy(max_attempts=3, backoff_base=0.001, jitter=False)
    result = asyncio.run(async_retry(fn, policy=policy))
    assert result == "ok"
    assert calls["n"] == 2


def test_async_retry_exhausted_raises() -> None:
    async def fn() -> None:
        raise ConnectionError("always fails")

    policy = RetryPolicy(max_attempts=2, backoff_base=0.001, jitter=False)
    with pytest.raises(ConnectionError, match="always fails"):
        asyncio.run(async_retry(fn, policy=policy))


def test_async_retry_non_retryable_raises_immediately() -> None:
    async def fn() -> None:
        raise ValueError("nope")

    policy = RetryPolicy(
        max_attempts=5, backoff_base=0.001, retryable=(ConnectionError,)
    )
    with pytest.raises(ValueError, match="nope"):
        asyncio.run(async_retry(fn, policy=policy))


# ---- poll ---------------------------------------------------------------


def test_poll_returns_matching_value() -> None:
    clock = _FakeClock()
    sleep = _RecordedSleep(clock=clock)
    values = iter(["pending", "pending", "done"])

    def fn() -> str:
        return next(values)

    result = poll(
        fn,
        until=lambda v: v == "done",
        timeout_s=10.0,
        interval_s=1.0,
        sleep=sleep,
        clock=clock,
    )
    assert result == "done"
    # Fired two sleeps between three fn() calls.
    assert sleep.calls == [1.0, 1.0]


def test_poll_timeout_raises_carl_timeout_with_context() -> None:
    clock = _FakeClock()
    sleep = _RecordedSleep(clock=clock)

    def fn() -> str:
        return "pending"

    with pytest.raises(CARLTimeoutError) as exc_info:
        poll(
            fn,
            until=lambda v: v == "done",
            timeout_s=5.0,
            interval_s=2.0,
            sleep=sleep,
            clock=clock,
        )
    err = exc_info.value
    assert err.code == "carl.timeout"
    assert err.context["timeout_s"] == 5.0
    assert err.context["interval_s"] == 2.0


def test_poll_sleeps_at_most_remaining_budget() -> None:
    clock = _FakeClock()
    sleep = _RecordedSleep(clock=clock)
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "pending"

    with pytest.raises(CARLTimeoutError):
        poll(
            fn,
            until=lambda v: v == "done",
            timeout_s=2.5,
            interval_s=1.0,
            sleep=sleep,
            clock=clock,
        )
    # Expected sleeps: 1.0, 1.0, 0.5 (trimmed to remaining budget).
    assert sleep.calls == [1.0, 1.0, 0.5]


def test_poll_rejects_negative_timeout() -> None:
    with pytest.raises(ValueError):
        poll(lambda: None, until=lambda v: True, timeout_s=-1.0)


def test_poll_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError):
        poll(lambda: None, until=lambda v: True, timeout_s=1.0, interval_s=0.0)


# ---- CircuitBreaker -----------------------------------------------------


def test_circuit_breaker_cycle_closed_open_halfopen_closed() -> None:
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=2, reset_s=10.0, clock=clock)

    # Start: closed.
    assert cb.state == CircuitState.CLOSED

    # First failure: still closed.
    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("f1")
    assert cb.state == CircuitState.CLOSED

    # Second failure: open.
    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("f2")
    assert cb.state == CircuitState.OPEN

    # Entering while open raises carl.circuit_open.
    with pytest.raises(CARLError) as exc_info:
        with cb:
            pass  # pragma: no cover — should not execute
    assert exc_info.value.code == "carl.circuit_open"
    assert exc_info.value.context["failures"] == 2
    assert exc_info.value.context["reset_s"] == 10.0

    # Advance past reset_s: transitions to half-open.
    clock.tick(10.0)
    assert cb.state == CircuitState.HALF_OPEN

    # Successful call closes.
    with cb:
        pass
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_halfopen_failure_reopens() -> None:
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=1, reset_s=5.0, clock=clock)

    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("f1")
    assert cb.state == CircuitState.OPEN

    clock.tick(5.0)
    assert cb.state == CircuitState.HALF_OPEN

    # Failure while half-open: still increments failures past threshold, reopens.
    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("f2")
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_raises_carl_error_when_open() -> None:
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=1, reset_s=100.0, clock=clock)

    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("fail")

    with pytest.raises(CARLError) as exc_info:
        with cb:
            pass  # pragma: no cover
    assert exc_info.value.code == "carl.circuit_open"


def test_circuit_breaker_success_resets_failure_count() -> None:
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=3, reset_s=10.0, clock=clock)

    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("f1")
    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("f2")

    # Success: clears failures.
    with cb:
        pass
    assert cb.state == CircuitState.CLOSED

    # Two more failures should not trip now (threshold 3, counter reset).
    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("f3")
    with pytest.raises(RuntimeError):
        with cb:
            raise RuntimeError("f4")
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_exports_in_package() -> None:
    import carl_core

    assert carl_core.CircuitBreaker is CircuitBreaker
    assert carl_core.CircuitState is CircuitState
    assert carl_core.RetryPolicy is RetryPolicy
    assert carl_core.retry is retry
    assert carl_core.async_retry is async_retry
    assert carl_core.poll is poll
