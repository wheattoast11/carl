"""Retry, backoff, and circuit breaker primitives.

Stdlib-only. No heavy imports. Deterministic via injectable sleep/clock.

Replaces ad-hoc poll-with-sleep loops (eval/runner.py, a2a/_cli.py,
observe/data_source.py) and silent exception swallowing (training/trainer.py).
"""
from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Exponential-backoff retry policy with optional jitter and max delay."""

    max_attempts: int = 3
    backoff_base: float = 2.0  # seconds for first retry; doubles each attempt
    max_delay: float = 30.0
    jitter: bool = True  # adds 0..0.5 * delay random jitter
    retryable: tuple[type[BaseException], ...] = (
        IOError,
        ConnectionError,
        TimeoutError,
    )
    on_retry: Callable[[int, BaseException, float], None] | None = None

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Delay before the `attempt`th retry (1-indexed).

        Returns exponentially increasing delay, capped at max_delay, optionally
        with up to 50% random jitter added.
        """
        if attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {attempt}")
        base = min(self.backoff_base * (2 ** (attempt - 1)), self.max_delay)
        if not self.jitter:
            return base
        r = rng if rng is not None else random
        return base + r.random() * (base * 0.5)


def retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Synchronously retry fn() according to policy.

    Raises the last exception if all attempts exhausted, or immediately on
    non-retryable error (which propagates unchanged).
    """
    p = policy or RetryPolicy()
    if p.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {p.max_attempts}")
    last: BaseException | None = None
    for attempt in range(1, p.max_attempts + 1):
        try:
            return fn()
        except p.retryable as exc:
            last = exc
            if attempt == p.max_attempts:
                break
            delay = p.delay_for(attempt)
            if p.on_retry is not None:
                p.on_retry(attempt, exc, delay)
            sleep(delay)
    assert last is not None
    raise last


async def async_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
) -> T:
    """Async retry variant. fn is a zero-arg awaitable factory.

    Build with functools.partial: async_retry(partial(client.get, url), ...).
    """
    p = policy or RetryPolicy()
    if p.max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {p.max_attempts}")
    last: BaseException | None = None
    for attempt in range(1, p.max_attempts + 1):
        try:
            return await fn()
        except p.retryable as exc:
            last = exc
            if attempt == p.max_attempts:
                break
            delay = p.delay_for(attempt)
            if p.on_retry is not None:
                p.on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)
    assert last is not None
    raise last


def poll(
    fn: Callable[[], T],
    *,
    until: Callable[[T], bool],
    timeout_s: float,
    interval_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Poll fn() until until(result) is True or deadline. Raises CARLTimeoutError.

    Clock + sleep are injectable for deterministic testing. Sleeps for at most
    `interval_s` or the remaining deadline budget, whichever is smaller.
    """
    from carl_core.errors import CARLTimeoutError

    if timeout_s < 0:
        raise ValueError(f"timeout_s must be >= 0, got {timeout_s}")
    if interval_s <= 0:
        raise ValueError(f"interval_s must be > 0, got {interval_s}")
    deadline = clock() + timeout_s
    while True:
        result = fn()
        if until(result):
            return result
        remaining = deadline - clock()
        if remaining <= 0:
            raise CARLTimeoutError(
                f"poll timed out after {timeout_s}s",
                context={"timeout_s": timeout_s, "interval_s": interval_s},
            )
        sleep(min(interval_s, remaining))


class CircuitState:
    """Circuit breaker states. Compared as strings for lightweight logging."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Three-state breaker: closed → open after N failures → half-open after reset_s → closed on success.

    Thread-safe. Usage:

        cb = CircuitBreaker(failure_threshold=3, reset_s=10)
        with cb:
            risky_call()

    When open, entering the context raises CARLError(code="carl.circuit_open").
    After reset_s elapses, transitions to half-open and allows one probe; on
    success closes, on failure reopens.
    """

    failure_threshold: int = 5
    reset_s: float = 30.0
    clock: Callable[[], float] = field(default=time.monotonic)
    _state: str = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> str:
        """Current state, observing any pending half-open transition."""
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    def _maybe_half_open_locked(self) -> None:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if self.clock() - self._opened_at >= self.reset_s:
                self._state = CircuitState.HALF_OPEN

    def __enter__(self) -> CircuitBreaker:
        from carl_core.errors import CARLError

        with self._lock:
            self._maybe_half_open_locked()
            if self._state == CircuitState.OPEN:
                raise CARLError(
                    "circuit breaker is open",
                    code="carl.circuit_open",
                    context={
                        "failures": self._consecutive_failures,
                        "reset_s": self.reset_s,
                    },
                )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        with self._lock:
            if exc_type is None:
                self._state = CircuitState.CLOSED
                self._consecutive_failures = 0
                self._opened_at = None
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = self.clock()


__all__ = [
    "RetryPolicy",
    "retry",
    "async_retry",
    "poll",
    "CircuitBreaker",
    "CircuitState",
]
