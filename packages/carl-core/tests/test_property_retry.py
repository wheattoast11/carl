"""Property tests for carl_core.retry.

Invariants:
- For a function that succeeds on attempt k (1 <= k <= n), total sleep
  budget grows monotonically with n (wall clock correctness).
- ``delay_for`` is monotonically non-decreasing (with jitter disabled).
- ``max_delay`` clamps ``delay_for`` across all attempt numbers.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.retry import RetryPolicy, retry


class _RecordedSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@given(
    n=st.integers(min_value=1, max_value=10),
    k=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=75, deadline=None)
def test_total_sleep_budget_monotonic(n: int, k: int) -> None:
    """Total sleep duration should equal sum(delay_for(i)) for i in 1..k-1."""
    if k > n:
        return  # irrelevant: succeeds after n attempts not possible
    sleep = _RecordedSleep()
    calls = {"i": 0}

    def fn() -> str:
        calls["i"] += 1
        if calls["i"] < k:
            raise ConnectionError("flaky")
        return "ok"

    policy = RetryPolicy(
        max_attempts=n,
        backoff_base=1.0,
        max_delay=10.0,
        jitter=False,
    )
    result = retry(fn, policy=policy, sleep=sleep)
    assert result == "ok"
    # Sleeps happen before every retry attempt (k-1 sleeps total).
    assert len(sleep.calls) == k - 1
    # Total budget equals the sum of delay_for(i) for i in 1..k-1.
    expected = sum(policy.delay_for(i) for i in range(1, k))
    assert abs(sum(sleep.calls) - expected) < 1e-9


@given(
    backoff=st.floats(
        min_value=0.001, max_value=2.0, allow_nan=False, allow_infinity=False
    ),
    max_delay=st.floats(
        min_value=0.5, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    upto=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=75, deadline=None)
def test_delay_for_monotonic_non_decreasing(
    backoff: float, max_delay: float, upto: int
) -> None:
    """delay_for is non-decreasing when jitter is off."""
    p = RetryPolicy(backoff_base=backoff, max_delay=max_delay, jitter=False)
    delays = [p.delay_for(i) for i in range(1, upto + 1)]
    for prev, curr in zip(delays, delays[1:], strict=False):
        assert curr >= prev


@given(
    backoff=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    cap=st.floats(
        min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False
    ),
    attempt=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=75, deadline=None)
def test_max_delay_clamps(backoff: float, cap: float, attempt: int) -> None:
    """delay_for(attempt) <= max_delay for any attempt, with jitter off."""
    p = RetryPolicy(backoff_base=backoff, max_delay=cap, jitter=False)
    assert p.delay_for(attempt) <= cap + 1e-9
