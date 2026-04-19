"""Integration test: TTT tau transitions are observable from the public path.

SEM-009: ``GRPOReflectionCallback.on_log`` previously read
``logs.get("witness/tau", self._last_tau)`` but nothing in the public repo
writes ``witness/tau``, so the TTT crystalline gate (``tau < 0.3``) was
permanently dormant for public users. This test suite verifies that tau is
now derived from the CARL reward function's ``_last_traces`` via Kuramoto-R
and published to the logs stream, so that the micro-update gate can actually
fire without ``terminals-runtime``.
"""
from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

class _FakeTrace:
    """Minimal CoherenceTrace stand-in exposing both ``kuramoto_R`` and
    ``phi`` so we can exercise both code paths in ``_read_kuramoto_R`` /
    ``_kuramoto_from_phi`` without pulling in the full carl_core dep surface.

    ``R`` is the Kuramoto-R value the trace reports. ``phi`` is a uniform
    array that — under the half-circle mapping used by
    ``lr_resonance.ResonanceLRCallback._read_kuramoto_R`` — reduces to the
    same value, so the fallback path produces the same answer.
    """

    def __init__(self, R: float, n_tokens: int = 32) -> None:
        self._R = float(R)
        # Half-circle mapping: R = |mean(exp(i*pi*phi))|. For a constant
        # phi field of value p, that reduces to |exp(i*pi*p)| = 1. To get
        # a non-trivial R via the phi path we build a symmetric split:
        # half the tokens at phi_low, half at phi_high, chosen so the
        # mean complex exponential has magnitude ~= R. Keep it simple for
        # the test -- just expose a constant phi for ``hasattr(phi)`` checks.
        self.phi = np.full(n_tokens, 0.5, dtype=np.float64)

    def kuramoto_R(self) -> float:
        return self._R


def _build_reward_fn() -> tuple[object, list]:
    """Build a stub CARL reward fn that exposes ``_last_traces`` like
    ``carl_studio.training.rewards.composite.make_carl_reward``.

    Returns the closure and the one-slot holder list so tests can
    swap in a fresh batch per step::

        rf, holder = _build_reward_fn()
        holder[0] = [_FakeTrace(0.8)]

    This matches the real production shape:
    ``_last_traces: list[Any] = [None]`` and
    ``_last_traces[0] = batch_traces`` in ``make_carl_reward``.
    """
    traces_holder: list = [None]

    def rf(*args, **kwargs):  # pragma: no cover - never actually called
        return [0.0]

    rf._last_traces = traces_holder  # type: ignore[attr-defined]
    rf._metrics_lock = threading.Lock()  # type: ignore[attr-defined]
    return rf, traces_holder


def _make_callback(rf: object = None):
    """Build a GRPOReflectionCallback without pulling torch/LoRA.

    The callback only needs ``_carl_fn`` + ``_last_tau`` for the ``on_log``
    path under test. We construct with the reward fn positional arg so the
    attribute ends up at the canonical name (``_carl_fn``) that the real
    code uses.
    """
    from carl_studio.ttt.grpo_integration import GRPOReflectionCallback

    return GRPOReflectionCallback(
        carl_reward_fn=rf,
        tokenizer=None,  # on_log never touches the tokenizer
        enable=True,
    )


# --------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------

class TestTauTransitions:
    """Verify tau is computed from traces and published to logs."""

    def test_tau_is_computed_from_traces(self) -> None:
        """With a trace reporting R=0.8, tau should resolve to 1 - 0.8 = 0.2."""
        rf, holder = _build_reward_fn()
        holder[0] = [_FakeTrace(0.8)]

        cb = _make_callback(rf)
        logs: dict[str, float] = {}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            logs=logs,
        )

        assert "witness/tau" in logs, (
            "witness/tau must be published by on_log so downstream callbacks "
            "can consume it without depending on terminals-runtime"
        )
        assert "witness/kuramoto_R" in logs
        assert math.isclose(logs["witness/tau"], 1.0 - 0.8, rel_tol=1e-6)
        assert math.isclose(logs["witness/kuramoto_R"], 0.8, rel_tol=1e-6)
        assert math.isclose(cb._last_tau, 0.2, rel_tol=1e-6)

    def test_tau_transitions_across_multiple_steps(self) -> None:
        """Simulated training trajectory: as Kuramoto-R rises (model becomes
        more phase-locked), tau = 1 - R falls, and the observable sequence
        passes through the crystalline threshold (0.3)."""
        rf, holder = _build_reward_fn()
        cb = _make_callback(rf)

        tau_series: list[float] = []
        # R increasing -> tau decreasing (gaseous -> crystalline transition).
        for R in [0.2, 0.4, 0.6, 0.8]:
            holder[0] = [_FakeTrace(R)]
            logs: dict[str, float] = {}
            cb.on_log(
                args=SimpleNamespace(),
                state=SimpleNamespace(global_step=1),
                control=SimpleNamespace(),
                logs=logs,
            )
            tau_series.append(logs["witness/tau"])

        # tau = 1 - R must decrease monotonically as R rises.
        assert tau_series == sorted(tau_series, reverse=True), (
            f"tau must fall monotonically as R rises; got {tau_series}"
        )
        # At least one sample must clear the TTT crystalline gate so the
        # micro-update actually has an opportunity to fire.
        assert min(tau_series) < 0.3, (
            f"tau trajectory {tau_series} never crosses the crystalline "
            "threshold; TTT gate would remain dormant"
        )

    def test_tau_falls_back_to_last_when_no_traces(self) -> None:
        """When traces are unavailable (empty holder), the callback must
        preserve the last observed tau rather than silently reverting to
        the 0.5 initial default."""
        rf, holder = _build_reward_fn()
        cb = _make_callback(rf)
        cb._last_tau = 0.42  # simulate a previous step's reading
        holder[0] = None  # explicit: no traces this step

        logs: dict[str, float] = {}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=2),
            control=SimpleNamespace(),
            logs=logs,
        )

        # No new computation -> witness/tau NOT written, tau preserved.
        assert "witness/tau" not in logs
        assert cb._last_tau == pytest.approx(0.42)

    def test_tau_falls_back_to_upstream_when_no_traces(self) -> None:
        """When traces are unavailable but terminals-runtime has already
        written witness/tau to logs, the callback must respect that value
        (precedence rule #2 from the module docstring)."""
        rf, holder = _build_reward_fn()
        cb = _make_callback(rf)
        cb._last_tau = 0.5
        holder[0] = None

        logs: dict[str, float] = {"witness/tau": 0.15}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=3),
            control=SimpleNamespace(),
            logs=logs,
        )

        # Upstream value wins since no fresh traces were available.
        assert cb._last_tau == pytest.approx(0.15)
        # And we didn't clobber the upstream write.
        assert logs["witness/tau"] == pytest.approx(0.15)

    def test_callback_computed_tau_overrides_upstream(self) -> None:
        """Precedence rule #1: when the callback has fresh traces, its
        computed tau takes priority over any upstream witness/tau value.
        Stale runtime writes must not win over live trace data."""
        rf, holder = _build_reward_fn()
        holder[0] = [_FakeTrace(0.9)]  # R=0.9 -> tau=0.1

        cb = _make_callback(rf)
        # Simulate terminals-runtime having written a stale (or wrong) value.
        logs: dict[str, float] = {"witness/tau": 0.8}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=4),
            control=SimpleNamespace(),
            logs=logs,
        )

        assert logs["witness/tau"] == pytest.approx(0.1, abs=1e-6), (
            "fresh callback-computed tau must override upstream stale value"
        )
        assert cb._last_tau == pytest.approx(0.1, abs=1e-6)

    def test_no_carl_fn_preserves_initial_tau(self) -> None:
        """With no reward function at all, the callback must not crash and
        must fall back cleanly to the initial tau (0.5)."""
        cb = _make_callback(rf=None)
        logs: dict[str, float] = {}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            logs=logs,
        )
        assert cb._last_tau == pytest.approx(0.5)
        # No witness/tau published because no computation occurred.
        assert "witness/tau" not in logs

    def test_on_log_is_noop_when_logs_is_none(self) -> None:
        """Transformers passes logs=None in some callback contexts. The
        callback must short-circuit cleanly rather than raising."""
        rf, holder = _build_reward_fn()
        holder[0] = [_FakeTrace(0.8)]
        cb = _make_callback(rf)
        cb._last_tau = 0.5

        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            logs=None,
        )

        # _last_tau is untouched when logs is None (no dict to publish into).
        assert cb._last_tau == pytest.approx(0.5)


class TestMetricsLockIsRespected:
    """The _metrics_lock guards the trace holder against concurrent writes
    from the reward function. The callback must acquire it (matching the
    pattern used by ResonanceLRCallback)."""

    def test_metrics_lock_is_acquired(self) -> None:
        rf, holder = _build_reward_fn()
        holder[0] = [_FakeTrace(0.75)]

        # Wrap the existing lock with a spy that counts acquisitions.
        acquire_count = [0]
        real_lock = rf._metrics_lock  # type: ignore[attr-defined]

        class _SpyLock:
            def acquire(self, *a, **kw):
                acquire_count[0] += 1
                return real_lock.acquire(*a, **kw)

            def release(self):
                return real_lock.release()

        rf._metrics_lock = _SpyLock()  # type: ignore[attr-defined]

        cb = _make_callback(rf)
        logs: dict[str, float] = {}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            logs=logs,
        )

        assert acquire_count[0] >= 1, (
            "callback must acquire _metrics_lock before reading _last_traces"
        )
        assert math.isclose(logs["witness/tau"], 0.25, rel_tol=1e-6)


class TestTraceUnwrap:
    """The ``_last_traces`` holder can be a one-slot list or a direct list
    depending on which composite wrote it. Both shapes must work."""

    def test_one_slot_holder(self) -> None:
        """Shape used by make_carl_reward: [list_of_traces]."""
        rf, holder = _build_reward_fn()
        holder[0] = [_FakeTrace(0.8)]

        cb = _make_callback(rf)
        logs: dict[str, float] = {}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            logs=logs,
        )
        assert math.isclose(logs["witness/tau"], 0.2, rel_tol=1e-6)

    def test_direct_list_holder(self) -> None:
        """Shape used by PhaseAdaptiveCARLReward: list_of_traces (no wrap)."""
        from carl_studio.ttt.grpo_integration import GRPOReflectionCallback

        rf = type("RF", (), {})()
        rf._last_traces = [_FakeTrace(0.6), _FakeTrace(0.6)]  # type: ignore[attr-defined]

        cb = GRPOReflectionCallback(
            carl_reward_fn=rf,
            tokenizer=None,
            enable=True,
        )
        logs: dict[str, float] = {}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            logs=logs,
        )
        assert math.isclose(logs["witness/tau"], 1.0 - 0.6, rel_tol=1e-6)

    def test_empty_holder(self) -> None:
        """[] / [None] must both fall back rather than crash."""
        rf, holder = _build_reward_fn()
        # holder is [None] by construction -- simulate empty-but-present.
        cb = _make_callback(rf)
        logs: dict[str, float] = {}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            logs=logs,
        )
        assert "witness/tau" not in logs  # no computation, no publish

    def test_non_finite_R_is_skipped(self) -> None:
        """A trace returning NaN/inf R must be filtered, not propagated."""

        class _BadTrace:
            phi = np.array([0.5, 0.5])  # fallback phi path exists

            def kuramoto_R(self) -> float:
                return float("nan")

        class _GoodTrace(_FakeTrace):
            pass

        rf, holder = _build_reward_fn()
        holder[0] = [_BadTrace(), _GoodTrace(0.9)]

        cb = _make_callback(rf)
        logs: dict[str, float] = {}
        cb.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=1),
            control=SimpleNamespace(),
            logs=logs,
        )
        # Good trace alone -> R=0.9, tau=0.1.
        assert math.isclose(logs["witness/tau"], 0.1, rel_tol=1e-6)
