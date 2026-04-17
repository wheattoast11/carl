"""Tests for CoherenceMonitorCallback — WS-T4 + WS-T6."""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from carl_studio.training.callbacks import CoherenceMonitorCallback, _safe_mean
from carl_studio.training.rewards.multiscale import (
    clamp_counts,
    reset_clamp_counts,
)


@pytest.fixture(autouse=True)
def _fresh_state():
    reset_clamp_counts()
    yield
    reset_clamp_counts()


def _state(step: int = 1) -> Any:
    return SimpleNamespace(global_step=step)


class _FakeReward:
    """Stand-in for make_carl_reward's closure exposing the needed attrs."""

    def __init__(
        self,
        metrics: list[dict[str, float]] | None = None,
        step: int = 1,
        with_lock: bool = True,
    ) -> None:
        self._metrics_lock = threading.Lock() if with_lock else None
        self._last_metrics = [(step, metrics) if metrics is not None else None]


# ---------------------------------------------------------------------------
# _safe_mean — WS-T4 foundation
# ---------------------------------------------------------------------------


class TestSafeMean:
    def test_empty_iterable_returns_default(self):
        assert _safe_mean([]) == 0.0
        assert _safe_mean([], default=-1.0) == -1.0

    def test_non_empty_mean(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0

    def test_mixed_numeric(self):
        assert _safe_mean([1, 2, 3]) == 2.0


# ---------------------------------------------------------------------------
# on_log empty / missing inputs
# ---------------------------------------------------------------------------


class TestOnLogEmpty:
    def test_logs_none_is_noop(self):
        cb = CoherenceMonitorCallback(_FakeReward(metrics=[{"multiscale": 0.5}]))
        cb.on_log(None, _state(), None, logs=None)  # must not raise

    def test_missing_metrics_is_noop(self):
        fake = SimpleNamespace()  # no _last_metrics attr
        cb = CoherenceMonitorCallback(fake)
        # Use a step far past 1 so we skip the kappa/sigma first-step emit
        logs: dict[str, Any] = {}
        cb.on_log(None, _state(100), None, logs=logs)
        # No batch-derived metrics (phi_mean/cloud_quality/etc) should be emitted
        assert "coherence/phi_mean" not in logs
        assert "coherence/cloud_quality" not in logs

    def test_empty_batch_metrics_no_division(self):
        cb = CoherenceMonitorCallback(_FakeReward(metrics=[]))
        logs: dict[str, Any] = {}
        # Must not ZeroDivisionError
        cb.on_log(None, _state(), None, logs=logs)

    def test_single_batch_metric_new_format(self):
        cb = CoherenceMonitorCallback(
            _FakeReward(
                metrics=[
                    {"multiscale": 0.8, "cloud_quality": 0.6, "discontinuity": 0.4}
                ]
            )
        )
        logs: dict[str, Any] = {}
        cb.on_log(None, _state(1), None, logs=logs)
        assert logs["coherence/phi_mean"] == 0.8
        assert logs["coherence/cloud_quality"] == 0.6
        assert logs["coherence/discontinuity_density"] == 0.4
        # Composite
        assert abs(logs["coherence/cryst_to_melt_ratio"] - (0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4)) < 1e-9
        # Clamp telemetry present
        assert "coherence/reward_clamp_total" in logs

    def test_legacy_format_path(self):
        cb = CoherenceMonitorCallback(
            _FakeReward(
                metrics=[
                    {
                        "phi_mean": 0.1,
                        "cloud_quality_mean": 0.2,
                        "defect_density": 0.3,
                        "n_crystallizations": 4,
                        "n_meltings": 2,
                        "scale_coherence": {0: 0.5, 1: 0.6},
                    }
                ]
            )
        )
        logs: dict[str, Any] = {}
        cb.on_log(None, _state(1), None, logs=logs)
        assert logs["coherence/phi_mean"] == 0.1
        assert logs["coherence/cloud_quality"] == 0.2
        assert logs["coherence/discontinuity_density"] == 0.3
        assert logs["coherence/cryst_to_melt_ratio"] == 4 / 2
        assert logs["coherence/coherence_scale_0"] == 0.5
        assert logs["coherence/coherence_scale_1"] == 0.6


# ---------------------------------------------------------------------------
# Error isolation — WS-T6
# ---------------------------------------------------------------------------


class _ExplodingReward:
    """Reward stand-in whose _last_metrics attribute raises on read."""

    @property
    def _last_metrics(self):
        raise RuntimeError("boom")


class TestErrorIsolation:
    def test_callback_exception_does_not_propagate(self, caplog):
        caplog.set_level(logging.WARNING, logger="carl_studio.training.callbacks")
        cb = CoherenceMonitorCallback(_ExplodingReward())
        # Must not raise
        cb.on_log(None, _state(1), None, logs={})
        # Warning recorded
        msgs = [r.getMessage() for r in caplog.records]
        assert any("callback" in m and "failed" in m for m in msgs)

    def test_on_epoch_end_does_not_propagate(self, caplog):
        caplog.set_level(logging.WARNING, logger="carl_studio.training.callbacks")

        class _BadReward:
            _metrics_lock = threading.Lock()
            _last_metrics = [None]

        cb = CoherenceMonitorCallback(_BadReward())
        # Should succeed even though nothing interesting happens.
        cb.on_epoch_end(None, _state(1), None)

    def test_on_step_end_isolation(self):
        cb = CoherenceMonitorCallback(_FakeReward(metrics=[]))
        # Must return None cleanly
        assert cb.on_step_end(None, _state(1), None) is None


# ---------------------------------------------------------------------------
# Thread-safety — WS-T6
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_no_deadlock_under_contention(self):
        """10 threads hammering on_log + epoch end concurrently."""
        reward = _FakeReward(
            metrics=[{"multiscale": 0.5, "cloud_quality": 0.5, "discontinuity": 0.5}],
            step=1,
        )
        cb = CoherenceMonitorCallback(reward)

        errors: list[BaseException] = []
        stop = threading.Event()

        def worker():
            try:
                while not stop.is_set():
                    cb.on_log(None, _state(1), None, logs={})
                    cb.on_epoch_end(None, _state(1), None)
            except BaseException as exc:  # pragma: no cover - would be a bug
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()

        time.sleep(0.05)  # let them spin
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), "deadlock detected"

        assert not errors, f"worker raised: {errors!r}"

    def test_lock_is_released_after_use(self):
        reward = _FakeReward(metrics=[{"multiscale": 0.1}], step=1)
        cb = CoherenceMonitorCallback(reward)
        cb.on_log(None, _state(1), None, logs={})
        # Lock should be free after the callback returns.
        assert reward._metrics_lock.acquire(timeout=0.1)
        reward._metrics_lock.release()


# ---------------------------------------------------------------------------
# Clamp telemetry aggregation
# ---------------------------------------------------------------------------


class TestClampTelemetry:
    def test_epoch_end_resets_counters(self):
        # Bump the counters via the shared helper first.
        from carl_studio.training.rewards.multiscale import _clamp_reward

        _clamp_reward(float("nan"))
        _clamp_reward(1e9)
        assert clamp_counts()["total"] == 2

        cb = CoherenceMonitorCallback(_FakeReward(metrics=[]))
        cb.on_epoch_end(None, _state(1), None)
        assert clamp_counts()["total"] == 0

    def test_on_log_emits_running_counters(self):
        from carl_studio.training.rewards.multiscale import _clamp_reward

        _clamp_reward(float("nan"))
        cb = CoherenceMonitorCallback(
            _FakeReward(metrics=[{"multiscale": 0.5, "cloud_quality": 0.5, "discontinuity": 0.5}])
        )
        logs: dict[str, Any] = {}
        cb.on_log(None, _state(1), None, logs=logs)
        assert logs["coherence/reward_clamp_nonfinite"] >= 1
        assert logs["coherence/reward_clamp_total"] >= 1
