"""Contraction mapping probe for training trajectories.

The framework claims learning-from-failure is a contraction on a complete
metric space, with the trajectory converging to a fixed-point attractor
(the "zero construct"). A contraction satisfies d(T(x), T(y)) <= q*d(x,y)
for some 0 <= q < 1. This probe measures whether that condition holds
empirically: it records per-step coherence snapshots and fits a contraction
constant q via the geometric mean of consecutive distance ratios. If
q >= 1 for a sustained window the probe fires ``contraction_violation``
so the caller can decide (fix the reward, adjust the curriculum, or
stop training).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ContractionReport:
    """Immutable snapshot of contraction fit over the probe window."""

    q_hat: float
    """Fitted contraction constant. q < 1 = contracting, q >= 1 = diverging."""

    sample_size: int
    """Number of log-ratio pairs used in the OLS fit."""

    contraction_holds: bool
    """True iff q_hat < 1.0."""

    residual_variance: float
    """Variance of log-ratios around the mean. Lower = tighter fit."""

    window_trajectory_length: int
    """Number of trajectory points currently in the probe window."""


class ContractionProbe:
    """Tracks (phi, cloud_quality) trajectory and fits q_hat over a window.

    Distance metric:
        ``d = sqrt((phi_i - phi_j)**2 + (cq_i - cq_j)**2)``

    Fit: model each step as a contraction ``d_{t+1} = q * d_t``; take the
    geometric mean of consecutive distance ratios as ``q_hat``. Equivalent
    to OLS on ``log(d_{t+1}) = log(q) + log(d_t)`` when ``log(d_t)`` is
    approximately centered (it is, since adjacent distances share scale).

    Call :meth:`record` after each training step with ``(phi_mean,
    cloud_quality)``. Call :meth:`report` to get a snapshot. Call
    :meth:`contraction_violation` to detect divergence.
    """

    def __init__(self, window: int = 50) -> None:
        if window < 3:
            raise ValueError("ContractionProbe window must be >= 3")
        self.window: int = window
        self._history: deque[tuple[float, float]] = deque(maxlen=window)

    def record(self, phi: float, cloud_quality: float) -> None:
        """Append a trajectory point. Non-finite values are silently ignored.

        The probe is designed to be driven from inside a training-step hook
        where the caller may not want to crash on a single bad metric; it
        simply drops the sample and continues.
        """
        if not math.isfinite(phi) or not math.isfinite(cloud_quality):
            return
        self._history.append((float(phi), float(cloud_quality)))

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def report(self) -> ContractionReport | None:
        """Fit q_hat over the current window. Returns ``None`` when the
        window has fewer than three usable points or all consecutive
        distances are zero (degenerate constant trajectory).
        """
        if len(self._history) < 3:
            return None

        pts = list(self._history)
        # Step-to-step distances along the trajectory.
        ds: list[float] = [
            self._distance(pts[i], pts[i + 1]) for i in range(len(pts) - 1)
        ]

        # Need at least two positive distances to form a single ratio.
        positive_ds = [d for d in ds if d > 0.0]
        if len(positive_ds) < 2:
            return None

        # Adjacent ratios d_{t+1}/d_t. Skip when predecessor is zero
        # (cannot normalise) -- treat as trajectory segment boundary.
        ratios: list[float] = []
        for i in range(len(ds) - 1):
            if ds[i] > 0.0 and ds[i + 1] > 0.0:
                ratios.append(ds[i + 1] / ds[i])
        if not ratios:
            return None

        log_ratios = [math.log(r) for r in ratios]
        mean_log = sum(log_ratios) / len(log_ratios)
        q_hat = math.exp(mean_log)
        variance = (
            sum((lr - mean_log) ** 2 for lr in log_ratios) / len(log_ratios)
        )

        return ContractionReport(
            q_hat=q_hat,
            sample_size=len(log_ratios),
            contraction_holds=q_hat < 1.0,
            residual_variance=variance,
            window_trajectory_length=len(pts),
        )

    def contraction_violation(self, threshold: float = 1.0) -> bool:
        """True iff we have enough data AND the fit shows q_hat >= threshold.

        ``threshold`` defaults to 1.0 (the mathematical divergence line).
        Callers sometimes use a slightly stricter threshold (e.g. 0.99) to
        catch marginal trajectories before they diverge; any value in
        (0, infty) is accepted.
        """
        rep = self.report()
        return rep is not None and rep.q_hat >= threshold

    def reset(self) -> None:
        """Clear the history. Probe returns to "needs more data" state."""
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)
