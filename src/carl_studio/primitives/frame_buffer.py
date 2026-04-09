"""
Frame Buffer — temporal Phi tracking and Kuramoto coherence gate.

Maintains a rolling window of FrameRecords with trajectory analysis,
Kuramoto order parameter R, and phase transition detection.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameRecord:
    """Single observation frame with Phi and action metadata.

    The ``screenshot`` field is typed as ``object`` so that callers can pass
    a PIL Image (or any other representation) without pulling PIL into
    carl-studio's dependency surface.
    """

    screenshot: object
    phi: float
    action_type: str
    x: int
    y: int
    reward: float
    state_changed: bool
    response: str


class FrameBuffer:
    """Tracks observations and metrics across steps for temporal reasoning.

    Maintains a rolling window of frames with Phi trajectory analysis,
    Kuramoto order parameter, and trend detection.  Foundation for video
    understanding — the agent reasons about state transitions across frames.
    """

    def __init__(self, max_frames: int = 32) -> None:
        self._records: deque[FrameRecord] = deque(maxlen=max_frames)

    # -- mutation --------------------------------------------------------

    def add(
        self,
        screenshot: object,
        phi: float,
        action_type: str,
        x: int = 0,
        y: int = 0,
        reward: float = 0.0,
        state_changed: bool = False,
        response: str = "",
    ) -> None:
        self._records.append(
            FrameRecord(
                screenshot=screenshot,
                phi=phi,
                action_type=action_type,
                x=x,
                y=y,
                reward=reward,
                state_changed=state_changed,
                response=response,
            )
        )

    # -- properties ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    @property
    def phi_values(self) -> list[float]:
        return [r.phi for r in self._records]

    @property
    def rewards(self) -> list[float]:
        return [r.reward for r in self._records]

    @property
    def recent(self) -> FrameRecord | None:
        return self._records[-1] if self._records else None

    # -- analysis --------------------------------------------------------

    def temporal_coherence(self) -> dict:
        """Compute cross-step coherence from Phi trajectory.

        Returns metrics about how coherently the agent is behaving across
        multiple steps.  Key for detecting phase transitions during deployment.
        """
        if len(self._records) < 2:
            return {
                "mean_phi": 0.0,
                "phi_std": 0.0,
                "phi_trend": 0.0,
                "n_frames": len(self._records),
                "kuramoto_R": 0.0,
            }

        phi_arr = np.array(self.phi_values)
        phi_std = float(np.std(phi_arr))
        mean_phi = float(np.mean(phi_arr))

        # Linear trend: positive = crystallizing, negative = melting.
        trend = 0.0
        if len(phi_arr) >= 3:
            coeffs = np.polyfit(range(len(phi_arr)), phi_arr, 1)
            trend = float(coeffs[0])

        return {
            "mean_phi": mean_phi,
            "phi_std": phi_std,
            "phi_trend": trend,
            "n_frames": len(self._records),
            "synchronization": self._synchronization_index(),
            "reward_sum": float(sum(self.rewards)),
            "state_changes": sum(1 for r in self._records if r.state_changed),
        }

    def _synchronization_index(self) -> float:
        """Phi trajectory consistency index. Range [0, 1].

        Higher values indicate more consistent phi across the buffer window.
        Implementation details in terminals-runtime.
        """
        if len(self._records) < 2:
            return 0.0
        try:
            from terminals_runtime.primitives import kuramoto_R
            return kuramoto_R(self.phi_values)
        except ImportError:
            # Fallback: simple coefficient of variation inverse
            phi_arr = np.array(self.phi_values)
            if phi_arr.std() < 1e-8:
                return 1.0
            return float(max(0.0, 1.0 - phi_arr.std() / max(phi_arr.mean(), 1e-8)))

    def detect_phase_transition(
        self, window: int = 5, threshold: float = 0.15
    ) -> dict:
        """Detect if a phase transition is occurring in the Phi trajectory.

        Returns transition type (crystallization / melting / none) and
        magnitude.
        """
        if len(self._records) < window * 2:
            return {"type": "insufficient_data", "magnitude": 0.0}

        phi_arr = np.array(self.phi_values)
        early = phi_arr[-window * 2 : -window]
        late = phi_arr[-window:]

        delta = float(np.mean(late) - np.mean(early))
        magnitude = abs(delta)

        if magnitude < threshold:
            return {"type": "stable", "magnitude": magnitude}
        elif delta > 0:
            return {"type": "crystallization", "magnitude": magnitude}
        else:
            return {"type": "melting", "magnitude": magnitude}
