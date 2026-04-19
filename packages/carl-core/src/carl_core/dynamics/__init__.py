"""Trajectory dynamics probes for CARL training loops.

Exports primitives that measure whether training trajectories satisfy
the framework's dynamical claims (contraction, conservation law, etc.).
"""
from __future__ import annotations

from carl_core.dynamics.trajectory import ContractionProbe, ContractionReport

__all__ = ["ContractionProbe", "ContractionReport"]
