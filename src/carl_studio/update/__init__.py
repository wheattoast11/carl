"""carl update — self-updating agentic meta-pipeline (v0.11).

Surfaces (no auto-apply):
- Recent git commits (deltas since last run)
- Dependency-version deltas vs PyPI
- Positive-framed blast-radius summary

Respects consent: every network call is gated by consent_gate("telemetry").
Back-off + retry via carl_core's BreakAndRetryStrategy.
"""

from __future__ import annotations

from carl_studio.update.report import (
    BlastRadiusEntry,
    DependencyDelta,
    GitCommitDelta,
    UpdateReport,
    build_update_report,
)

__all__ = [
    "BlastRadiusEntry",
    "DependencyDelta",
    "GitCommitDelta",
    "UpdateReport",
    "build_update_report",
]
