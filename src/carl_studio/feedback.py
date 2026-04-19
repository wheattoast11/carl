"""Training-feedback engine: bridge heartbeat-evaluated sticky notes back
into concrete ``TrainingProposal`` objects that the user can accept.

This closes the intelligence loop:
    sticky -> heartbeat cycle -> eval -> baseline diff ->
    proposal -> user confirms -> ``carl train --from-queue`` -> milestone.

The engine is intentionally independent of the heartbeat orchestrator. It
exposes three operations:

- :py:meth:`FeedbackEngine.record_baseline` -- pin a checkpoint/metrics
  snapshot as the comparison target for future eval runs.
- :py:meth:`FeedbackEngine.evaluate_against_note` -- run the eval runner for
  a given checkpoint, tagged with a note id, and return the :py:class:`EvalReport`.
- :py:meth:`FeedbackEngine.propose` -- diff the eval report against the
  baseline and emit a :py:class:`TrainingProposal`.

Proposals persist to the local config store (``carl.feedback.pending_proposal``)
so ``carl train --from-queue`` can pick them up in a later session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.db import LocalDB
from carl_studio.sticky import StickyNote

if TYPE_CHECKING:
    from carl_studio.eval.runner import EvalReport


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EvalBaseline(BaseModel):
    """Pinned reference point for future eval comparisons."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    checkpoint: str
    metrics: dict[str, float] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=_now_iso)


ProposalType = Literal["curriculum", "reward_tweak", "checkpoint_rollback"]


class TrainingProposal(BaseModel):
    """Structured suggestion derived from a sticky-note/eval diff."""

    model_config = ConfigDict(extra="forbid")

    note_id: str
    eval_baseline: EvalBaseline | None = None
    eval_result: dict[str, Any] = Field(default_factory=dict)
    gap_summary: str = ""
    proposal_type: ProposalType = "curriculum"
    suggested_config: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Persistence keys (config table)
# ---------------------------------------------------------------------------

_DB_KEY_BASELINE = "carl.feedback.eval_baseline"
_DB_KEY_PENDING = "carl.feedback.pending_proposal"


# Thresholds used by the gap-diagnosis heuristic. Keeping them module-level
# makes them trivially tunable from tests without pickling mocks.
_REGRESSION_DELTA = -0.05
_PLATEAU_BAND = 0.01


class FeedbackEngine:
    """Close the loop between sticky-note evals and training proposals."""

    def __init__(self, db: LocalDB, chain: InteractionChain | None = None) -> None:
        if db is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError("FeedbackEngine: db must be a LocalDB instance")
        self._db = db
        # NB: ``InteractionChain.__len__`` means an empty chain is falsy --
        # use an explicit identity check so callers can pass a fresh chain.
        self._chain = chain if chain is not None else InteractionChain()

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def record_baseline(
        self,
        checkpoint: str,
        metrics: dict[str, float],
        run_id: str = "",
    ) -> EvalBaseline:
        """Persist a new baseline snapshot. Overwrites the previous value."""
        if not checkpoint or not checkpoint.strip():
            raise ValueError("FeedbackEngine.record_baseline: checkpoint must be non-empty")

        baseline = EvalBaseline(
            run_id=run_id or checkpoint,
            checkpoint=checkpoint,
            metrics=dict(metrics),
        )
        self._db.set_config(_DB_KEY_BASELINE, baseline.model_dump_json())
        self._chain.record(
            ActionType.CHECKPOINT,
            "feedback:baseline",
            input={"checkpoint": checkpoint, "run_id": baseline.run_id},
            output={"metrics": dict(metrics)},
            success=True,
            duration_ms=0.0,
        )
        return baseline

    def current_baseline(self) -> EvalBaseline | None:
        """Return the persisted baseline, or ``None`` if none has been recorded."""
        raw = self._db.get_config(_DB_KEY_BASELINE)
        if not raw:
            return None
        try:
            return EvalBaseline.model_validate_json(raw)
        except ValueError:
            # Corrupted row -- treat as absent rather than raising at callers.
            return None

    # ------------------------------------------------------------------
    # Evaluation (lazy-imports EvalRunner to preserve import-time lightness)
    # ------------------------------------------------------------------

    def evaluate_against_note(
        self,
        note: StickyNote,
        checkpoint: str,
    ) -> EvalReport:
        """Run an eval for ``checkpoint`` tagged to ``note.id``.

        Heavy training dependencies are imported lazily so ``feedback`` stays
        cheap to import from the CLI.
        """
        if not checkpoint or not checkpoint.strip():
            raise ValueError("FeedbackEngine.evaluate_against_note: checkpoint required")

        from carl_studio.eval.runner import EvalConfig, EvalRunner

        dataset = (note.content or "").strip()[:80] or "wikitext"
        cfg = EvalConfig(checkpoint=checkpoint, dataset=dataset)
        runner = EvalRunner(cfg, interaction_chain=self._chain)
        report = runner.run()

        self._chain.record(
            ActionType.EVAL_PHASE,
            "feedback:evaluate",
            input={"note_id": note.id, "checkpoint": checkpoint, "dataset": dataset},
            output={
                "passed": getattr(report, "passed", None),
                "primary_metric": getattr(report, "primary_metric", None),
                "primary_value": getattr(report, "primary_value", None),
            },
            success=bool(getattr(report, "passed", False)),
            duration_ms=0.0,
        )
        return report

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose(
        self,
        note: StickyNote,
        eval_report: EvalReport,
        baseline: EvalBaseline | None = None,
    ) -> TrainingProposal:
        """Diff ``eval_report`` against ``baseline`` and persist a proposal."""
        metrics = dict(getattr(eval_report, "metrics", {}) or {})
        baseline = baseline if baseline is not None else self.current_baseline()

        gap, proposal_type, suggested = self._diagnose_gap(metrics, baseline)
        proposal = TrainingProposal(
            note_id=note.id,
            eval_baseline=baseline,
            eval_result={
                "metrics": metrics,
                "passed": getattr(eval_report, "passed", None),
                "primary_metric": getattr(eval_report, "primary_metric", None),
                "primary_value": getattr(eval_report, "primary_value", None),
            },
            gap_summary=gap,
            proposal_type=proposal_type,
            suggested_config=suggested,
            confidence=self._confidence(metrics, baseline),
        )
        self._db.set_config(_DB_KEY_PENDING, proposal.model_dump_json())
        self._chain.record(
            ActionType.CHECKPOINT,
            "feedback:propose",
            input={"note_id": note.id, "baseline_present": baseline is not None},
            output={
                "proposal_type": proposal_type,
                "confidence": proposal.confidence,
                "gap_summary": gap,
            },
            success=True,
            duration_ms=0.0,
        )
        return proposal

    def pending_proposal(self) -> TrainingProposal | None:
        """Return the persisted pending proposal, or ``None``."""
        raw = self._db.get_config(_DB_KEY_PENDING)
        if not raw:
            return None
        try:
            return TrainingProposal.model_validate_json(raw)
        except ValueError:
            return None

    def accept(self, proposal: TrainingProposal) -> None:
        """Mark a proposal as consumed; the next train cycle updates the baseline."""
        if not isinstance(proposal, TrainingProposal):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("FeedbackEngine.accept: proposal must be a TrainingProposal")
        self._db.set_config(_DB_KEY_PENDING, "")
        self._chain.record(
            ActionType.CHECKPOINT,
            "feedback:accept",
            input={"note_id": proposal.note_id, "proposal_type": proposal.proposal_type},
            output={"accepted": True},
            success=True,
            duration_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Diagnosis helpers
    # ------------------------------------------------------------------

    def _diagnose_gap(
        self,
        metrics: dict[str, float],
        baseline: EvalBaseline | None,
    ) -> tuple[str, ProposalType, dict[str, Any]]:
        """Classify the eval delta.

        Decision table (in order):

        1. No baseline -> seed the loop by recording one next cycle.
        2. Any metric regressed more than 5 pp -> propose rollback.
        3. All metrics inside a +/-1 pp band -> plateau; raise curriculum difficulty.
        4. Otherwise -> incremental gain; tune reward weights.
        """
        if baseline is None:
            return (
                "no baseline -- record one",
                "curriculum",
                {"action": "record_baseline"},
            )

        shared_keys = [k for k in baseline.metrics if k in metrics]
        if not shared_keys:
            # No overlap -- nothing actionable to diff against.
            return (
                "no overlapping metrics with baseline -- record a new baseline",
                "curriculum",
                {"action": "record_baseline"},
            )

        deltas = {k: metrics[k] - baseline.metrics[k] for k in shared_keys}
        worst_key, worst_delta = min(deltas.items(), key=lambda kv: kv[1])

        if worst_delta <= _REGRESSION_DELTA:
            return (
                f"{worst_key} regressed by {worst_delta:+.3f}",
                "checkpoint_rollback",
                {
                    "action": "rollback",
                    "metric": worst_key,
                    "delta": worst_delta,
                },
            )

        if all(abs(d) < _PLATEAU_BAND for d in deltas.values()):
            return (
                "plateau detected -- raise curriculum difficulty",
                "curriculum",
                {"action": "raise_difficulty", "step": 1},
            )

        return (
            "incremental gain -- tune reward weights",
            "reward_tweak",
            {"action": "perturb_reward_weights", "magnitude": 0.1},
        )

    def _confidence(
        self,
        metrics: dict[str, float],
        baseline: EvalBaseline | None,
    ) -> float:
        """Confidence grows with the number of shared metric keys."""
        if baseline is None or not metrics:
            return 0.3
        n_shared = sum(1 for k in baseline.metrics if k in metrics)
        return min(0.95, 0.3 + 0.15 * n_shared)


__all__ = [
    "EvalBaseline",
    "FeedbackEngine",
    "ProposalType",
    "TrainingProposal",
]
