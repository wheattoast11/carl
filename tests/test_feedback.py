"""Tests for carl_studio.feedback (ARC-005 training-feedback engine)."""

from __future__ import annotations

from pathlib import Path

import pytest

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.db import LocalDB
from carl_studio.feedback import (
    EvalBaseline,
    FeedbackEngine,
    TrainingProposal,
)
from carl_studio.sticky import StickyNote


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> LocalDB:
    """Isolated LocalDB per test, rooted in tmp_path."""
    return LocalDB(tmp_path / "carl.db")


@pytest.fixture()
def chain() -> InteractionChain:
    return InteractionChain()


@pytest.fixture()
def engine(db: LocalDB, chain: InteractionChain) -> FeedbackEngine:
    return FeedbackEngine(db, chain=chain)


@pytest.fixture()
def note() -> StickyNote:
    return StickyNote(content="probe checkpoint drift on wikitext")


class _StubReport:
    """Minimal EvalReport-shaped stub -- avoids importing the training extras."""

    def __init__(
        self,
        metrics: dict[str, float],
        *,
        passed: bool = True,
        primary_metric: str = "accuracy",
        primary_value: float = 0.0,
    ) -> None:
        self.metrics = metrics
        self.passed = passed
        self.primary_metric = primary_metric
        self.primary_value = primary_value


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------


def test_record_baseline_persists_to_db(engine: FeedbackEngine, db: LocalDB) -> None:
    baseline = engine.record_baseline(
        "org/ckpt-v1",
        {"accuracy": 0.72, "f1": 0.68},
        run_id="run-42",
    )

    assert isinstance(baseline, EvalBaseline)
    assert baseline.checkpoint == "org/ckpt-v1"
    assert baseline.run_id == "run-42"
    assert baseline.metrics == {"accuracy": 0.72, "f1": 0.68}
    assert baseline.recorded_at  # non-empty timestamp

    raw = db.get_config("carl.feedback.eval_baseline")
    assert raw and "org/ckpt-v1" in raw


def test_record_baseline_rejects_empty_checkpoint(engine: FeedbackEngine) -> None:
    with pytest.raises(ValueError):
        engine.record_baseline("", {"accuracy": 0.5})
    with pytest.raises(ValueError):
        engine.record_baseline("   ", {"accuracy": 0.5})


def test_record_baseline_records_to_chain(
    engine: FeedbackEngine, chain: InteractionChain
) -> None:
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.72})

    ckpt_steps = [s for s in chain.steps if s.action == ActionType.CHECKPOINT]
    assert any(s.name == "feedback:baseline" for s in ckpt_steps)


def test_current_baseline_returns_recorded(engine: FeedbackEngine) -> None:
    assert engine.current_baseline() is None

    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.72})
    loaded = engine.current_baseline()

    assert loaded is not None
    assert loaded.checkpoint == "org/ckpt-v1"
    assert loaded.metrics["accuracy"] == pytest.approx(0.72)


def test_current_baseline_overwrites_on_new_record(engine: FeedbackEngine) -> None:
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.72})
    engine.record_baseline("org/ckpt-v2", {"accuracy": 0.80, "f1": 0.75})

    loaded = engine.current_baseline()
    assert loaded is not None
    assert loaded.checkpoint == "org/ckpt-v2"
    assert loaded.metrics == {"accuracy": 0.80, "f1": 0.75}


def test_current_baseline_handles_corrupt_row(engine: FeedbackEngine, db: LocalDB) -> None:
    db.set_config("carl.feedback.eval_baseline", "not-json")
    assert engine.current_baseline() is None


# ---------------------------------------------------------------------------
# Propose -- gap diagnosis
# ---------------------------------------------------------------------------


def test_propose_without_baseline_returns_curriculum_action(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    report = _StubReport({"accuracy": 0.75})

    proposal = engine.propose(note, report)

    assert proposal.proposal_type == "curriculum"
    assert proposal.suggested_config == {"action": "record_baseline"}
    assert "no baseline" in proposal.gap_summary.lower()
    assert proposal.eval_baseline is None
    assert proposal.confidence == pytest.approx(0.3)
    assert proposal.note_id == note.id


def test_propose_regression_triggers_rollback_proposal(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.80, "f1": 0.75})
    # accuracy drops by 10 pp -- clearly past the -5 pp regression band.
    report = _StubReport({"accuracy": 0.70, "f1": 0.75})

    proposal = engine.propose(note, report)

    assert proposal.proposal_type == "checkpoint_rollback"
    assert proposal.suggested_config["action"] == "rollback"
    assert proposal.suggested_config["metric"] == "accuracy"
    assert proposal.suggested_config["delta"] == pytest.approx(-0.10)
    assert "accuracy" in proposal.gap_summary


def test_propose_plateau_raises_difficulty(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.80, "f1": 0.75})
    # All metrics within +/-1 pp.
    report = _StubReport({"accuracy": 0.8005, "f1": 0.7498})

    proposal = engine.propose(note, report)

    assert proposal.proposal_type == "curriculum"
    assert proposal.suggested_config == {"action": "raise_difficulty", "step": 1}
    assert "plateau" in proposal.gap_summary.lower()


def test_propose_incremental_tunes_reward(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.80, "f1": 0.75})
    # Moderate gain that does not trigger plateau band and does not regress.
    report = _StubReport({"accuracy": 0.83, "f1": 0.78})

    proposal = engine.propose(note, report)

    assert proposal.proposal_type == "reward_tweak"
    assert proposal.suggested_config["action"] == "perturb_reward_weights"
    assert proposal.suggested_config["magnitude"] == pytest.approx(0.1)
    assert "incremental" in proposal.gap_summary.lower()


def test_propose_with_disjoint_metrics_requests_new_baseline(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.80})
    report = _StubReport({"rouge_l": 0.42})

    proposal = engine.propose(note, report)

    assert proposal.proposal_type == "curriculum"
    assert proposal.suggested_config == {"action": "record_baseline"}


# ---------------------------------------------------------------------------
# Pending proposal round-trip
# ---------------------------------------------------------------------------


def test_pending_proposal_round_trips(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.80, "f1": 0.75})
    report = _StubReport({"accuracy": 0.83, "f1": 0.78})

    proposal = engine.propose(note, report)

    loaded = engine.pending_proposal()
    assert isinstance(loaded, TrainingProposal)
    assert loaded.note_id == proposal.note_id
    assert loaded.proposal_type == proposal.proposal_type
    assert loaded.gap_summary == proposal.gap_summary
    assert loaded.suggested_config == proposal.suggested_config
    assert loaded.confidence == pytest.approx(proposal.confidence)
    assert loaded.eval_baseline is not None
    assert loaded.eval_baseline.checkpoint == "org/ckpt-v1"


def test_pending_proposal_absent_returns_none(engine: FeedbackEngine) -> None:
    assert engine.pending_proposal() is None


def test_pending_proposal_handles_corrupt_row(
    engine: FeedbackEngine, db: LocalDB
) -> None:
    db.set_config("carl.feedback.pending_proposal", "{bad json")
    assert engine.pending_proposal() is None


def test_accept_clears_pending_proposal(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.80})
    report = _StubReport({"accuracy": 0.83})
    proposal = engine.propose(note, report)

    assert engine.pending_proposal() is not None
    engine.accept(proposal)
    assert engine.pending_proposal() is None


def test_accept_rejects_non_proposal(engine: FeedbackEngine) -> None:
    with pytest.raises(TypeError):
        engine.accept("not-a-proposal")  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Confidence heuristic
# ---------------------------------------------------------------------------


def test_confidence_scales_with_shared_metric_count(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    engine.record_baseline("org/ckpt-v1", {"a": 0.5, "b": 0.5, "c": 0.5})
    report = _StubReport({"a": 0.6, "b": 0.6, "c": 0.6})

    proposal = engine.propose(note, report)

    # 3 shared metrics -> 0.3 + 0.15*3 = 0.75
    assert proposal.confidence == pytest.approx(0.75)


def test_confidence_no_baseline_is_floor(
    engine: FeedbackEngine, note: StickyNote
) -> None:
    report = _StubReport({"a": 0.6})
    proposal = engine.propose(note, report)
    assert proposal.confidence == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_engine_rejects_none_db() -> None:
    with pytest.raises(ValueError):
        FeedbackEngine(None)  # pyright: ignore[reportArgumentType]


def test_engine_uses_supplied_chain(db: LocalDB) -> None:
    chain = InteractionChain()
    engine = FeedbackEngine(db, chain=chain)
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.72})
    assert any(s.name == "feedback:baseline" for s in chain.steps)


def test_engine_defaults_to_fresh_chain(db: LocalDB) -> None:
    engine = FeedbackEngine(db)
    engine.record_baseline("org/ckpt-v1", {"accuracy": 0.72})
    # Internal chain should have accumulated the baseline record.
    internal = engine._chain  # pyright: ignore[reportPrivateUsage]
    assert any(s.name == "feedback:baseline" for s in internal.steps)
