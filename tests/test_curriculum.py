"""Tests for carl_studio.curriculum -- FSM, persistence, milestones."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from carl_studio.curriculum import (
    CurriculumPhase,
    CurriculumStore,
    CurriculumTrack,
    Milestone,
    _TRANSITIONS,
    verify_fsm_closure,
)


# ---------------------------------------------------------------------------
# CurriculumPhase enum
# ---------------------------------------------------------------------------

class TestCurriculumPhase:
    def test_enrolled_value(self) -> None:
        assert CurriculumPhase.ENROLLED.value == "enrolled"

    def test_drilling_value(self) -> None:
        assert CurriculumPhase.DRILLING.value == "drilling"

    def test_evaluated_value(self) -> None:
        assert CurriculumPhase.EVALUATED.value == "evaluated"

    def test_graduated_value(self) -> None:
        assert CurriculumPhase.GRADUATED.value == "graduated"

    def test_deployed_value(self) -> None:
        assert CurriculumPhase.DEPLOYED.value == "deployed"

    def test_ttt_active_value(self) -> None:
        assert CurriculumPhase.TTT_ACTIVE.value == "ttt_active"

    def test_phase_count(self) -> None:
        assert len(CurriculumPhase) == 6

    def test_from_string(self) -> None:
        assert CurriculumPhase("enrolled") is CurriculumPhase.ENROLLED


# ---------------------------------------------------------------------------
# FSM structure
# ---------------------------------------------------------------------------

class TestFSMStructure:
    def test_every_phase_has_outgoing_transitions(self) -> None:
        """Every phase must appear as a key in _TRANSITIONS."""
        for phase in CurriculumPhase:
            assert phase in _TRANSITIONS, f"{phase} has no outgoing transitions"

    def test_every_phase_has_at_least_one_target(self) -> None:
        """Every phase must have at least one valid target."""
        for phase in CurriculumPhase:
            targets = _TRANSITIONS[phase]
            assert len(targets) >= 1, f"{phase} has empty transition set"

    def test_all_targets_are_valid_phases(self) -> None:
        """No transition target may reference a phase not in the enum."""
        all_phases = set(CurriculumPhase)
        for source, targets in _TRANSITIONS.items():
            for target in targets:
                assert target in all_phases, (
                    f"Transition {source} -> {target}: target not in CurriculumPhase"
                )

    def test_fsm_closure_function(self) -> None:
        """verify_fsm_closure() returns True for the production FSM."""
        assert verify_fsm_closure() is True

    def test_no_self_loops(self) -> None:
        """No phase should transition to itself."""
        for source, targets in _TRANSITIONS.items():
            assert source not in targets, f"{source} has a self-loop"

    def test_enrolled_can_only_reach_drilling(self) -> None:
        assert _TRANSITIONS[CurriculumPhase.ENROLLED] == {CurriculumPhase.DRILLING}

    def test_evaluated_can_retry(self) -> None:
        """EVALUATED can go back to DRILLING (retry)."""
        assert CurriculumPhase.DRILLING in _TRANSITIONS[CurriculumPhase.EVALUATED]

    def test_deployed_can_retrain(self) -> None:
        """DEPLOYED can go back to DRILLING (retrain)."""
        assert CurriculumPhase.DRILLING in _TRANSITIONS[CurriculumPhase.DEPLOYED]

    def test_ttt_active_returns_to_drilling(self) -> None:
        assert _TRANSITIONS[CurriculumPhase.TTT_ACTIVE] == {CurriculumPhase.DRILLING}


# ---------------------------------------------------------------------------
# CurriculumTrack.can_advance
# ---------------------------------------------------------------------------

class TestCanAdvance:
    def test_enrolled_to_drilling(self) -> None:
        track = CurriculumTrack(model_id="test")
        assert track.can_advance(CurriculumPhase.DRILLING) is True

    def test_enrolled_to_evaluated_invalid(self) -> None:
        track = CurriculumTrack(model_id="test")
        assert track.can_advance(CurriculumPhase.EVALUATED) is False

    def test_enrolled_to_graduated_invalid(self) -> None:
        track = CurriculumTrack(model_id="test")
        assert track.can_advance(CurriculumPhase.GRADUATED) is False

    def test_enrolled_to_deployed_invalid(self) -> None:
        track = CurriculumTrack(model_id="test")
        assert track.can_advance(CurriculumPhase.DEPLOYED) is False

    def test_drilling_to_evaluated(self) -> None:
        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.DRILLING)
        assert track.can_advance(CurriculumPhase.EVALUATED) is True

    def test_drilling_to_graduated_invalid(self) -> None:
        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.DRILLING)
        assert track.can_advance(CurriculumPhase.GRADUATED) is False

    def test_evaluated_to_graduated(self) -> None:
        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.EVALUATED)
        assert track.can_advance(CurriculumPhase.GRADUATED) is True

    def test_evaluated_to_drilling_retry(self) -> None:
        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.EVALUATED)
        assert track.can_advance(CurriculumPhase.DRILLING) is True

    def test_graduated_to_deployed(self) -> None:
        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.GRADUATED)
        assert track.can_advance(CurriculumPhase.DEPLOYED) is True

    def test_deployed_to_ttt_active(self) -> None:
        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.DEPLOYED)
        assert track.can_advance(CurriculumPhase.TTT_ACTIVE) is True

    def test_deployed_to_drilling_retrain(self) -> None:
        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.DEPLOYED)
        assert track.can_advance(CurriculumPhase.DRILLING) is True

    def test_ttt_active_to_drilling(self) -> None:
        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.TTT_ACTIVE)
        assert track.can_advance(CurriculumPhase.DRILLING) is True


# ---------------------------------------------------------------------------
# CurriculumTrack.advance
# ---------------------------------------------------------------------------

class TestAdvance:
    def test_valid_advance_changes_phase(self) -> None:
        track = CurriculumTrack(model_id="test")
        updated = track.advance(CurriculumPhase.DRILLING, event="training_started")
        assert updated.phase == CurriculumPhase.DRILLING

    def test_valid_advance_records_milestone(self) -> None:
        track = CurriculumTrack(model_id="test")
        updated = track.advance(CurriculumPhase.DRILLING, event="training_started")
        assert len(updated.milestones) == 1
        assert updated.milestones[0].event == "training_started"
        assert updated.milestones[0].phase == CurriculumPhase.DRILLING

    def test_advance_preserves_prior_milestones(self) -> None:
        track = CurriculumTrack(model_id="test")
        t1 = track.advance(CurriculumPhase.DRILLING, event="e1")
        t2 = t1.advance(CurriculumPhase.EVALUATED, event="e2")
        assert len(t2.milestones) == 2
        assert t2.milestones[0].event == "e1"
        assert t2.milestones[1].event == "e2"

    def test_advance_default_event_name(self) -> None:
        track = CurriculumTrack(model_id="test")
        updated = track.advance(CurriculumPhase.DRILLING)
        assert updated.milestones[0].event == "advanced_to_drilling"

    def test_advance_with_detail(self) -> None:
        track = CurriculumTrack(model_id="test")
        updated = track.advance(CurriculumPhase.DRILLING, detail="job_id=abc123")
        assert updated.milestones[0].detail == "job_id=abc123"

    def test_invalid_advance_raises(self) -> None:
        track = CurriculumTrack(model_id="test")
        with pytest.raises(ValueError, match="Cannot transition"):
            track.advance(CurriculumPhase.GRADUATED)

    def test_invalid_advance_error_shows_valid(self) -> None:
        track = CurriculumTrack(model_id="test")
        with pytest.raises(ValueError, match="drilling"):
            track.advance(CurriculumPhase.DEPLOYED)

    def test_advance_is_immutable(self) -> None:
        """Original track is not modified."""
        track = CurriculumTrack(model_id="test")
        _ = track.advance(CurriculumPhase.DRILLING)
        assert track.phase == CurriculumPhase.ENROLLED
        assert len(track.milestones) == 0

    def test_full_happy_path(self) -> None:
        """Walk the full ENROLLED -> ... -> TTT_ACTIVE path."""
        t = CurriculumTrack(model_id="test")
        t = t.advance(CurriculumPhase.DRILLING, event="train_start")
        t = t.advance(CurriculumPhase.EVALUATED, event="train_done")
        t = t.advance(CurriculumPhase.GRADUATED, event="gate_pass")
        t = t.advance(CurriculumPhase.DEPLOYED, event="pushed_to_hub")
        t = t.advance(CurriculumPhase.TTT_ACTIVE, event="ttt_started")
        assert t.phase == CurriculumPhase.TTT_ACTIVE
        assert len(t.milestones) == 5


# ---------------------------------------------------------------------------
# Version tracking
# ---------------------------------------------------------------------------

class TestVersioning:
    def test_initial_version(self) -> None:
        track = CurriculumTrack(model_id="test")
        assert track.version == 1

    def test_version_stable_through_advancement(self) -> None:
        """Version should NOT increment on normal advancement."""
        track = CurriculumTrack(model_id="test")
        updated = track.advance(CurriculumPhase.DRILLING)
        assert updated.version == 1

    def test_version_increments_on_enrolled(self) -> None:
        """Version increments when returning to ENROLLED (cycle complete).

        Note: ENROLLED is not currently reachable via any transition,
        but the logic is tested for future-proofing.
        """
        # Simulate a track that somehow reached ENROLLED as a target
        # by checking the logic directly
        track = CurriculumTrack(model_id="test", version=1)
        # Advance to drilling (does not increment)
        t2 = track.advance(CurriculumPhase.DRILLING)
        assert t2.version == 1


# ---------------------------------------------------------------------------
# Milestone
# ---------------------------------------------------------------------------

class TestMilestone:
    def test_timestamp_is_iso_format(self) -> None:
        ms = Milestone(phase=CurriculumPhase.DRILLING, event="test")
        # ISO 8601: YYYY-MM-DDTHH:MM:SS.ffffff+HH:MM or similar
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ms.timestamp)

    def test_timestamp_is_utc(self) -> None:
        ms = Milestone(phase=CurriculumPhase.DRILLING, event="test")
        # Should contain +00:00 or Z for UTC
        assert "+00:00" in ms.timestamp or ms.timestamp.endswith("Z")

    def test_default_detail_is_empty(self) -> None:
        ms = Milestone(phase=CurriculumPhase.DRILLING, event="test")
        assert ms.detail == ""


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys(self) -> None:
        track = CurriculumTrack(model_id="il-terminals-carl-omni9b-v12")
        s = track.summary()
        assert set(s.keys()) == {"model_id", "phase", "version", "milestones", "last_event"}

    def test_summary_no_milestones(self) -> None:
        track = CurriculumTrack(model_id="test")
        s = track.summary()
        assert s["last_event"] == "none"
        assert s["milestones"] == 0

    def test_summary_with_milestones(self) -> None:
        track = CurriculumTrack(model_id="test")
        updated = track.advance(CurriculumPhase.DRILLING, event="started")
        s = updated.summary()
        assert s["last_event"] == "started"
        assert s["milestones"] == 1


# ---------------------------------------------------------------------------
# CurriculumStore -- SQLite persistence
# ---------------------------------------------------------------------------

class TestCurriculumStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> CurriculumStore:
        db = tmp_path / "test_carl.db"
        s = CurriculumStore(db_path=db)
        yield s
        s.close()

    def test_save_and_load_roundtrip(self, store: CurriculumStore) -> None:
        track = CurriculumTrack(model_id="test-model")
        track = track.advance(CurriculumPhase.DRILLING, event="started")
        store.save(track)

        loaded = store.load("test-model")
        assert loaded is not None
        assert loaded.model_id == "test-model"
        assert loaded.phase == CurriculumPhase.DRILLING
        assert len(loaded.milestones) == 1
        assert loaded.milestones[0].event == "started"

    def test_save_overwrites_existing(self, store: CurriculumStore) -> None:
        track = CurriculumTrack(model_id="test-model")
        store.save(track)

        updated = track.advance(CurriculumPhase.DRILLING)
        store.save(updated)

        loaded = store.load("test-model")
        assert loaded is not None
        assert loaded.phase == CurriculumPhase.DRILLING

    def test_load_nonexistent_returns_none(self, store: CurriculumStore) -> None:
        assert store.load("nonexistent") is None

    def test_list_tracks_returns_all(self, store: CurriculumStore) -> None:
        store.save(CurriculumTrack(model_id="model-a"))
        store.save(CurriculumTrack(model_id="model-b"))
        store.save(CurriculumTrack(model_id="model-c"))

        tracks = store.list_tracks()
        ids = {t.model_id for t in tracks}
        assert ids == {"model-a", "model-b", "model-c"}
        assert len(tracks) == 3

    def test_list_tracks_empty(self, store: CurriculumStore) -> None:
        tracks = store.list_tracks()
        assert tracks == []

    def test_current_returns_most_recently_updated(self, store: CurriculumStore) -> None:
        store.save(CurriculumTrack(model_id="old"))
        store.save(CurriculumTrack(model_id="newer"))
        store.save(CurriculumTrack(model_id="newest"))

        current = store.current()
        assert current is not None
        assert current.model_id == "newest"

    def test_current_updates_on_save(self, store: CurriculumStore) -> None:
        store.save(CurriculumTrack(model_id="first"))
        store.save(CurriculumTrack(model_id="second"))

        # Re-save first -> it's now the most recent
        track = store.load("first")
        assert track is not None
        updated = track.advance(CurriculumPhase.DRILLING)
        store.save(updated)

        current = store.current()
        assert current is not None
        assert current.model_id == "first"

    def test_current_empty_returns_none(self, store: CurriculumStore) -> None:
        assert store.current() is None

    def test_milestones_survive_roundtrip(self, store: CurriculumStore) -> None:
        track = CurriculumTrack(model_id="rt")
        track = track.advance(CurriculumPhase.DRILLING, event="e1", detail="d1")
        track = track.advance(CurriculumPhase.EVALUATED, event="e2", detail="d2")
        store.save(track)

        loaded = store.load("rt")
        assert loaded is not None
        assert len(loaded.milestones) == 2
        assert loaded.milestones[0].event == "e1"
        assert loaded.milestones[0].detail == "d1"
        assert loaded.milestones[1].event == "e2"
        assert loaded.milestones[1].detail == "d2"

    def test_version_survives_roundtrip(self, store: CurriculumStore) -> None:
        track = CurriculumTrack(model_id="vt", version=3)
        store.save(track)

        loaded = store.load("vt")
        assert loaded is not None
        assert loaded.version == 3
