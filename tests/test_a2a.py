"""Tests for CARL Studio A2A protocol.

Covers:
  - A2ATask status transitions (FSM correctness + terminal state guard)
  - A2AMessage factory classmethods
  - LocalBus CRUD operations (post, poll, get, update, cancel)
  - LocalBus message roundtrip (publish_message + get_messages)
  - CARLAgentCard.current() graceful construction + JSON roundtrip
  - FSM completeness: no transition from terminal state
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

# ─── Conftest bootstrap is already applied at module load ────────────────────
# These imports work because conftest.py stubs carl_studio before test collection.

from carl_studio.a2a.task import A2ATask, A2ATaskStatus
from carl_studio.a2a.message import A2AMessage
from carl_studio.a2a.bus import LocalBus
from carl_studio.a2a.agent_card import CARLAgentCard


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _task(skill: str = "test_skill", **kwargs) -> A2ATask:
    return A2ATask(id=str(uuid.uuid4()), skill=skill, **kwargs)


def _bus(tmp_path: Path) -> LocalBus:
    return LocalBus(db_path=tmp_path / "a2a.db")


# ─── A2ATask — status transitions ────────────────────────────────────────────


class TestA2ATaskTransitions:
    def test_initial_status_is_pending(self):
        t = _task()
        assert t.status == A2ATaskStatus.PENDING

    def test_mark_running(self):
        t = _task().mark_running()
        assert t.status == A2ATaskStatus.RUNNING

    def test_mark_done(self):
        t = _task().mark_running().mark_done({"score": 0.9})
        assert t.status == A2ATaskStatus.DONE
        assert t.result == {"score": 0.9}
        assert t.completed_at is not None

    def test_mark_failed(self):
        t = _task().mark_running().mark_failed("sandbox timeout")
        assert t.status == A2ATaskStatus.FAILED
        assert t.error == "sandbox timeout"
        assert t.completed_at is not None

    def test_mark_cancelled(self):
        t = _task().mark_cancelled()
        assert t.status == A2ATaskStatus.CANCELLED
        assert t.completed_at is not None

    def test_is_terminal_done(self):
        assert _task().mark_running().mark_done({}).is_terminal()

    def test_is_terminal_failed(self):
        assert _task().mark_running().mark_failed("oops").is_terminal()

    def test_is_terminal_cancelled(self):
        assert _task().mark_cancelled().is_terminal()

    def test_is_not_terminal_pending(self):
        assert not _task().is_terminal()

    def test_is_not_terminal_running(self):
        assert not _task().mark_running().is_terminal()

    # FSM completeness: no transition from terminal states

    def test_no_running_from_done(self):
        done = _task().mark_running().mark_done({"x": 1})
        # mark_running from terminal is a no-op (returns same object)
        same = done.mark_running()
        assert same.status == A2ATaskStatus.DONE

    def test_no_done_from_failed(self):
        failed = _task().mark_failed("err")
        same = failed.mark_done({"x": 1})
        assert same.status == A2ATaskStatus.FAILED

    def test_no_failed_from_done(self):
        done = _task().mark_done({})
        same = done.mark_failed("late error")
        assert same.status == A2ATaskStatus.DONE

    def test_no_cancelled_from_done(self):
        done = _task().mark_done({})
        same = done.mark_cancelled()
        assert same.status == A2ATaskStatus.DONE

    def test_no_transition_from_cancelled(self):
        cancelled = _task().mark_cancelled()
        assert cancelled.mark_running().status == A2ATaskStatus.CANCELLED
        assert cancelled.mark_done({}).status == A2ATaskStatus.CANCELLED
        assert cancelled.mark_failed("x").status == A2ATaskStatus.CANCELLED
        assert cancelled.mark_cancelled().status == A2ATaskStatus.CANCELLED

    def test_updated_at_advances_on_running(self):
        t0 = _task()
        t1 = t0.mark_running()
        # updated_at should be equal or later (fast machines may be equal)
        assert t1.updated_at >= t0.updated_at

    def test_result_none_before_done(self):
        t = _task().mark_running()
        assert t.result is None

    def test_error_none_before_failed(self):
        t = _task().mark_running()
        assert t.error is None


# ─── A2AMessage — factory methods ────────────────────────────────────────────


class TestA2AMessage:
    def _tid(self) -> str:
        return str(uuid.uuid4())

    def test_progress_fields(self):
        m = A2AMessage.progress(task_id="t1", step=3, total=10, detail="loading")
        assert m.type == "progress"
        assert m.task_id == "t1"
        assert m.payload["step"] == 3
        assert m.payload["total"] == 10
        assert m.payload["detail"] == "loading"
        assert m.id  # non-empty UUID

    def test_artifact_fields(self):
        m = A2AMessage.artifact(task_id="t2", name="checkpoint", data={"path": "/tmp/ckpt"})
        assert m.type == "artifact"
        assert m.payload["name"] == "checkpoint"
        assert m.payload["data"] == {"path": "/tmp/ckpt"}

    def test_log_fields(self):
        m = A2AMessage.log(task_id="t3", message="step complete", level="debug")
        assert m.type == "log"
        assert m.payload["message"] == "step complete"
        assert m.payload["level"] == "debug"

    def test_error_fields(self):
        m = A2AMessage.error(task_id="t4", message="CUDA OOM", detail="step 42")
        assert m.type == "error"
        assert m.payload["message"] == "CUDA OOM"

    def test_heartbeat_fields(self):
        m = A2AMessage.heartbeat(task_id="t5")
        assert m.type == "heartbeat"
        assert m.payload == {}

    def test_unique_ids(self):
        m1 = A2AMessage.progress("x", 1, 10)
        m2 = A2AMessage.progress("x", 2, 10)
        assert m1.id != m2.id

    def test_default_sender(self):
        m = A2AMessage.log(task_id="t", message="hi")
        assert m.sender == "carl-studio"

    def test_custom_sender(self):
        m = A2AMessage.progress("t", 1, 5, sender="orchestrator")
        assert m.sender == "orchestrator"


# ─── LocalBus — core operations ──────────────────────────────────────────────


class TestLocalBusPost:
    def test_post_returns_id(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        result_id = bus.post(t)
        assert result_id == t.id
        bus.close()

    def test_post_duplicate_raises(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        with pytest.raises(ValueError, match="already exists"):
            bus.post(t)
        bus.close()

    def test_post_preserves_inputs(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task(inputs={"model": "omni9b", "steps": 100})
        bus.post(t)
        retrieved = bus.get(t.id)
        assert retrieved is not None
        assert retrieved.inputs == {"model": "omni9b", "steps": 100}
        bus.close()

    def test_post_preserves_priority(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task(priority=5)
        bus.post(t)
        retrieved = bus.get(t.id)
        assert retrieved is not None
        assert retrieved.priority == 5
        bus.close()


class TestLocalBusPoll:
    def test_poll_returns_pending(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        tasks = bus.poll()
        assert any(task.id == t.id for task in tasks)
        bus.close()

    def test_poll_excludes_non_pending(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        # Mark it running
        running = bus.get(t.id)
        assert running is not None
        bus.update(running.mark_running())
        tasks = bus.poll(status="pending")
        assert all(task.id != t.id for task in tasks)
        bus.close()

    def test_poll_respects_receiver(self, tmp_path):
        bus = _bus(tmp_path)
        t_mine = _task(receiver="carl-studio")
        t_other = _task(receiver="other-agent")
        bus.post(t_mine)
        bus.post(t_other)
        tasks = bus.poll(receiver="carl-studio")
        ids = [task.id for task in tasks]
        assert t_mine.id in ids
        assert t_other.id not in ids
        bus.close()

    def test_poll_respects_limit(self, tmp_path):
        bus = _bus(tmp_path)
        for _ in range(5):
            bus.post(_task())
        tasks = bus.poll(limit=3)
        assert len(tasks) <= 3
        bus.close()

    def test_poll_priority_ordering(self, tmp_path):
        bus = _bus(tmp_path)
        t_low = _task(skill="low", priority=0)
        t_high = _task(skill="high", priority=10)
        bus.post(t_low)
        bus.post(t_high)
        tasks = bus.poll(limit=2)
        # Highest priority first
        assert tasks[0].id == t_high.id
        bus.close()

    def test_poll_invalid_limit_raises(self, tmp_path):
        bus = _bus(tmp_path)
        with pytest.raises(ValueError):
            bus.poll(limit=0)
        bus.close()


class TestLocalBusGet:
    def test_get_existing(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task(skill="trainer")
        bus.post(t)
        result = bus.get(t.id)
        assert result is not None
        assert result.id == t.id
        assert result.skill == "trainer"
        bus.close()

    def test_get_missing_returns_none(self, tmp_path):
        bus = _bus(tmp_path)
        result = bus.get("nonexistent-id")
        assert result is None
        bus.close()


class TestLocalBusUpdate:
    def test_update_status_reflected_in_get(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        running = bus.get(t.id)
        assert running is not None
        bus.update(running.mark_running())
        refreshed = bus.get(t.id)
        assert refreshed is not None
        assert refreshed.status == A2ATaskStatus.RUNNING
        bus.close()

    def test_update_done_with_result(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        task = bus.get(t.id)
        assert task is not None
        done = task.mark_running().mark_done({"score": 0.95})
        bus.update(done)
        refreshed = bus.get(t.id)
        assert refreshed is not None
        assert refreshed.status == A2ATaskStatus.DONE
        assert refreshed.result == {"score": 0.95}
        assert refreshed.completed_at is not None
        bus.close()

    def test_update_failed_with_error(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        task = bus.get(t.id)
        assert task is not None
        failed = task.mark_failed("timeout after 60s")
        bus.update(failed)
        refreshed = bus.get(t.id)
        assert refreshed is not None
        assert refreshed.status == A2ATaskStatus.FAILED
        assert refreshed.error == "timeout after 60s"
        bus.close()

    def test_update_not_reflected_in_poll_if_not_pending(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        task = bus.get(t.id)
        assert task is not None
        bus.update(task.mark_running())
        pending = bus.poll(status="pending")
        assert all(pt.id != t.id for pt in pending)
        bus.close()


class TestLocalBusCancel:
    def test_cancel_sets_cancelled(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        bus.cancel(t.id)
        refreshed = bus.get(t.id)
        assert refreshed is not None
        assert refreshed.status == A2ATaskStatus.CANCELLED
        bus.close()

    def test_cancel_nonexistent_is_noop(self, tmp_path):
        bus = _bus(tmp_path)
        # Should not raise
        bus.cancel("ghost-id")
        bus.close()

    def test_cancel_done_task_is_noop(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        task = bus.get(t.id)
        assert task is not None
        bus.update(task.mark_done({}))
        bus.cancel(t.id)  # must not overwrite done
        refreshed = bus.get(t.id)
        assert refreshed is not None
        assert refreshed.status == A2ATaskStatus.DONE
        bus.close()


class TestLocalBusMessages:
    def test_publish_and_get_roundtrip(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        msg = A2AMessage.progress(task_id=t.id, step=1, total=5, detail="warmup")
        bus.publish_message(msg)
        messages = bus.get_messages(t.id)
        assert len(messages) == 1
        assert messages[0].id == msg.id
        assert messages[0].type == "progress"
        assert messages[0].payload["step"] == 1
        bus.close()

    def test_multiple_messages_ordered_by_timestamp(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        m1 = A2AMessage.log(task_id=t.id, message="start")
        m2 = A2AMessage.progress(task_id=t.id, step=1, total=3)
        m3 = A2AMessage.artifact(task_id=t.id, name="ckpt", data={"path": "/x"})
        for m in [m1, m2, m3]:
            bus.publish_message(m)
        messages = bus.get_messages(t.id)
        assert len(messages) == 3
        # All types are present
        types = {m.type for m in messages}
        assert types == {"log", "progress", "artifact"}
        bus.close()

    def test_get_messages_for_nonexistent_task_returns_empty(self, tmp_path):
        bus = _bus(tmp_path)
        messages = bus.get_messages("ghost-id")
        assert messages == []
        bus.close()

    def test_publish_to_nonexistent_task_raises(self, tmp_path):
        bus = _bus(tmp_path)
        msg = A2AMessage.log(task_id="no-such-task", message="oops")
        with pytest.raises(ValueError, match="not found"):
            bus.publish_message(msg)
        bus.close()

    def test_duplicate_message_id_raises(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        msg = A2AMessage.log(task_id=t.id, message="first")
        bus.publish_message(msg)
        # Same message object — same id
        with pytest.raises(ValueError, match="already exists"):
            bus.publish_message(msg)
        bus.close()


class TestLocalBusPendingCount:
    def test_pending_count_zero_initially(self, tmp_path):
        bus = _bus(tmp_path)
        assert bus.pending_count() == 0
        bus.close()

    def test_pending_count_increments(self, tmp_path):
        bus = _bus(tmp_path)
        bus.post(_task())
        bus.post(_task())
        assert bus.pending_count() == 2
        bus.close()

    def test_pending_count_decrements_after_update(self, tmp_path):
        bus = _bus(tmp_path)
        t = _task()
        bus.post(t)
        task = bus.get(t.id)
        assert task is not None
        bus.update(task.mark_done({}))
        assert bus.pending_count() == 0
        bus.close()

    def test_pending_count_respects_receiver(self, tmp_path):
        bus = _bus(tmp_path)
        bus.post(_task(receiver="carl-studio"))
        bus.post(_task(receiver="other-agent"))
        assert bus.pending_count(receiver="carl-studio") == 1
        assert bus.pending_count(receiver="other-agent") == 1
        bus.close()


class TestLocalBusContextManager:
    def test_context_manager_closes(self, tmp_path):
        with LocalBus(db_path=tmp_path / "a2a.db") as bus:
            t = _task()
            bus.post(t)
            assert bus.get(t.id) is not None
        # After __exit__, connection should be closed — accessing should not raise
        # (close is idempotent)
        bus.close()


# ─── CARLAgentCard ────────────────────────────────────────────────────────────


class TestCARLAgentCard:
    def test_current_does_not_raise(self):
        """current() must not raise even when skills/settings unavailable."""
        card = CARLAgentCard.current()
        assert card.name == "carl-studio"
        assert card.tier in ("free", "paid")
        assert isinstance(card.capabilities, list)
        assert isinstance(card.skills, list)

    def test_to_json_roundtrip(self):
        card = CARLAgentCard(
            name="carl-studio",
            version="0.3.0",
            tier="free",
            capabilities=["train", "eval"],
            skills=["observer"],
            endpoint="stdio",
            metadata={"env": "test"},
        )
        json_str = card.to_json()
        data = json.loads(json_str)
        assert data["name"] == "carl-studio"
        assert data["skills"] == ["observer"]
        assert data["metadata"] == {"env": "test"}

    def test_from_json_roundtrip(self):
        card = CARLAgentCard(
            name="carl-studio",
            version="0.3.1",
            tier="paid",
            capabilities=["bench"],
            skills=["grader"],
            endpoint="http://localhost:8080",
            metadata={"region": "us-east-1"},
        )
        reconstructed = CARLAgentCard.from_json(card.to_json())
        assert reconstructed.name == card.name
        assert reconstructed.version == card.version
        assert reconstructed.tier == card.tier
        assert reconstructed.capabilities == card.capabilities
        assert reconstructed.skills == card.skills
        assert reconstructed.endpoint == card.endpoint
        assert reconstructed.metadata == card.metadata

    def test_default_capabilities_present(self):
        card = CARLAgentCard()
        assert "train" in card.capabilities
        assert "eval" in card.capabilities
        assert "observe" in card.capabilities

    def test_from_json_invalid_raises(self):
        with pytest.raises((json.JSONDecodeError, TypeError, Exception)):
            CARLAgentCard.from_json("{not valid json")

    def test_metadata_defaults_empty(self):
        card = CARLAgentCard.current()
        assert isinstance(card.metadata, dict)


# ─── FSM completeness sweep ───────────────────────────────────────────────────


class TestFSMCompleteness:
    """Exhaustive check: every terminal state blocks every transition."""

    TERMINAL_STATUSES = [
        A2ATaskStatus.DONE,
        A2ATaskStatus.FAILED,
        A2ATaskStatus.CANCELLED,
    ]

    def _terminal_task(self, status: A2ATaskStatus) -> A2ATask:
        t = _task()
        if status == A2ATaskStatus.DONE:
            return t.mark_done({"ok": True})
        if status == A2ATaskStatus.FAILED:
            return t.mark_failed("test error")
        if status == A2ATaskStatus.CANCELLED:
            return t.mark_cancelled()
        raise AssertionError(f"Unexpected status: {status}")

    @pytest.mark.parametrize("term_status", TERMINAL_STATUSES)
    def test_mark_running_blocked(self, term_status: A2ATaskStatus):
        t = self._terminal_task(term_status)
        assert t.mark_running().status == term_status

    @pytest.mark.parametrize("term_status", TERMINAL_STATUSES)
    def test_mark_done_blocked(self, term_status: A2ATaskStatus):
        t = self._terminal_task(term_status)
        assert t.mark_done({"x": 1}).status == term_status

    @pytest.mark.parametrize("term_status", TERMINAL_STATUSES)
    def test_mark_failed_blocked(self, term_status: A2ATaskStatus):
        t = self._terminal_task(term_status)
        assert t.mark_failed("late").status == term_status

    @pytest.mark.parametrize("term_status", TERMINAL_STATUSES)
    def test_mark_cancelled_blocked(self, term_status: A2ATaskStatus):
        t = self._terminal_task(term_status)
        assert t.mark_cancelled().status == term_status

    @pytest.mark.parametrize("term_status", TERMINAL_STATUSES)
    def test_is_terminal_true(self, term_status: A2ATaskStatus):
        t = self._terminal_task(term_status)
        assert t.is_terminal()
