"""Tests for carl_core.interaction_store — JSONL append-only chain persistence."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from carl_core.interaction import ActionType, InteractionChain, Step
from carl_core.interaction_store import InteractionStore


def _make_step(i: int, *, action: ActionType = ActionType.TOOL_CALL) -> Step:
    return Step(
        action=action,
        name=f"step-{i}",
        input={"i": i},
        output={"ok": True},
    )


class TestBasic:
    def test_root_auto_created(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path / "nested" / "interactions")
        chain = InteractionChain()
        store.append(chain.chain_id, _make_step(0))
        assert (tmp_path / "nested" / "interactions").is_dir()

    def test_append_then_load_roundtrip(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        chain = InteractionChain()
        chain.context["source"] = "test"

        # Append a few steps
        step_0 = _make_step(0)
        step_1 = _make_step(1, action=ActionType.LLM_REPLY)
        step_2 = _make_step(2, action=ActionType.CHECKPOINT)

        store.append(
            chain.chain_id,
            step_0,
            header={
                "chain_id": chain.chain_id,
                "started_at": chain.started_at.isoformat(),
                "context": chain.context,
            },
        )
        store.append(chain.chain_id, step_1)
        store.append(chain.chain_id, step_2)

        restored = store.load(chain.chain_id)
        assert restored.chain_id == chain.chain_id
        assert len(restored.steps) == 3
        assert [s.name for s in restored.steps] == ["step-0", "step-1", "step-2"]
        assert restored.steps[1].action == ActionType.LLM_REPLY
        assert restored.steps[2].action == ActionType.CHECKPOINT
        assert restored.context == {"source": "test"}

    def test_missing_file_returns_empty_chain(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        chain = store.load("nonexistent")
        assert isinstance(chain, InteractionChain)
        assert chain.chain_id == "nonexistent"
        assert len(chain) == 0

    def test_list_chains(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        store.append("chain-a", _make_step(0))
        store.append("chain-b", _make_step(1))
        ids = store.list_chains()
        assert sorted(ids) == ["chain-a", "chain-b"]

    def test_list_chains_empty_when_root_missing(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path / "never-exists")
        assert store.list_chains() == []

    def test_append_chain_snapshot(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        chain = InteractionChain()
        chain.record(ActionType.TRAINING_STEP, "train.start")
        chain.record(ActionType.TRAINING_STEP, "train.step", output={"loss": 0.5})
        chain.record(ActionType.CHECKPOINT, "train.save")

        path = store.append_chain(chain)
        assert path.exists()

        restored = store.load(chain.chain_id)
        assert len(restored.steps) == 3
        assert restored.steps[-1].action == ActionType.CHECKPOINT

    def test_delete(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        store.append("doomed", _make_step(0))
        assert store.delete("doomed") is True
        assert store.delete("doomed") is False

    def test_rejects_traversal_chain_id(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        with pytest.raises(ValueError):
            store.append("../escape", _make_step(0))
        with pytest.raises(ValueError):
            store.load("../escape")
        with pytest.raises(ValueError):
            store.load("")


class TestConcurrency:
    def test_concurrent_append_safety(self, tmp_path: Path) -> None:
        """Two threads writing 100 steps each must not corrupt JSONL."""
        store = InteractionStore(tmp_path)
        chain_id = "concurrent"
        step_count = 100

        errors: list[BaseException] = []

        def worker(prefix: str) -> None:
            try:
                for i in range(step_count):
                    step = Step(
                        action=ActionType.EXTERNAL,
                        name=f"{prefix}-{i}",
                        input={"i": i},
                    )
                    store.append(chain_id, step)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors

        path = store.path_for(chain_id)
        text = path.read_text().splitlines()
        # header + 200 step rows
        assert len(text) == 1 + 2 * step_count

        # Every line must parse as JSON
        for line in text:
            json.loads(line)

        chain = store.load(chain_id)
        assert len(chain.steps) == 2 * step_count
        names = {s.name for s in chain.steps}
        assert len(names) == 2 * step_count


class TestMetadata:
    def test_session_and_trace_id_persist(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        step = Step(
            action=ActionType.TRAINING_STEP,
            name="train.step",
            session_id="sess-1",
            trace_id="trace-42",
        )
        store.append("cx", step)
        chain = store.load("cx")
        assert chain.steps[0].session_id == "sess-1"
        assert chain.steps[0].trace_id == "trace-42"

    def test_corrupt_line_does_not_break_load(self, tmp_path: Path) -> None:
        store = InteractionStore(tmp_path)
        store.append("cid", _make_step(0))
        path = store.path_for("cid")
        with open(path, "a") as fh:
            fh.write("{ not valid json\n")
        store.append("cid", _make_step(1))

        chain = store.load("cid")
        # One corrupt line should be skipped, two valid steps recovered.
        assert len(chain.steps) == 2
