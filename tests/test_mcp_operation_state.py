"""Generic operation generations stay at the existing task owner."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from carl_core.errors import CARLError

from carl_studio.mcp.tasks import MCPTaskStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("carl_studio.mcp.tasks._a2a_bus_path", lambda: tmp_path / "absent.db")
    result = MCPTaskStore(tmp_path / "operations.db")
    yield result
    result.close()


def test_constructive_cycle_callback_and_same_commit_replay(store):
    callbacks = []
    assert store.load_operation("cycle") is None
    token = store.claim_operation("cycle", 0, {"phase": "claimed"}, "action:single")
    callbacks.append("single")
    completed = store.commit_operation(token, {"phase": "measured", "accepted_modes": []})
    assert completed == store.load_operation("cycle")
    assert completed["generation"] == 1
    assert store.commit_operation(token, {"phase": "measured", "accepted_modes": []}) == completed
    assert callbacks == ["single"]
    token2 = store.claim_operation("cycle", 1, completed, "action:tcg")
    callbacks.append("tcg")
    store.commit_operation(token2, {"phase": "rejected", "accepted_modes": []})
    assert store.load_operation("cycle")["generation"] == 2
    assert store.load_operation("cycle")["accepted_modes"] == []


def test_stale_generation_inflight_and_changed_commit_are_refused(store):
    token = store.claim_operation("cycle", 0, {"phase": "claimed"}, "action:one")
    for generation in [0, 1]:
        with pytest.raises(CARLError):
            store.claim_operation("cycle", generation, {}, "action:again")
    store.commit_operation(token, {"phase": "retained"})
    with pytest.raises(CARLError):
        store.commit_operation(token, {"phase": "changed"})
    with pytest.raises(CARLError):
        store.claim_operation("cycle", 0, {}, "action:stale")
    token2 = store.claim_operation("cycle", 1, {}, "action:next")
    with pytest.raises(CARLError):
        store.commit_operation(token, {"phase": "retained"})
    store.commit_operation(token2, {"phase": "current"})


def test_independent_connections_allow_one_claim_and_restart_retains_it(store):
    barrier = threading.Barrier(4)

    def claim(_):
        independent = MCPTaskStore(store.path)
        try:
            barrier.wait(timeout=5)
            try:
                return independent.claim_operation("cycle", 0, {}, "action:one")
            except CARLError:
                return None
        finally:
            independent.close()

    with ThreadPoolExecutor(max_workers=4) as workers:
        tokens = list(workers.map(claim, range(4)))
    assert sum(token is not None for token in tokens) == 1
    restarted = MCPTaskStore(store.path)
    try:
        assert restarted.load_operation("cycle")["generation"] == 1
        with pytest.raises(CARLError):
            restarted.claim_operation("cycle", 1, {}, "action:duplicate")
        restarted.commit_operation(next(token for token in tokens if token), {"phase": "retained"})
        assert restarted.load_operation("cycle")["phase"] == "retained"
    finally:
        restarted.close()


def test_non_json_record_wrong_identity_and_unowned_token_are_refused(store):
    with pytest.raises(ValueError):
        store.claim_operation("cycle", 0, {"operation_id": "other"}, "action:one")
    with pytest.raises(ValueError):
        store.claim_operation("cycle", 0, {"value": float("nan")}, "action:one")
    token = store.claim_operation("cycle", 0, {}, "action:one")
    with pytest.raises(CARLError):
        store.commit_operation(token + "changed", {})
    store.commit_operation(token, {"phase": "retained"})


def test_dead_owner_reopens_for_reconciliation_without_resubmitting(store):
    child = os.fork()
    if child == 0:
        independent = MCPTaskStore(store.path)
        independent.claim_operation("cycle", 0, {"phase": "submitted"}, "action:submit")
        independent.close()
        os._exit(0)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    restarted = MCPTaskStore(store.path)
    try:
        with pytest.raises(CARLError):
            restarted.claim_operation("cycle", 1, {}, "action:resubmit")
        token = restarted.reconcile_operation("cycle", 1, {"phase": "observing"}, "action:read")
        task = restarted.get(token.split(".", 1)[0])
        assert task.metadata["inflight"]["mode"] == "reconcile-only"
        callbacks = ["read"]
        restarted.commit_operation(token, {"phase": "retained", "callbacks": callbacks})
        assert restarted.load_operation("cycle")["callbacks"] == ["read"]
        assert restarted.load_operation("cycle")["generation"] == 2
    finally:
        restarted.close()


def test_live_owner_refuses_parallel_recovery_but_allows_exact_token_handoff(store):
    original = store.claim_operation("cycle", 0, {}, "action:submit")
    with pytest.raises(CARLError):
        store.reconcile_operation("cycle", 1, {}, "action:read")
    with pytest.raises(CARLError):
        store.reconcile_operation("cycle", 1, {}, "action:read", previous_token="wrong")
    recovered = store.reconcile_operation("cycle", 1, {}, "action:read", previous_token=original)
    with pytest.raises(CARLError):
        store.commit_operation(original, {})
    store.commit_operation(recovered, {"phase": "observed"})
