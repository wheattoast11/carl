"""Tests for carl_core.optimizer_state — durable Adam (m, v)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from carl_core.errors import ValidationError
from carl_core.optimizer_state import (
    AdamMoments,
    OptimizerStateStore,
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> OptimizerStateStore:
    return OptimizerStateStore(tmp_path / "opt")


def _moments(n: int = 4, *, step: int = 5) -> AdamMoments:
    rng = np.random.default_rng(seed=7)
    now = time.time()
    return AdamMoments(
        m=rng.standard_normal(n).astype(np.float64),
        v=np.abs(rng.standard_normal(n)).astype(np.float64),
        step=step,
        beta1=0.9,
        beta2=0.999,
        created_at=now,
        last_updated_at=now,
    )


# ---------------------------------------------------------------------------
# 1. Round-trip: save -> load preserves every field byte-identically
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_all_fields(store: OptimizerStateStore) -> None:
    original = _moments()
    path = store.save("s1", "mod_a", original)
    assert path.exists()
    recovered = store.load("s1", "mod_a")
    assert recovered is not None
    np.testing.assert_array_equal(recovered.m, original.m)
    np.testing.assert_array_equal(recovered.v, original.v)
    assert recovered.step == original.step
    assert recovered.beta1 == original.beta1
    assert recovered.beta2 == original.beta2
    assert recovered.created_at == original.created_at
    assert recovered.last_updated_at == original.last_updated_at
    assert recovered.hash() == original.hash()


def test_load_missing_returns_none(store: OptimizerStateStore) -> None:
    assert store.load("nope", "also_nope") is None


# ---------------------------------------------------------------------------
# 2. Corruption: tampered .npz -> load returns None + quarantine file appears
# ---------------------------------------------------------------------------


def test_corrupt_file_is_quarantined(store: OptimizerStateStore) -> None:
    original = _moments()
    path = store.save("s2", "mod_b", original)
    assert path.exists()
    # Corrupt the file — overwrite with garbage.
    path.write_bytes(b"\x00not a valid npz\x00")
    result = store.load("s2", "mod_b")
    assert result is None
    # Quarantine file should be present.
    siblings = list(path.parent.iterdir())
    names = [p.name for p in siblings]
    assert any(n.startswith(path.name + ".quarantine_") for n in names)
    # Original path no longer exists.
    assert not path.exists()


def test_load_after_quarantine_can_succeed_on_second_save(
    store: OptimizerStateStore,
) -> None:
    original = _moments()
    path = store.save("s3", "mod_c", original)
    path.write_bytes(b"garbage")
    assert store.load("s3", "mod_c") is None
    # Re-save cleanly, load should succeed.
    store.save("s3", "mod_c", original)
    recovered = store.load("s3", "mod_c")
    assert recovered is not None
    assert recovered.hash() == original.hash()


# ---------------------------------------------------------------------------
# 3. Atomic save — partial writes must not be visible
# ---------------------------------------------------------------------------


def test_atomic_save_leaves_no_partial_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = OptimizerStateStore(tmp_path / "opt")
    # Save a good prior version.
    prior = _moments()
    store.save("s4", "mod_d", prior)
    prior_bytes = store._file_path("s4", "mod_d").read_bytes()  # type: ignore[attr-defined]

    # Force np.savez to raise AFTER the tempfile is created but BEFORE rename.
    real_savez = np.savez

    def exploding_savez(*args: object, **kwargs: object) -> None:
        real_savez(*args, **kwargs)  # write the temp file
        raise RuntimeError("simulated crash after tempfile write")

    monkeypatch.setattr(np, "savez", exploding_savez)
    new_moments = _moments(step=99)
    with pytest.raises(RuntimeError):
        store.save("s4", "mod_d", new_moments)

    # Prior file must remain intact.
    post_bytes = store._file_path("s4", "mod_d").read_bytes()  # type: ignore[attr-defined]
    assert post_bytes == prior_bytes
    # No leftover temp file.
    leftovers = [
        p for p in store._file_path("s4", "mod_d").parent.iterdir()  # type: ignore[attr-defined]
        if p.name.endswith(".tmp")
    ]
    assert leftovers == []


# ---------------------------------------------------------------------------
# 4. list_sessions returns sorted list
# ---------------------------------------------------------------------------


def test_list_sessions_sorted(store: OptimizerStateStore) -> None:
    names = ["beta", "alpha", "zulu", "mike"]
    for n in names:
        store.save(n, "m", _moments())
    assert store.list_sessions() == sorted(names)


def test_list_sessions_empty(tmp_path: Path) -> None:
    empty = OptimizerStateStore(tmp_path / "empty")
    assert empty.list_sessions() == []


def test_list_modules_returns_sorted(store: OptimizerStateStore) -> None:
    for m in ["zed", "apple", "middle"]:
        store.save("s5", m, _moments())
    assert store.list_modules("s5") == ["apple", "middle", "zed"]


def test_list_modules_missing_session(store: OptimizerStateStore) -> None:
    assert store.list_modules("phantom") == []


# ---------------------------------------------------------------------------
# 5. delete_session removes all session files
# ---------------------------------------------------------------------------


def test_delete_session_removes_files(store: OptimizerStateStore) -> None:
    for m in ["a", "b", "c"]:
        store.save("to_delete", m, _moments())
    assert store.list_modules("to_delete") == ["a", "b", "c"]
    removed = store.delete_session("to_delete")
    assert removed == 3
    assert store.list_modules("to_delete") == []
    assert "to_delete" not in store.list_sessions()


def test_delete_missing_session_is_zero(store: OptimizerStateStore) -> None:
    assert store.delete_session("never_was") == 0


# ---------------------------------------------------------------------------
# 6. Concurrent save of same slot does not corrupt
# ---------------------------------------------------------------------------


def test_concurrent_saves_are_safe(store: OptimizerStateStore) -> None:
    # Multiple threads save different versions of the same slot; at the end
    # the file must be readable and its contents must match one of the
    # versions we attempted to save.
    versions = [_moments(step=i + 1) for i in range(10)]
    errors: list[BaseException] = []

    def worker(v: AdamMoments) -> None:
        try:
            store.save("s6", "shared", v)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(v,)) for v in versions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    loaded = store.load("s6", "shared")
    assert loaded is not None
    assert loaded.hash() in {v.hash() for v in versions}


# ---------------------------------------------------------------------------
# 7. id validation rejects path escapes and junk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", ".", "..", "a/b", "c\\d", "with space", "semicolon;x"],
)
def test_bad_ids_rejected(store: OptimizerStateStore, bad: str) -> None:
    with pytest.raises(ValidationError):
        store.save(bad, "ok", _moments())
    with pytest.raises(ValidationError):
        store.save("ok_session", bad, _moments())


# ---------------------------------------------------------------------------
# 8. AdamMoments validation
# ---------------------------------------------------------------------------


def test_adam_moments_shape_mismatch_raises() -> None:
    now = time.time()
    with pytest.raises(ValidationError):
        AdamMoments(
            m=np.zeros(3),
            v=np.zeros(4),
            step=0,
            beta1=0.9,
            beta2=0.999,
            created_at=now,
            last_updated_at=now,
        )


def test_adam_moments_bad_betas_raise() -> None:
    now = time.time()
    with pytest.raises(ValidationError):
        AdamMoments(
            m=np.zeros(2),
            v=np.zeros(2),
            step=0,
            beta1=1.2,
            beta2=0.999,
            created_at=now,
            last_updated_at=now,
        )


def test_adam_moments_negative_step_raises() -> None:
    now = time.time()
    with pytest.raises(ValidationError):
        AdamMoments(
            m=np.zeros(2),
            v=np.zeros(2),
            step=-1,
            beta1=0.9,
            beta2=0.999,
            created_at=now,
            last_updated_at=now,
        )


# ---------------------------------------------------------------------------
# 9. save_arrays convenience preserves created_at across updates
# ---------------------------------------------------------------------------


def test_save_arrays_preserves_created_at(store: OptimizerStateStore) -> None:
    m = np.arange(4, dtype=np.float64)
    v = np.arange(4, dtype=np.float64) + 0.5
    store.save_arrays("s7", "mod", m, v, step=1)
    first = store.load("s7", "mod")
    assert first is not None
    time.sleep(0.01)
    store.save_arrays("s7", "mod", m * 2, v * 2, step=2)
    second = store.load("s7", "mod")
    assert second is not None
    assert second.created_at == first.created_at
    assert second.last_updated_at >= first.last_updated_at
    assert second.step == 2


def test_has_existence_check(store: OptimizerStateStore) -> None:
    assert not store.has("s8", "m")
    store.save("s8", "m", _moments())
    assert store.has("s8", "m")


def test_summary(store: OptimizerStateStore) -> None:
    store.save("s9a", "m1", _moments())
    store.save("s9a", "m2", _moments())
    store.save("s9b", "m1", _moments())
    summary = store.summary()
    assert summary["session_count"] == 2
    assert summary["module_count"] == 3
    assert set(summary["sessions"]) == {"s9a", "s9b"}
