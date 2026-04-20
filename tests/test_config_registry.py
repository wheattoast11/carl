"""Tests for :class:`carl_studio.config_registry.ConfigRegistry`.

Exercises round-trip / default / clear / error paths plus the
namespace-derivation and LocalDB convenience factory. Uses a
per-test SQLite file (via the ``tmp_path`` fixture) so SQLite locking
behaviour is real rather than mocked.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from carl_core.errors import CARLError

from carl_studio.config_registry import ConfigRegistry
from carl_studio.db import LocalDB


# ---------------------------------------------------------------------------
# Fixtures + test models
# ---------------------------------------------------------------------------


class _DemoState(BaseModel):
    count: int = 0
    label: str = ""


class _DifferentShape(BaseModel):
    # Incompatible with _DemoState — forces a schema mismatch when read.
    required_int: int
    extra: str = Field(default="x")


@pytest.fixture
def tmp_db(tmp_path: Path) -> LocalDB:
    return LocalDB(db_path=tmp_path / "carl.db")


# ---------------------------------------------------------------------------
# Construction + key derivation
# ---------------------------------------------------------------------------


def test_default_key_is_namespace_plus_lowercased_model_name(
    tmp_db: LocalDB,
) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.example")
    assert reg.key == "carl.example._demostate"
    assert reg.namespace == "carl.example"
    assert reg.model_cls is _DemoState


def test_explicit_key_override_wins(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(
        _DemoState,
        db=tmp_db,
        namespace="carl.example",
        key="custom.key.name",
    )
    assert reg.key == "custom.key.name"


def test_empty_namespace_rejected(tmp_db: LocalDB) -> None:
    with pytest.raises(ValueError):
        ConfigRegistry(_DemoState, db=tmp_db, namespace="")


def test_empty_explicit_key_rejected(tmp_db: LocalDB) -> None:
    with pytest.raises(ValueError):
        ConfigRegistry(
            _DemoState, db=tmp_db, namespace="carl.example", key=""
        )


# ---------------------------------------------------------------------------
# Round-trip, missing, and clear
# ---------------------------------------------------------------------------


def test_get_returns_none_when_absent(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.test")
    assert reg.get() is None


def test_round_trip_get_set(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.test")
    reg.set(_DemoState(count=3, label="hello"))
    stored = reg.get()
    assert stored is not None
    assert stored.count == 3
    assert stored.label == "hello"


def test_set_rejects_wrong_type(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.test")
    with pytest.raises(TypeError):
        reg.set("not a model")  # type: ignore[arg-type]


def test_clear_removes_stored_value(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.test")
    reg.set(_DemoState(count=1))
    assert reg.get() is not None
    reg.clear()
    assert reg.get() is None


def test_clear_is_noop_when_absent(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.test")
    # No prior set — must not raise.
    reg.clear()
    assert reg.get() is None


# ---------------------------------------------------------------------------
# Schema mismatch
# ---------------------------------------------------------------------------


def test_schema_mismatch_raises_carl_error(tmp_db: LocalDB) -> None:
    # Write a value under one schema...
    wrong_reg = ConfigRegistry(
        _DemoState, db=tmp_db, namespace="carl.test", key="shared.key"
    )
    wrong_reg.set(_DemoState(count=1))

    # ...then read under a different, incompatible schema.
    other_reg = ConfigRegistry(
        _DifferentShape,
        db=tmp_db,
        namespace="carl.test",
        key="shared.key",
    )
    with pytest.raises(CARLError) as exc_info:
        other_reg.get()
    assert exc_info.value.code == "carl.config.schema_mismatch"
    ctx = exc_info.value.context
    assert ctx["key"] == "shared.key"
    assert ctx["model"] == "_DifferentShape"


# ---------------------------------------------------------------------------
# with_defaults
# ---------------------------------------------------------------------------


def test_with_defaults_returns_stored_when_present(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.test")
    reg.set(_DemoState(count=42, label="persisted"))
    value = reg.with_defaults(lambda: _DemoState(count=-1, label="fallback"))
    assert value.count == 42
    assert value.label == "persisted"


def test_with_defaults_returns_factory_when_absent(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.test")
    value = reg.with_defaults(lambda: _DemoState(count=9, label="new"))
    assert value.count == 9
    assert value.label == "new"
    # with_defaults is side-effect free — nothing persisted.
    assert reg.get() is None


def test_with_defaults_factory_wrong_type_raises(tmp_db: LocalDB) -> None:
    reg = ConfigRegistry(_DemoState, db=tmp_db, namespace="carl.test")
    with pytest.raises(TypeError):
        reg.with_defaults(lambda: "not a model")  # type: ignore[arg-type,return-value]


# ---------------------------------------------------------------------------
# LocalDB.config_registry factory
# ---------------------------------------------------------------------------


def test_localdb_factory_yields_equivalent_registry(tmp_db: LocalDB) -> None:
    reg_via_factory = tmp_db.config_registry(
        _DemoState, namespace="carl.factory"
    )
    reg_via_factory.set(_DemoState(count=7, label="factory"))

    reg_direct = ConfigRegistry(
        _DemoState, db=tmp_db, namespace="carl.factory"
    )
    stored = reg_direct.get()
    assert stored is not None
    assert stored.count == 7
    assert stored.label == "factory"


# ---------------------------------------------------------------------------
# Concurrent reads — SQLite locking sanity
# ---------------------------------------------------------------------------


def test_concurrent_reads_are_safe(tmp_db: LocalDB) -> None:
    reg = tmp_db.config_registry(_DemoState, namespace="carl.concurrent")
    reg.set(_DemoState(count=5, label="shared"))

    results: list[_DemoState | None] = []
    errors: list[BaseException] = []

    def _reader() -> None:
        try:
            for _ in range(20):
                results.append(reg.get())
        except BaseException as exc:  # pragma: no cover — defensive
            errors.append(exc)

    threads = [threading.Thread(target=_reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Every read must see the persisted value.
    assert len(results) == 4 * 20
    for r in results:
        assert r is not None
        assert r.count == 5
        assert r.label == "shared"
