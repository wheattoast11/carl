"""Tests for :mod:`carl_studio.envutil` — the env-var parsing helpers.

Collapses three near-identical scaffolds in ``heartbeat/connection.py`` and
``heartbeat/loop.py`` into a single point of truth (R2-002). These tests
pin the contract so future callers know exactly what ``arg → env →
default`` does on malformed, missing, explicit, and minimum-clamped
inputs.

The smoke regression at the bottom (:func:`test_reclaim_default_matches_doctor_threshold`)
guards the R2-005 "magic number drift" fix — ``carl doctor``'s stuck-row
heuristic and ``StickyQueue.reclaim_stale`` must read the same constant
so the doctor never flags work the daemon already reclaimed.
"""

from __future__ import annotations

import logging

import pytest

from carl_studio.envutil import env_bool, env_float, env_int


# ---------------------------------------------------------------------------
# env_int
# ---------------------------------------------------------------------------


def test_env_int_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid int env-var value overrides the default."""
    monkeypatch.setenv("CARL_TEST_INT", "42")
    assert env_int("CARL_TEST_INT", default=7) == 42


def test_env_int_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing env var falls back to ``default``."""
    monkeypatch.delenv("CARL_TEST_INT", raising=False)
    assert env_int("CARL_TEST_INT", default=7) == 7


def test_env_int_returns_default_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty/whitespace-only env var is treated as unset."""
    monkeypatch.setenv("CARL_TEST_INT", "   ")
    assert env_int("CARL_TEST_INT", default=11) == 11


def test_env_int_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``explicit`` takes precedence over both env and default."""
    monkeypatch.setenv("CARL_TEST_INT", "42")
    assert env_int("CARL_TEST_INT", default=7, explicit=99) == 99


def test_env_int_explicit_coerces_to_int(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``bool`` or ``float``-shaped explicit override still lands as int."""
    monkeypatch.delenv("CARL_TEST_INT", raising=False)
    assert env_int("CARL_TEST_INT", default=0, explicit=True) == 1


def test_env_int_invalid_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mistyped env value logs a warning and returns the default — no raise."""
    monkeypatch.setenv("CARL_TEST_INT", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="carl.envutil"):
        assert env_int("CARL_TEST_INT", default=123) == 123
    assert any(
        "invalid CARL_TEST_INT" in rec.getMessage() for rec in caplog.records
    )


def test_env_int_minimum_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values below ``minimum`` are clamped upward to ``minimum``."""
    monkeypatch.setenv("CARL_TEST_INT", "-5")
    assert env_int("CARL_TEST_INT", default=0, minimum=0) == 0


def test_env_int_minimum_clamps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``minimum`` clamps even when the default itself falls below the floor."""
    monkeypatch.delenv("CARL_TEST_INT", raising=False)
    assert env_int("CARL_TEST_INT", default=-1, minimum=0) == 0


def test_env_int_minimum_allows_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value *equal* to ``minimum`` is kept (strict less-than clamp)."""
    monkeypatch.setenv("CARL_TEST_INT", "0")
    assert env_int("CARL_TEST_INT", default=5, minimum=0) == 0


# ---------------------------------------------------------------------------
# env_float
# ---------------------------------------------------------------------------


def test_env_float_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARL_TEST_FLOAT", "3.14")
    assert env_float("CARL_TEST_FLOAT", default=0.0) == 3.14


def test_env_float_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARL_TEST_FLOAT", raising=False)
    assert env_float("CARL_TEST_FLOAT", default=2.5) == 2.5


def test_env_float_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARL_TEST_FLOAT", "3.14")
    assert env_float("CARL_TEST_FLOAT", default=0.0, explicit=7.5) == 7.5


def test_env_float_invalid_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CARL_TEST_FLOAT", "wobble")
    with caplog.at_level(logging.WARNING, logger="carl.envutil"):
        assert env_float("CARL_TEST_FLOAT", default=1.5) == 1.5
    assert any(
        "invalid CARL_TEST_FLOAT" in rec.getMessage() for rec in caplog.records
    )


def test_env_float_minimum_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARL_TEST_FLOAT", "-1.5")
    assert env_float("CARL_TEST_FLOAT", default=0.0, minimum=0.0) == 0.0


def test_env_float_accepts_int_shaped_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integer literals round-trip through ``float(...)`` — ``45`` → ``45.0``."""
    monkeypatch.setenv("CARL_TEST_FLOAT", "45")
    assert env_float("CARL_TEST_FLOAT", default=0.0) == 45.0


# ---------------------------------------------------------------------------
# env_bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "YES", "on", "ON"])
def test_env_bool_truthy(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Truthy aliases all resolve to ``True`` regardless of case."""
    monkeypatch.setenv("CARL_TEST_BOOL", raw)
    assert env_bool("CARL_TEST_BOOL") is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "NO", "off", "OFF"])
def test_env_bool_falsy(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Falsy aliases all resolve to ``False`` regardless of case."""
    monkeypatch.setenv("CARL_TEST_BOOL", raw)
    assert env_bool("CARL_TEST_BOOL", default=True) is False


def test_env_bool_unknown_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecognized values fall back to ``default`` without raising."""
    monkeypatch.setenv("CARL_TEST_BOOL", "maybe")
    assert env_bool("CARL_TEST_BOOL", default=True) is True
    assert env_bool("CARL_TEST_BOOL", default=False) is False


def test_env_bool_unset_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARL_TEST_BOOL", raising=False)
    assert env_bool("CARL_TEST_BOOL", default=True) is True
    assert env_bool("CARL_TEST_BOOL") is False  # default=False by default


def test_env_bool_whitespace_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only env value falls back to ``default`` (empty after strip)."""
    monkeypatch.setenv("CARL_TEST_BOOL", "   ")
    assert env_bool("CARL_TEST_BOOL", default=True) is True


# ---------------------------------------------------------------------------
# Smoke regression — R2-005 guard
# ---------------------------------------------------------------------------


def test_reclaim_default_matches_doctor_threshold() -> None:
    """``carl doctor`` stuck-row threshold and ``StickyQueue.reclaim_stale``
    default must be the same literal.

    Drift between these two used to produce false ``stuck`` flags —
    ``carl doctor`` would flag a row the daemon had already reclaimed
    because the two sites had hand-rolled different magic numbers. The
    fix is that both read :data:`carl_studio.sticky.DEFAULT_RECLAIM_MAX_AGE_S`.
    This test fails loudly the moment that invariant breaks.
    """
    from carl_studio.sticky import DEFAULT_RECLAIM_MAX_AGE_S, StickyQueue

    # Default of the CLI/public API is the constant.
    import inspect

    sig = inspect.signature(StickyQueue.reclaim_stale)
    default_max_age = sig.parameters["max_age_seconds"].default
    assert default_max_age == DEFAULT_RECLAIM_MAX_AGE_S

    # ``carl doctor`` pulls the same constant — re-import from the CLI
    # surface and confirm the alias resolves to the same value. We access
    # the attribute dynamically so pyright's ``reportPrivateImportUsage``
    # does not flag the cross-module reach; the test only cares that the
    # doctor's module-level binding is the shared constant.
    from carl_studio.cli import startup as startup_mod

    doctor_constant = getattr(startup_mod, "DEFAULT_RECLAIM_MAX_AGE_S")
    assert doctor_constant == DEFAULT_RECLAIM_MAX_AGE_S


def test_retention_default_matches_across_surfaces() -> None:
    """``LocalDB.maintenance``, the CLI default, and ``StickyQueue.maintenance``
    all anchor to the same :data:`DEFAULT_RETENTION_DAYS` literal.
    """
    import inspect

    from carl_studio.cli import db as db_cli
    from carl_studio.db import DEFAULT_RETENTION_DAYS, LocalDB
    from carl_studio.sticky import DEFAULT_RETENTION_DAYS as STICKY_ALIAS
    from carl_studio.sticky import StickyQueue

    # The alias re-exported from ``sticky`` must still resolve to the
    # owning module's value.
    assert STICKY_ALIAS == DEFAULT_RETENTION_DAYS

    local_db_default = (
        inspect.signature(LocalDB.maintenance).parameters["retention_days"].default
    )
    sticky_default = (
        inspect.signature(StickyQueue.maintenance).parameters["retention_days"].default
    )
    cli_default = (
        inspect.signature(db_cli.db_maintenance).parameters["retention_days"].default
    )

    assert local_db_default == DEFAULT_RETENTION_DAYS
    assert sticky_default == DEFAULT_RETENTION_DAYS
    assert cli_default == DEFAULT_RETENTION_DAYS
