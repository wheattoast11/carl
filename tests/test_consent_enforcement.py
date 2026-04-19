"""Runtime consent-gate enforcement tests.

These tests verify that the four network-egress boundaries raise
``ConsentError`` when the corresponding consent flag is off:

* ``TELEMETRY``           -> ``carl_studio.sync.push`` / ``pull`` / queue drain
* ``TELEMETRY``           -> ``MCPServerConnection._authenticate``
* ``CONTRACT_WITNESSING`` -> ``X402Client.execute`` and
                             ``PaymentConnection.get``
* ``OBSERVABILITY``       -> ``TrackioSource._get_client`` (the only
                             remote observe path)

All tests construct a :class:`ConsentManager` bound to a local
``FakeDB`` and install it via ``monkeypatch`` so the production
fail-closed default is exercised without touching the real ``~/.carl``
state. The autouse ``_grant_all_consent_by_default`` fixture from
``tests/conftest.py`` is neutralized per-test with
``monkeypatch.undo()`` first so the gate sees the FakeDB-backed state.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from carl_studio.consent import (
    ConsentError,
    ConsentFlagKey,
    ConsentManager,
    ConsentState,
    consent_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDB:
    """Minimal LocalDB substitute for consent-state roundtrips."""

    def __init__(self) -> None:
        self.config: dict[str, str] = {}

    def get_config(self, key: str, default: str | None = None) -> str | None:
        return self.config.get(key, default)

    def set_config(self, key: str, value: str) -> None:
        self.config[key] = value


def _install_all_off(monkeypatch: pytest.MonkeyPatch) -> _FakeDB:
    """Return a FakeDB+ConsentManager wiring with every flag off.

    Also disables the autouse "grant all" fixture so the real gate runs.
    """
    monkeypatch.undo()  # drop autouse grant-all
    db = _FakeDB()
    manager = ConsentManager(db=db)
    # Persist an explicit all-off state so ``load()`` reads back the
    # defaults instead of relying on the empty-config branch.
    manager.save(ConsentState())
    # Patch the no-arg ``ConsentManager()`` constructor used inside
    # ``consent_gate`` so production-path callers hit our FakeDB.
    original_init = ConsentManager.__init__

    def _init(self: ConsentManager, db: Any | None = None) -> None:
        original_init(self, db=db if db is not None else _FakeDB.__new__(_FakeDB))
        # Actually install our fake DB with the off state.
        self._db = manager._db  # type: ignore[attr-defined]

    monkeypatch.setattr(ConsentManager, "__init__", _init)
    return db


# ---------------------------------------------------------------------------
# consent_gate primitive
# ---------------------------------------------------------------------------


class TestConsentGate:
    def test_gate_blocks_unknown_flag(self) -> None:
        # Unknown-flag path runs before ``is_granted``, so the autouse
        # grant-all patch does not affect the assertion.
        mgr = ConsentManager(db=_FakeDB())
        with pytest.raises(ConsentError) as exc:
            consent_gate("not_a_flag", manager=mgr)
        assert exc.value.code == "carl.consent.unknown_flag"
        assert exc.value.context["flag"] == "not_a_flag"

    def test_gate_blocks_when_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()  # drop autouse grant-all; exercise real load()
        mgr = ConsentManager(db=_FakeDB())
        with pytest.raises(ConsentError) as exc:
            consent_gate(ConsentFlagKey.TELEMETRY, manager=mgr)
        assert exc.value.code == "carl.consent.denied"
        assert exc.value.context["flag"] == "telemetry"

    def test_gate_allows_when_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()
        db = _FakeDB()
        mgr = ConsentManager(db=db)
        mgr.update("telemetry", True)
        # No raise -> success.
        consent_gate(ConsentFlagKey.TELEMETRY, manager=mgr)

    def test_gate_accepts_string_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()
        db = _FakeDB()
        mgr = ConsentManager(db=db)
        mgr.update("contract_witnessing", True)
        consent_gate("contract_witnessing", manager=mgr)

    def test_is_granted_unknown_flag_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.undo()
        mgr = ConsentManager(db=_FakeDB())
        assert mgr.is_granted("does_not_exist") is False


# ---------------------------------------------------------------------------
# sync.py — TELEMETRY
# ---------------------------------------------------------------------------


class TestSyncGate:
    def test_sync_push_blocked_when_telemetry_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_all_off(monkeypatch)
        from carl_studio.sync import push

        with pytest.raises(ConsentError) as exc:
            push()
        assert exc.value.code == "carl.consent.denied"
        assert exc.value.context["flag"] == "telemetry"

    def test_sync_pull_blocked_when_telemetry_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_all_off(monkeypatch)
        from carl_studio.sync import pull

        with pytest.raises(ConsentError) as exc:
            pull()
        assert exc.value.context["flag"] == "telemetry"

    def test_sync_queue_drain_blocked_when_telemetry_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_all_off(monkeypatch)
        from carl_studio.sync import process_sync_queue

        with pytest.raises(ConsentError):
            process_sync_queue()


# ---------------------------------------------------------------------------
# MCPServerConnection — TELEMETRY
# ---------------------------------------------------------------------------


class TestMCPConnectGate:
    def test_mcp_connect_blocked_when_telemetry_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("mcp")
        _install_all_off(monkeypatch)
        from carl_studio.mcp.connection import MCPServerConnection

        async def go() -> None:
            conn = MCPServerConnection(transport="stdio")
            with pytest.raises(ConsentError) as exc:
                await conn.open()
            assert exc.value.context["flag"] == "telemetry"

        asyncio.run(go())


# ---------------------------------------------------------------------------
# x402 — CONTRACT_WITNESSING
# ---------------------------------------------------------------------------


class TestX402Gate:
    def test_x402_execute_blocked_when_contract_witnessing_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_all_off(monkeypatch)
        from carl_studio.x402 import X402Client, X402Config

        client = X402Client(X402Config())
        with pytest.raises(ConsentError) as exc:
            client.execute("https://example.com/paid", "dummy-token")
        assert exc.value.code == "carl.consent.denied"
        assert exc.value.context["flag"] == "contract_witnessing"

    def test_payment_connection_get_blocked_when_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_all_off(monkeypatch)
        from carl_studio.x402_connection import PaymentConnection

        async def go() -> None:
            conn = PaymentConnection(
                facilitator_url="https://example.com/facilitator",
                chain_name="base",
            )
            # No .open() needed — consent_gate in .get() runs before the
            # FSM check, so calling get() directly on a fresh connection
            # surfaces the ConsentError without needing real networking.
            with pytest.raises(ConsentError) as exc:
                await conn.get("https://example.com/paid")
            assert exc.value.context["flag"] == "contract_witnessing"

        asyncio.run(go())


# ---------------------------------------------------------------------------
# TrackioSource — OBSERVABILITY
# ---------------------------------------------------------------------------


class TestObserveGate:
    def test_coherence_probe_blocked_when_observability_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trackio is the only outbound observability path; blocking its
        client init is the runtime correlate of "coherence probes sent
        to carl.camp" in the consent-flag description."""
        _install_all_off(monkeypatch)
        from carl_studio.observe.data_source import TrackioSource

        src = TrackioSource(space="fake/space")
        with pytest.raises(ConsentError) as exc:
            src._get_client()
        assert exc.value.context["flag"] == "observability"
