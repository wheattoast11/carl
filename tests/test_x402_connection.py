# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Tests for carl_studio.x402_connection.PaymentConnection.

Test code deliberately introspects private attributes to exercise the FSM,
breaker state, and chain wiring. The autouse fixtures above the test classes
are referenced by pytest by decorator, not by name — pyright cannot see the
indirection, hence ``reportUnusedFunction=false``.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from carl_core.connection import (
    ConnectionState,
    ConnectionTrust,
    ConnectionUnavailableError,
    reset_registry,
)
from carl_core.errors import NetworkError
from carl_core.interaction import InteractionChain
from carl_core.retry import CircuitBreaker, CircuitState

import carl_studio.x402 as x402_module
from carl_studio.x402 import X402Client, X402Config
from carl_studio.x402_connection import (
    PaymentConnection,
    X402SDKClient,
    create_payment_connection,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    """Each test starts with an empty connection registry."""
    reset_registry()


@pytest.fixture(autouse=True)
def _reset_breaker() -> Any:
    """Snapshot and restore _FACILITATOR_BREAKER so tests don't leak state."""
    breaker = x402_module._FACILITATOR_BREAKER
    saved_state = breaker._state
    saved_failures = breaker._consecutive_failures
    saved_opened_at = breaker._opened_at
    yield
    breaker._state = saved_state
    breaker._consecutive_failures = saved_failures
    breaker._opened_at = saved_opened_at


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakePaymentConnection(PaymentConnection):
    """PaymentConnection with a fake client injected during _connect.

    Avoids the real SDK probe — test-side control over which client type is
    bound and what its methods return.
    """

    def __init__(
        self,
        *,
        fake_client: Any,
        sdk_openable: bool = True,
        authenticate_exc: BaseException | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self._fake_client = fake_client
        self._sdk_openable = sdk_openable
        self._authenticate_exc = authenticate_exc
        self.connect_calls = 0
        self.close_calls = 0

    async def _connect(self) -> None:
        # Preserve the breaker-early-raise contract from the base.
        if x402_module._FACILITATOR_BREAKER.state == CircuitState.OPEN:
            raise ConnectionUnavailableError(
                "x402 facilitator circuit breaker is open",
                context={"spec_name": self.spec.name},
            )
        self._client = self._fake_client
        self.connect_calls += 1

    async def _authenticate(self) -> None:
        if self._authenticate_exc is not None:
            raise self._authenticate_exc
        # No real signer registration in tests.

    async def _close(self) -> None:
        self._client = None
        self.close_calls += 1


def _make_sdk_mock() -> MagicMock:
    mock = MagicMock(spec=X402SDKClient)
    mock.get = AsyncMock(return_value={"status": 200, "body": b"ok"})
    mock.verify = AsyncMock(return_value={"isValid": True})
    mock.get_supported = AsyncMock(return_value={"kinds": ["exact"]})
    return mock


# ---------------------------------------------------------------------------
# Construction / spec surface
# ---------------------------------------------------------------------------


class TestPaymentConnectionSpec:
    def test_spec_fields_are_correct(self) -> None:
        conn = PaymentConnection()
        spec = conn.spec
        assert spec.name == "carl.payment.x402"
        assert spec.scope.value == "3p"
        assert spec.kind.value == "payment"
        assert spec.direction.value == "egress"
        assert spec.transport.value == "http"
        assert spec.trust == ConnectionTrust.METERED

    def test_to_dict_exposes_metered_trust(self) -> None:
        conn = PaymentConnection(connection_id="conn-test")
        d = conn.to_dict()
        assert d["spec"]["trust"] == "metered"
        assert d["spec"]["kind"] == "payment"
        assert d["state"] == "init"

    def test_init_accepts_interaction_chain_without_collision(self) -> None:
        chain = InteractionChain()
        conn = PaymentConnection(interaction_chain=chain)
        assert conn._chain is chain  # noqa: SLF001 — deliberate introspection

    def test_defaults(self) -> None:
        conn = PaymentConnection()
        assert conn.facilitator_url == "https://x402.org/facilitator"
        assert conn.chain_name == "base"
        assert conn.client is None
        assert conn.state == ConnectionState.INIT


# ---------------------------------------------------------------------------
# Lifecycle: open / close round-trip
# ---------------------------------------------------------------------------


class TestPaymentConnectionLifecycle:
    def test_open_close_round_trip(self) -> None:
        async def go() -> None:
            fake = _make_sdk_mock()
            conn = _FakePaymentConnection(fake_client=fake)
            await conn.open()
            assert conn.state == ConnectionState.READY
            assert conn.connect_calls == 1
            await conn.close()
            assert conn.state == ConnectionState.CLOSED
            assert conn.close_calls == 1

        asyncio.run(go())

    def test_context_manager_round_trip(self) -> None:
        async def go() -> None:
            fake = _make_sdk_mock()
            async with _FakePaymentConnection(fake_client=fake) as conn:
                assert conn.state == ConnectionState.READY
            assert conn.state == ConnectionState.CLOSED

        asyncio.run(go())

    def test_transact_round_trip_under_context(self) -> None:
        async def go() -> None:
            fake = _make_sdk_mock()
            async with _FakePaymentConnection(fake_client=fake) as conn:
                async with conn.transact("verify"):
                    assert conn.state == ConnectionState.TRANSACTING
                assert conn.state == ConnectionState.READY
                assert conn.stats.transactions_ok == 1

        asyncio.run(go())


# ---------------------------------------------------------------------------
# Telemetry: InteractionChain events
# ---------------------------------------------------------------------------


class TestPaymentConnectionTelemetry:
    def test_chain_records_open_transact_close(self) -> None:
        async def go() -> list[str]:
            chain = InteractionChain()
            fake = _make_sdk_mock()
            async with _FakePaymentConnection(
                fake_client=fake, interaction_chain=chain
            ) as conn:
                async with conn.transact("verify"):
                    pass
            return [s.name for s in chain.steps]

        names = asyncio.run(go())
        assert "connection.open" in names
        assert "connection.verify" in names
        assert "connection.close" in names

    def test_chain_records_get_via_sdk(self) -> None:
        async def go() -> list[str]:
            chain = InteractionChain()
            fake = _make_sdk_mock()
            conn = _FakePaymentConnection(
                fake_client=fake, interaction_chain=chain
            )
            await conn.open()
            await conn.get("https://example.com/paid")
            await conn.close()
            return [s.name for s in chain.steps]

        names = asyncio.run(go())
        assert "connection.get" in names

    def test_chain_records_verify(self) -> None:
        async def go() -> list[str]:
            chain = InteractionChain()
            fake = _make_sdk_mock()
            conn = _FakePaymentConnection(
                fake_client=fake, interaction_chain=chain
            )
            await conn.open()
            await conn.verify({"x": 1}, {"y": 2})
            await conn.close()
            return [s.name for s in chain.steps]

        names = asyncio.run(go())
        assert "connection.verify" in names

    def test_chain_records_get_supported(self) -> None:
        async def go() -> list[str]:
            chain = InteractionChain()
            fake = _make_sdk_mock()
            conn = _FakePaymentConnection(
                fake_client=fake, interaction_chain=chain
            )
            await conn.open()
            await conn.get_supported()
            await conn.close()
            return [s.name for s in chain.steps]

        names = asyncio.run(go())
        assert "connection.get_supported" in names


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------


class TestPaymentConnectionBreaker:
    def test_open_raises_when_breaker_is_open(self) -> None:
        """Swap the module-level breaker for a fast-tripping one, trip it,
        then verify open() surfaces ConnectionUnavailableError."""

        fast_breaker = CircuitBreaker(
            failure_threshold=1,
            reset_s=3600.0,
            tracked_exceptions=(NetworkError,),
        )
        # Trip it.
        try:
            with fast_breaker:
                raise NetworkError("boom")
        except NetworkError:
            pass
        assert fast_breaker.state == CircuitState.OPEN

        async def go() -> None:
            fake = _make_sdk_mock()
            conn = _FakePaymentConnection(fake_client=fake)
            with pytest.raises(ConnectionUnavailableError):
                await conn.open()
            # Connection must land on ERROR after a failed open.
            assert conn.state == ConnectionState.ERROR

        with patch.object(x402_module, "_FACILITATOR_BREAKER", fast_breaker):
            asyncio.run(go())

    def test_open_succeeds_when_breaker_is_closed(self) -> None:
        """Baseline: a CLOSED breaker does not block open()."""

        fresh_breaker = CircuitBreaker(
            failure_threshold=5,
            reset_s=60.0,
            tracked_exceptions=(NetworkError,),
        )
        assert fresh_breaker.state == CircuitState.CLOSED

        async def go() -> None:
            fake = _make_sdk_mock()
            conn = _FakePaymentConnection(fake_client=fake)
            await conn.open()
            assert conn.state == ConnectionState.READY
            await conn.close()

        with patch.object(x402_module, "_FACILITATOR_BREAKER", fresh_breaker):
            asyncio.run(go())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreatePaymentConnection:
    def test_returns_payment_connection_from_config(self) -> None:
        config = X402Config(
            facilitator_url="https://f.example.com",
            chain="ethereum",
        )
        conn = create_payment_connection(config, wallet_private_key="0xkey")
        assert isinstance(conn, PaymentConnection)
        assert conn.facilitator_url == "https://f.example.com"
        assert conn.chain_name == "ethereum"
        assert conn._wallet_private_key == "0xkey"  # noqa: SLF001

    def test_empty_config_fills_sensible_defaults(self) -> None:
        config = X402Config()  # facilitator_url="", chain="base"
        conn = create_payment_connection(config)
        assert conn.facilitator_url == "https://x402.org/facilitator"
        assert conn.chain_name == "base"

    def test_create_payment_connection_importable_from_x402_connection(self) -> None:
        """The factory lives on carl_studio.x402_connection post-consolidation."""
        from carl_studio.x402_connection import (
            create_payment_connection as conn_factory,
        )

        config = X402Config(facilitator_url="https://x.io", chain="base")
        conn = conn_factory(config)
        assert isinstance(conn, PaymentConnection)


# ---------------------------------------------------------------------------
# Legacy surface: must remain importable and instantiable.
# ---------------------------------------------------------------------------


class TestLegacyClientsPreserved:
    def test_x402_client_still_importable_and_instantiable(self) -> None:
        config = X402Config(facilitator_url="https://f.io", chain="base")
        client = X402Client(config)
        assert client._config.facilitator_url == "https://f.io"  # noqa: SLF001

    def test_x402_sdk_client_still_importable_and_instantiable(self) -> None:
        client = X402SDKClient(
            wallet_private_key="0xdead",
            facilitator_url="https://f.io",
            chain="base",
        )
        assert client._facilitator_url == "https://f.io"  # noqa: SLF001
        assert client._chain == "base"  # noqa: SLF001

    def test_create_x402_client_still_returns_fallback_without_sdk(self) -> None:
        from carl_studio.x402_connection import create_x402_client

        config = X402Config(facilitator_url="https://f.io", chain="base")
        with patch.dict(sys.modules, {"x402": None}):
            client = create_x402_client(config)
        assert isinstance(client, X402Client)


# ---------------------------------------------------------------------------
# urllib fallback path in PaymentConnection._connect
# ---------------------------------------------------------------------------


class TestPaymentConnectionFallback:
    def test_connect_binds_urllib_client_when_sdk_absent(self) -> None:
        """When sdk_available() is False, _connect stores an X402Client."""

        async def go() -> X402Client | X402SDKClient | None:
            with patch(
                "carl_studio.x402_connection.sdk_available",
                return_value=False,
            ):
                conn = PaymentConnection(
                    facilitator_url="https://f.io",
                    chain_name="base",
                )
                await conn.open()
                client = conn.client
                await conn.close()
                return client

        client = asyncio.run(go())
        assert isinstance(client, X402Client)

    def test_connect_binds_sdk_client_when_sdk_available(self) -> None:
        """When sdk_available() is True, _connect stores an X402SDKClient."""

        async def go() -> X402Client | X402SDKClient | None:
            with patch(
                "carl_studio.x402_connection.sdk_available",
                return_value=True,
            ):
                conn = PaymentConnection(
                    facilitator_url="https://f.io",
                    chain_name="base",
                )
                await conn.open()
                client = conn.client
                await conn.close()
                return client

        client = asyncio.run(go())
        assert isinstance(client, X402SDKClient)

    def test_verify_rejects_fallback_client(self) -> None:
        """verify requires the SDK — fallback clients raise."""

        async def go() -> None:
            fallback = X402Client(
                X402Config(facilitator_url="https://f.io", chain="base")
            )
            conn = _FakePaymentConnection(fake_client=fallback)
            await conn.open()
            with pytest.raises(ConnectionUnavailableError):
                await conn.verify({}, {})
            # verify() drops to DEGRADED on failure — that's the contract.
            assert conn.state == ConnectionState.DEGRADED
            await conn.close()

        asyncio.run(go())

    def test_get_supported_rejects_fallback_client(self) -> None:
        """get_supported requires the SDK — fallback clients raise."""

        async def go() -> None:
            fallback = X402Client(
                X402Config(facilitator_url="https://f.io", chain="base")
            )
            conn = _FakePaymentConnection(fake_client=fallback)
            await conn.open()
            with pytest.raises(ConnectionUnavailableError):
                await conn.get_supported()
            assert conn.state == ConnectionState.DEGRADED
            await conn.close()

        asyncio.run(go())
