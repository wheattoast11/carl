"""PaymentConnection — x402 payment rail as a carl_core.connection primitive.

This module ports the existing :mod:`carl_studio.x402` + :mod:`carl_studio.x402_sdk`
client surface onto :class:`carl_core.connection.AsyncBaseConnection`. The
underlying HTTP client (official x402 SDK or the urllib fallback) is selected
lazily inside :meth:`PaymentConnection._connect`; neither path is imported at
module load time so that the `x402` and `eth_account` extras remain optional.

The module-level :data:`_FACILITATOR_BREAKER` instance is re-used from
:mod:`carl_studio.x402` so the historical failure accounting (single breaker
across the process, tracking only infrastructure exceptions) is preserved.
When the breaker is ``OPEN``, :meth:`open` surfaces a
:class:`ConnectionUnavailableError` before any transport call is attempted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from carl_core.connection import (
    AsyncBaseConnection,
    ConnectionAuthError,
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
    ConnectionUnavailableError,
)
from carl_core.errors import NetworkError

from carl_studio.x402 import (
    _FACILITATOR_BREAKER,  # pyright: ignore[reportPrivateUsage]
    X402Client,
    X402Config,
)
from carl_studio.x402_sdk import X402SDKClient, sdk_available

if TYPE_CHECKING:
    from carl_core.interaction import InteractionChain


_DEFAULT_FACILITATOR_URL = "https://x402.org/facilitator"
_DEFAULT_CHAIN_NAME = "base"


_PAYMENT_SPEC = ConnectionSpec(
    name="carl.payment.x402",
    scope=ConnectionScope.THREE_P,
    kind=ConnectionKind.PAYMENT,
    direction=ConnectionDirection.EGRESS,
    transport=ConnectionTransport.HTTP,
    trust=ConnectionTrust.METERED,
    metadata={
        "facilitator": _DEFAULT_FACILITATOR_URL,
        "chain": _DEFAULT_CHAIN_NAME,
    },
)


class PaymentConnection(AsyncBaseConnection):
    """x402 facilitator connection with FSM + InteractionChain telemetry.

    Wraps either the official ``x402`` SDK (when installed) or the stdlib
    ``urllib`` fallback (:class:`carl_studio.x402.X402Client`) so callers get
    a single, lifecycle-aware surface regardless of which backend is available.

    Parameters
    ----------
    wallet_private_key
        EVM wallet private key for signing payments. Optional; empty disables
        EVM signer registration (verify/get-supported still work).
    facilitator_url
        Facilitator endpoint. Defaults to the public x402.org facilitator.
    chain_name
        Blockchain identifier (``"base"``, ``"ethereum"``, ...). Renamed from
        ``chain`` to avoid collision with :class:`AsyncBaseConnection`'s
        ``chain`` kwarg (which takes an :class:`InteractionChain`).
    interaction_chain
        Optional :class:`InteractionChain` — forwarded to the base as ``chain``
        so connection.* telemetry events are recorded.
    connection_id
        Stable identifier; auto-generated if omitted.
    """

    spec: ConnectionSpec = _PAYMENT_SPEC

    def __init__(
        self,
        *,
        wallet_private_key: str = "",
        facilitator_url: str = _DEFAULT_FACILITATOR_URL,
        chain_name: str = _DEFAULT_CHAIN_NAME,
        interaction_chain: InteractionChain | None = None,
        connection_id: str | None = None,
    ) -> None:
        super().__init__(chain=interaction_chain, connection_id=connection_id)
        self._wallet_private_key = wallet_private_key
        self._facilitator_url = facilitator_url
        self._chain_name = chain_name
        self._client: X402SDKClient | X402Client | None = None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def client(self) -> X402SDKClient | X402Client | None:
        """Underlying x402 client (SDK or urllib fallback). None until opened."""
        return self._client

    @property
    def facilitator_url(self) -> str:
        return self._facilitator_url

    @property
    def chain_name(self) -> str:
        return self._chain_name

    # ------------------------------------------------------------------
    # AsyncBaseConnection lifecycle hooks
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Lazy-init the underlying x402 client.

        Chooses :class:`X402SDKClient` when the official ``x402`` package is
        importable; otherwise falls back to the urllib-based
        :class:`X402Client`. The breaker is consulted early so that a fully
        tripped facilitator is surfaced before any transport attempt.
        """
        if _FACILITATOR_BREAKER.state == "open":
            raise ConnectionUnavailableError(
                "x402 facilitator circuit breaker is open",
                context={
                    "facilitator": self._facilitator_url,
                    "chain": self._chain_name,
                    "spec_name": self.spec.name,
                },
            )

        if sdk_available():
            self._client = X402SDKClient(
                wallet_private_key=self._wallet_private_key,
                facilitator_url=self._facilitator_url,
                chain=self._chain_name,
            )
        else:
            # urllib fallback — takes X402Config, not kwargs.
            self._client = X402Client(
                X402Config(
                    facilitator_url=self._facilitator_url,
                    chain=self._chain_name,
                )
            )

    async def _authenticate(self) -> None:
        """Register the EVM signer when a wallet key is configured.

        The underlying SDK path also registers inside its own
        ``_ensure_client`` call; invoking it here makes failures observable at
        the ``AUTHENTICATING`` phase instead of silently deferring them to the
        first ``get``/``verify`` call.
        """
        if self._client is None:
            raise ConnectionAuthError(
                "no x402 client available; _connect must be called first",
                context={"spec_name": self.spec.name},
            )

        if not self._wallet_private_key:
            # No wallet key — nothing to authenticate. Verify/get-supported
            # continue to work; only signer-dependent flows fail downstream.
            return

        if not isinstance(self._client, X402SDKClient):
            # urllib fallback has no signer concept.
            return

        try:
            # Trigger the SDK client's signer registration path up-front so a
            # bad wallet key fails fast with a typed error instead of at the
            # first `.get(...)`.
            self._client._ensure_client()  # pyright: ignore[reportPrivateUsage]
        except ImportError:
            # SDK missing between sdk_available() and now — treat as unavailable.
            raise ConnectionUnavailableError(
                "x402 SDK import failed after probe succeeded",
                context={"spec_name": self.spec.name},
            ) from None
        except Exception as exc:
            raise ConnectionAuthError(
                f"EVM signer registration failed: {exc}",
                context={
                    "facilitator": self._facilitator_url,
                    "chain": self._chain_name,
                    "spec_name": self.spec.name,
                },
            ) from exc

    async def _close(self) -> None:
        """Release underlying client resources.

        The SDK path uses :class:`httpx.AsyncClient` scoped to each ``get`` via
        ``async with x402HttpxClient(...)``, so there is nothing owned at the
        ``PaymentConnection`` layer to close. Dropping the reference is
        sufficient; the urllib fallback has no durable state either.
        """
        self._client = None

    # ------------------------------------------------------------------
    # Payment operations — wrapped in transact() for FSM + telemetry
    # ------------------------------------------------------------------

    async def get(self, url: str, **kwargs: Any) -> Any:
        """x402-aware GET. Returns the httpx response (SDK path) or bytes
        (urllib fallback)."""
        async with self.transact("get"):
            client = self._require_client()
            try:
                if isinstance(client, X402SDKClient):
                    return await client.get(url, **kwargs)
                # urllib fallback path: check for 402, negotiate, execute.
                return await self._get_via_fallback(client, url, **kwargs)
            except (NetworkError, ConnectionError, TimeoutError, OSError) as exc:
                # Feed the facilitator breaker on infrastructure faults.
                self._record_breaker_failure(exc)
                raise

    async def verify(
        self, payload: dict[str, Any], requirements: dict[str, Any]
    ) -> dict[str, Any]:
        """Verify a payment payload against facilitator requirements."""
        async with self.transact("verify"):
            client = self._require_client()
            if not isinstance(client, X402SDKClient):
                raise ConnectionUnavailableError(
                    "verify requires the official x402 SDK "
                    "(pip install x402)",
                    context={"spec_name": self.spec.name},
                )
            try:
                return await client.verify(payload, requirements)
            except (NetworkError, ConnectionError, TimeoutError, OSError) as exc:
                self._record_breaker_failure(exc)
                raise

    async def get_supported(self) -> Any:
        """Ask the facilitator what payment schemes / chains it accepts."""
        async with self.transact("get_supported"):
            client = self._require_client()
            if not isinstance(client, X402SDKClient):
                raise ConnectionUnavailableError(
                    "get_supported requires the official x402 SDK "
                    "(pip install x402)",
                    context={"spec_name": self.spec.name},
                )
            try:
                return await client.get_supported()
            except (NetworkError, ConnectionError, TimeoutError, OSError) as exc:
                self._record_breaker_failure(exc)
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_client(self) -> X402SDKClient | X402Client:
        if self._client is None:
            raise ConnectionUnavailableError(
                "x402 client is not initialized; call open() first",
                context={"spec_name": self.spec.name},
            )
        return self._client

    @staticmethod
    async def _get_via_fallback(
        client: X402Client, url: str, **_kwargs: Any
    ) -> bytes:
        """urllib-path GET — HEAD for 402, negotiate, then execute."""
        requirement = client.check_x402(url)
        if requirement is None:
            # No 402 / unsupported — caller should use a plain HTTP client.
            raise ConnectionUnavailableError(
                "resource did not return 402; urllib fallback cannot serve "
                "non-x402 resources",
                context={"url": url},
            )
        auth = client.negotiate(requirement)
        token = str(auth.get("token", ""))
        if not token:
            raise ConnectionUnavailableError(
                "facilitator returned no payment token",
                context={"url": url, "facilitator": requirement.facilitator},
            )
        return client.execute(url, token)

    def _record_breaker_failure(self, exc: BaseException) -> None:
        """Route an infrastructure failure through the module-level breaker.

        We use the breaker's context-manager semantics directly so the
        historical accounting — only ``tracked_exceptions`` count, success
        always resets — stays identical to the pre-connection codepath.
        """
        try:
            with _FACILITATOR_BREAKER:
                raise exc
        except type(exc):
            # Breaker has already recorded the failure; re-surface is handled
            # by the caller's own ``raise`` — we swallow here to avoid double
            # propagation.
            return
        except Exception:
            # Breaker was OPEN and re-raised as CARLError("carl.circuit_open")
            # — that path is benign; the outer caller will still raise the
            # original infrastructure exception.
            return


def create_payment_connection(
    config: X402Config,
    wallet_private_key: str = "",
    interaction_chain: InteractionChain | None = None,
) -> PaymentConnection:
    """Factory — build a :class:`PaymentConnection` from an :class:`X402Config`.

    This is the recommended surface going forward. The older
    :func:`carl_studio.x402_sdk.create_x402_client` remains available and
    unchanged for backward compatibility; it will gain a ``DeprecationWarning``
    in v0.6.
    """
    return PaymentConnection(
        wallet_private_key=wallet_private_key,
        facilitator_url=config.facilitator_url or _DEFAULT_FACILITATOR_URL,
        chain_name=config.chain or _DEFAULT_CHAIN_NAME,
        interaction_chain=interaction_chain,
    )


__all__ = [
    "PaymentConnection",
    "create_payment_connection",
]
