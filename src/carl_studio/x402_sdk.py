"""x402 SDK adapter -- wraps the official Coinbase x402 Python package.

Falls back to the custom urllib implementation when the SDK is not installed.
Install: pip install x402
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def sdk_available() -> bool:
    """Check if the official x402 SDK is installed."""
    try:
        import x402  # noqa: F401

        return True
    except ImportError:
        return False


class X402SDKClient:
    """x402 client using the official Coinbase SDK.

    Provides automatic 402 interception and facilitator communication.
    Falls back to custom X402Client when SDK is not available.
    """

    def __init__(
        self,
        wallet_private_key: str = "",
        facilitator_url: str = "https://x402.org/facilitator",
        chain: str = "base",
    ) -> None:
        self._wallet_key = wallet_private_key
        self._facilitator_url = facilitator_url
        self._chain = chain
        self._sdk_client: Any = None

    def _ensure_client(self) -> None:
        """Lazy-init the SDK client with payment scheme registration."""
        if self._sdk_client is not None:
            return

        try:
            from x402 import x402Client  # type: ignore[import-untyped]

            self._sdk_client = x402Client()

            # Register EVM signer if wallet key is provided
            if self._wallet_key:
                self._register_evm_signer()

        except ImportError:
            raise ImportError(
                "x402 SDK required. Install: pip install x402"
            ) from None

    def _register_evm_signer(self) -> None:
        """Register EVM account signer with the SDK client."""
        try:
            from eth_account import Account  # type: ignore[import-untyped]
            from x402.mechanisms.evm import EthAccountSigner  # type: ignore[import-untyped]
            from x402.mechanisms.evm.exact.register import register_exact_evm_client  # type: ignore[import-untyped]

            account = Account.from_key(self._wallet_key)
            register_exact_evm_client(self._sdk_client, EthAccountSigner(account))
        except ImportError:
            logger.debug("eth_account not installed, EVM signing unavailable")
        except Exception:
            logger.warning("EVM signer registration failed", exc_info=True)

    async def get(self, url: str, **kwargs: Any) -> Any:
        """Make an x402-aware GET request.

        Automatically handles 402 responses by negotiating payment.
        Returns the httpx Response object.
        """
        self._ensure_client()
        from x402.http.clients import x402HttpxClient  # type: ignore[import-untyped]

        async with x402HttpxClient(self._sdk_client) as http:
            return await http.get(url, **kwargs)

    async def verify(
        self, payload: dict[str, Any], requirements: dict[str, Any]
    ) -> dict[str, Any]:
        """Verify a payment via the facilitator."""
        self._ensure_client()
        from x402.http import HTTPFacilitatorClient, FacilitatorConfig  # type: ignore[import-untyped]

        facilitator = HTTPFacilitatorClient(
            FacilitatorConfig(url=self._facilitator_url)
        )
        result: dict[str, Any] = await facilitator.verify(payload, requirements)
        return result

    async def get_supported(self) -> Any:
        """Get supported payment kinds from facilitator."""
        self._ensure_client()
        from x402.http import HTTPFacilitatorClient, FacilitatorConfig  # type: ignore[import-untyped]

        facilitator = HTTPFacilitatorClient(
            FacilitatorConfig(url=self._facilitator_url)
        )
        return await facilitator.get_supported()


def create_x402_client(
    config: Any,
    wallet_private_key: str = "",
) -> Any:
    """Create the best available x402 client.

    Returns X402SDKClient if the SDK is installed, otherwise X402Client.
    """
    if sdk_available():
        return X402SDKClient(
            wallet_private_key=wallet_private_key,
            facilitator_url=config.facilitator_url,
            chain=config.chain,
        )

    # Fall back to custom implementation
    from carl_studio.x402 import X402Client

    return X402Client(config)
