"""Tests for the x402 SDK adapter (now consolidated into x402_connection).

The legacy module ``carl_studio.x402_sdk`` was removed. All of its symbols
moved to :mod:`carl_studio.x402_connection`. These tests exercise the SDK
adapter behavior (ensure_client, EVM signer registration, create_x402_client
fallback) at its new home AND verify that the old module import fails.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from carl_studio.x402_connection import (
    X402SDKClient,
    create_x402_client,
    sdk_available,
)


class TestDeletedModuleRaises:
    def test_x402_sdk_module_gone(self) -> None:
        """carl_studio.x402_sdk no longer exists."""
        with pytest.raises(ModuleNotFoundError):
            import carl_studio.x402_sdk  # noqa: F401

    def test_create_payment_connection_importable_from_connection_module(self) -> None:
        """create_payment_connection lives on x402_connection."""
        from carl_studio.x402_connection import create_payment_connection  # noqa: F401


class TestSdkAvailable:
    def test_returns_false_when_not_installed(self) -> None:
        """sdk_available() is False when the x402 package is absent."""
        with patch.dict(sys.modules, {"x402": None}):
            assert sdk_available() is False

    def test_returns_true_when_installed(self) -> None:
        """sdk_available() is True when the x402 package is importable."""
        fake_module = MagicMock()
        with patch.dict(sys.modules, {"x402": fake_module}):
            assert sdk_available() is True


class TestX402SDKClientInit:
    def test_init_does_not_crash(self) -> None:
        """Construction should succeed without the SDK installed."""
        client = X402SDKClient(
            wallet_private_key="0xdead",
            facilitator_url="https://f.example.com",
            chain="base",
        )
        assert client._facilitator_url == "https://f.example.com"
        assert client._chain == "base"
        assert client._sdk_client is None

    def test_init_defaults(self) -> None:
        """Default values match expected fallbacks."""
        client = X402SDKClient()
        assert client._wallet_key == ""
        assert client._facilitator_url == "https://x402.org/facilitator"
        assert client._chain == "base"


class TestEnsureClient:
    def test_raises_import_error_when_sdk_missing(self) -> None:
        """_ensure_client raises ImportError with install hint when SDK absent."""
        client = X402SDKClient()
        with patch.dict(sys.modules, {"x402": None}):
            with pytest.raises(ImportError, match="pip install x402"):
                client._ensure_client()

    def test_initializes_sdk_client_when_available(self) -> None:
        """_ensure_client sets _sdk_client when the SDK is importable."""
        fake_x402_module = MagicMock()
        fake_client_instance = MagicMock()
        fake_x402_module.x402Client.return_value = fake_client_instance

        with patch.dict(sys.modules, {"x402": fake_x402_module}):
            client = X402SDKClient()
            client._ensure_client()
            assert client._sdk_client is fake_client_instance

    def test_idempotent(self) -> None:
        """Calling _ensure_client twice does not re-create the SDK client."""
        fake_x402_module = MagicMock()
        first_instance = MagicMock()
        fake_x402_module.x402Client.return_value = first_instance

        with patch.dict(sys.modules, {"x402": fake_x402_module}):
            client = X402SDKClient()
            client._ensure_client()
            fake_x402_module.x402Client.return_value = MagicMock()
            client._ensure_client()
            assert client._sdk_client is first_instance

    def test_registers_evm_signer_when_wallet_key_provided(self) -> None:
        """EVM signer is registered when wallet key is present and deps available."""
        fake_x402 = MagicMock()
        fake_eth = MagicMock()
        fake_evm = MagicMock()
        fake_register = MagicMock()

        modules = {
            "x402": fake_x402,
            "x402.mechanisms": MagicMock(),
            "x402.mechanisms.evm": fake_evm,
            "x402.mechanisms.evm.exact": MagicMock(),
            "x402.mechanisms.evm.exact.register": fake_register,
            "eth_account": fake_eth,
        }

        with patch.dict(sys.modules, modules):
            client = X402SDKClient(wallet_private_key="0xkey")
            client._ensure_client()
            fake_register.register_exact_evm_client.assert_called_once()

    def test_skips_evm_signer_when_no_wallet_key(self) -> None:
        """EVM signer is NOT registered when wallet key is empty."""
        fake_x402 = MagicMock()
        with patch.dict(sys.modules, {"x402": fake_x402}):
            client = X402SDKClient(wallet_private_key="")
            client._ensure_client()
            # No crash, no signer registration attempted


class TestCreateX402Client:
    def test_returns_custom_client_when_sdk_unavailable(self) -> None:
        """create_x402_client returns X402Client when SDK is absent."""
        from carl_studio.x402 import X402Client, X402Config

        config = X402Config(facilitator_url="https://f.io", chain="base")
        with patch.dict(sys.modules, {"x402": None}):
            client = create_x402_client(config)
        assert isinstance(client, X402Client)

    def test_returns_sdk_client_when_available(self) -> None:
        """create_x402_client returns X402SDKClient when SDK is installed."""
        from carl_studio.x402 import X402Config

        fake_module = MagicMock()
        config = X402Config(facilitator_url="https://f.io", chain="ethereum")
        with patch.dict(sys.modules, {"x402": fake_module}):
            client = create_x402_client(config, wallet_private_key="0xkey")
        assert isinstance(client, X402SDKClient)
        assert client._facilitator_url == "https://f.io"
        assert client._chain == "ethereum"
        assert client._wallet_key == "0xkey"

    def test_passes_config_fields_to_custom_client(self) -> None:
        """Fallback X402Client receives the config object."""
        from carl_studio.x402 import X402Client, X402Config

        config = X402Config(
            facilitator_url="https://f.io",
            chain="base",
            wallet_address="0xwallet",
        )
        with patch.dict(sys.modules, {"x402": None}):
            client = create_x402_client(config)
        assert isinstance(client, X402Client)
        assert client._config.wallet_address == "0xwallet"
