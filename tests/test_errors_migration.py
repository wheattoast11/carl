"""Smoke tests for the CARLError migration across carl-studio.

Verifies that the seven legacy ``Exception``-based error classes now inherit
from :class:`carl_core.errors.CARLError`, carry stable ``code`` values in the
``carl.<namespace>`` scheme, and — for the network-capable sites — also
subclass :class:`carl_core.errors.NetworkError`.

These tests are deliberately thin; deep behavioral coverage lives in the
per-module suites (test_contract.py, test_credits.py, ...).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from carl_core.errors import CARLError, NetworkError

from carl_studio.billing import BillingError
from carl_studio.consent import ConsentError
from carl_studio.contract import ContractError
from carl_studio.credits.balance import CreditError, CreditNetworkError
from carl_studio.marketplace import MarketplaceError, MarketplaceNetworkError
from carl_studio.sync import SyncError
from carl_studio.x402 import X402Error


# ---------------------------------------------------------------------------
# Inheritance: every migrated error is a CARLError subclass.
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_contract_error_is_carl_error(self) -> None:
        assert issubclass(ContractError, CARLError)

    def test_consent_error_is_carl_error(self) -> None:
        assert issubclass(ConsentError, CARLError)

    def test_marketplace_error_is_carl_error(self) -> None:
        assert issubclass(MarketplaceError, CARLError)

    def test_sync_error_is_carl_error(self) -> None:
        assert issubclass(SyncError, CARLError)

    def test_x402_error_is_carl_error(self) -> None:
        assert issubclass(X402Error, CARLError)

    def test_billing_error_is_carl_error(self) -> None:
        assert issubclass(BillingError, CARLError)

    def test_credit_error_is_carl_error(self) -> None:
        assert issubclass(CreditError, CARLError)


# ---------------------------------------------------------------------------
# Multi-inheritance: network variants are both a NetworkError AND the
# legacy ``*Error`` class so ``except MarketplaceError`` still catches them.
# ---------------------------------------------------------------------------


class TestNetworkMultiInheritance:
    def test_marketplace_network_error_is_both(self) -> None:
        assert issubclass(MarketplaceNetworkError, NetworkError)
        assert issubclass(MarketplaceNetworkError, MarketplaceError)

    def test_credit_network_error_is_both(self) -> None:
        assert issubclass(CreditNetworkError, NetworkError)
        assert issubclass(CreditNetworkError, CreditError)

    def test_marketplace_network_catchable_as_legacy_marketplace_error(self) -> None:
        with pytest.raises(MarketplaceError):
            raise MarketplaceNetworkError("boom")

    def test_credit_network_catchable_as_legacy_credit_error(self) -> None:
        with pytest.raises(CreditError):
            raise CreditNetworkError("boom")


# ---------------------------------------------------------------------------
# Stable namespaced codes.
# ---------------------------------------------------------------------------


class TestCodes:
    def test_contract_error_default_code(self) -> None:
        assert ContractError("x").code == "carl.contract"

    def test_consent_error_default_code(self) -> None:
        assert ConsentError("x").code == "carl.consent"

    def test_marketplace_error_code_is_carl_marketplace_http(self) -> None:
        assert MarketplaceError("x").code == "carl.marketplace.http"

    def test_marketplace_network_error_code(self) -> None:
        assert MarketplaceNetworkError("x").code == "carl.marketplace.network"

    def test_sync_error_default_code(self) -> None:
        assert SyncError("x").code == "carl.sync"

    def test_x402_error_default_code(self) -> None:
        assert X402Error("x").code == "carl.x402"

    def test_billing_error_default_code(self) -> None:
        assert BillingError("x").code == "carl.billing"

    def test_credit_error_default_code(self) -> None:
        assert CreditError("x").code == "carl.credit"

    def test_credit_network_error_code(self) -> None:
        assert CreditNetworkError("x").code == "carl.credit.network"


# ---------------------------------------------------------------------------
# Secret redaction flows through the base class.
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_consent_error_redacts_secret_context(self) -> None:
        err = ConsentError("x", context={"token": "abc123", "key": "hf_XXXX"})
        d = err.to_dict()
        assert d["context"]["token"] == "***REDACTED***"
        assert d["context"]["key"] == "***REDACTED***"
        assert d["code"] == "carl.consent"

    def test_contract_error_preserves_non_secret_context(self) -> None:
        err = ContractError("x", context={"contract_id": "abc"})
        d = err.to_dict()
        assert d["context"]["contract_id"] == "abc"

    def test_sync_error_redacts_authorization_context(self) -> None:
        err = SyncError("x", context={"authorization": "Bearer xyz"})
        d = err.to_dict()
        assert d["context"]["authorization"] == "***REDACTED***"


# ---------------------------------------------------------------------------
# Network-error raising behavior: the two HTTPError / URLError sites in
# marketplace.py and credits/balance.py raise the network subclass.
# ---------------------------------------------------------------------------


class TestNetworkRaising:
    def test_marketplace_urlerror_raises_network_subclass(self) -> None:
        import urllib.error

        from carl_studio.marketplace import MarketplaceClient

        client = MarketplaceClient(supabase_url="https://f.invalid", jwt="x")

        def boom(*_args: object, **_kw: object) -> object:
            raise urllib.error.URLError("nope")

        with patch("urllib.request.urlopen", side_effect=boom):
            with pytest.raises(MarketplaceNetworkError) as exc_info:
                client._request("marketplace-list", params={"type": "models"})
            # Catchable as either arm:
            assert isinstance(exc_info.value, MarketplaceError)
            assert isinstance(exc_info.value, NetworkError)
            assert exc_info.value.code == "carl.marketplace.network"

    def test_credit_urlerror_raises_network_subclass(self) -> None:
        import urllib.error

        from carl_studio.credits.balance import deduct_credits

        def boom(*_args: object, **_kw: object) -> object:
            raise urllib.error.URLError("nope")

        with patch("urllib.request.urlopen", side_effect=boom):
            with pytest.raises(CreditNetworkError) as exc_info:
                deduct_credits("jwt", "https://f.invalid", 1, "job-1")
            assert isinstance(exc_info.value, CreditError)
            assert isinstance(exc_info.value, NetworkError)
            assert exc_info.value.code == "carl.credit.network"
