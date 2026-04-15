"""Tests for carl_studio.x402 -- X402Config, X402Client, persistence."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

from carl_studio.x402 import (
    PaymentRequirement,
    X402Client,
    X402Config,
    X402Error,
    _parse_x_payment_header,
    load_x402_config,
    save_x402_config,
    x402_config_from_profile,
)


class FakeDB:
    def __init__(self) -> None:
        self.config: dict[str, str] = {}

    def get_config(self, key: str, default: str | None = None) -> str | None:
        return self.config.get(key, default)

    def set_config(self, key: str, value: str) -> None:
        self.config[key] = value


class TestX402Config:
    def test_defaults(self) -> None:
        config = X402Config()
        assert config.enabled is False
        assert config.chain == "base"
        assert config.payment_token == "USDC"
        assert config.auto_approve_below == 0.0

    def test_enabled_flag(self) -> None:
        config = X402Config(enabled=True, wallet_address="0xabc")
        assert config.enabled is True


class TestPaymentRequirement:
    def test_parse_json_header(self) -> None:
        header = json.dumps({
            "amount": "0.01",
            "token": "USDC",
            "chain": "base",
            "recipient": "0xdef",
            "facilitator": "https://x402.org/f",
        })
        req = _parse_x_payment_header(header)
        assert req.amount == "0.01"
        assert req.token == "USDC"
        assert req.facilitator == "https://x402.org/f"

    def test_parse_kv_header(self) -> None:
        header = "amount=0.05; token=USDC; chain=base; recipient=0x1; facilitator=https://f.io"
        req = _parse_x_payment_header(header)
        assert req.amount == "0.05"
        assert req.token == "USDC"

    def test_model_dump(self) -> None:
        req = PaymentRequirement(amount="1", token="USDC", chain="base", recipient="0x1", facilitator="f")
        d = req.model_dump()
        assert d["amount"] == "1"


class TestX402Client:
    def test_check_x402_returns_requirement_on_402(self) -> None:
        config = X402Config()
        client = X402Client(config)
        exc = urllib.error.HTTPError(
            "url", 402, "Payment Required", {"x-payment": json.dumps({"amount": "0.01", "token": "USDC", "chain": "base", "recipient": "0x1", "facilitator": "f"})}, None  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            req = client.check_x402("https://example.com/resource")
        assert req is not None
        assert req.amount == "0.01"

    def test_check_x402_returns_none_on_200(self) -> None:
        config = X402Config()
        client = X402Client(config)
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            req = client.check_x402("https://example.com")
        assert req is None

    def test_check_x402_returns_none_on_network_error(self) -> None:
        config = X402Config()
        client = X402Client(config)
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail")):
            req = client.check_x402("https://example.com")
        assert req is None

    def test_negotiate_posts_to_facilitator(self) -> None:
        config = X402Config(facilitator_url="https://f.io", wallet_address="0xabc")
        client = X402Client(config)
        requirement = PaymentRequirement(
            amount="0.01", token="USDC", chain="base", recipient="0x1", facilitator=""
        )
        resp = MagicMock()
        resp.read.return_value = json.dumps({"token": "pay-tok-123"}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.negotiate(requirement)
        assert result["token"] == "pay-tok-123"

    def test_negotiate_no_facilitator_raises(self) -> None:
        config = X402Config()  # no facilitator
        client = X402Client(config)
        requirement = PaymentRequirement(amount="0.01", token="USDC", chain="base", recipient="0x1", facilitator="")
        try:
            client.negotiate(requirement)
            assert False, "Should have raised"
        except X402Error as exc:
            assert "facilitator" in str(exc).lower()


class TestX402Persistence:
    def test_save_and_load_round_trip(self) -> None:
        db = FakeDB()
        config = X402Config(wallet_address="0xabc", chain="ethereum", enabled=True)
        save_x402_config(config, db=db)
        loaded = load_x402_config(db=db)
        assert loaded.wallet_address == "0xabc"
        assert loaded.chain == "ethereum"
        assert loaded.enabled is True

    def test_load_empty_returns_defaults(self) -> None:
        db = FakeDB()
        config = load_x402_config(db=db)
        assert config.enabled is False
        assert config.wallet_address == ""


class TestX402FromProfile:
    def test_projection(self) -> None:
        class FakeProfile:
            x402_enabled = True
            metadata = {"wallet_address": "0xabc", "x402_chain": "base", "x402_facilitator": "https://f.io"}

        config = x402_config_from_profile(FakeProfile())
        assert config.enabled is True
        assert config.wallet_address == "0xabc"
        assert config.facilitator_url == "https://f.io"

    def test_projection_defaults(self) -> None:
        class FakeProfile:
            x402_enabled = False
            metadata: dict = {}  # type: ignore[assignment]

        config = x402_config_from_profile(FakeProfile())
        assert config.enabled is False
        assert config.wallet_address == ""
