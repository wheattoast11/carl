"""Tests for wallet integration module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from carl_studio.wallet import WalletInfo, check_agentkit_available, create_wallet, get_wallet_info

wallet_runner = CliRunner()


class TestWalletInfo:
    """WalletInfo dataclass."""

    def test_defaults(self) -> None:
        w = WalletInfo(address="0xABC")
        assert w.address == "0xABC"
        assert w.network == "base"
        assert w.provider == "unknown"
        assert w.balance is None
        assert w.metadata == {}

    def test_custom_fields(self) -> None:
        w = WalletInfo(
            address="0xDEF",
            network="ethereum",
            provider="agentkit",
            balance="1.5 ETH",
            metadata={"created": "2026-01-01"},
        )
        assert w.address == "0xDEF"
        assert w.network == "ethereum"
        assert w.provider == "agentkit"
        assert w.balance == "1.5 ETH"
        assert w.metadata == {"created": "2026-01-01"}

    def test_metadata_isolation(self) -> None:
        """Each instance gets its own metadata dict."""
        a = WalletInfo(address="0x1")
        b = WalletInfo(address="0x2")
        a.metadata["key"] = "val"
        assert "key" not in b.metadata


class TestCheckAgentkitAvailable:
    """check_agentkit_available() without real agentkit."""

    def test_returns_false_when_not_installed(self) -> None:
        assert check_agentkit_available() is False

    @patch.dict("sys.modules", {"coinbase_agentkit": MagicMock()})
    def test_returns_true_when_installed(self) -> None:
        assert check_agentkit_available() is True


class TestGetWalletInfo:
    """get_wallet_info() — DB/network calls mocked.

    LocalDB, fetch_camp_profile, load_x402_config are lazy-imported inside
    get_wallet_info, so we patch at their source modules.
    """

    def test_returns_none_when_db_unavailable(self) -> None:
        with patch("carl_studio.db.LocalDB", side_effect=Exception("no db")):
            result = get_wallet_info()
        assert result is None

    def test_returns_none_when_no_jwt(self) -> None:
        mock_db = MagicMock()
        mock_db.get_auth.return_value = None
        with patch("carl_studio.db.LocalDB", return_value=mock_db):
            result = get_wallet_info()
        assert result is None

    def test_returns_wallet_from_camp_profile(self) -> None:
        mock_db = MagicMock()
        mock_db.get_auth.return_value = "fake-jwt"
        mock_db.get_config.return_value = "https://example.supabase.co"

        mock_profile = MagicMock()
        mock_profile.metadata = {"wallet_address": "0xCAMP", "x402_chain": "base"}

        with (
            patch("carl_studio.db.LocalDB", return_value=mock_db),
            patch("carl_studio.camp.fetch_camp_profile", return_value=mock_profile),
        ):
            result = get_wallet_info()

        assert result is not None
        assert result.address == "0xCAMP"
        assert result.provider == "camp"
        assert result.network == "base"

    def test_falls_back_to_x402_config(self) -> None:
        mock_db = MagicMock()
        mock_db.get_auth.return_value = "fake-jwt"
        mock_db.get_config.return_value = None

        mock_x402 = MagicMock()
        mock_x402.wallet_address = "0xLOCAL"
        mock_x402.chain = "base"

        with (
            patch("carl_studio.db.LocalDB", return_value=mock_db),
            patch(
                "carl_studio.camp.fetch_camp_profile",
                side_effect=Exception("network down"),
            ),
            patch("carl_studio.x402.load_x402_config", return_value=mock_x402),
        ):
            result = get_wallet_info()

        assert result is not None
        assert result.address == "0xLOCAL"
        assert result.provider == "x402-local"


class TestCreateWallet:
    """create_wallet() — agentkit not installed."""

    def test_returns_none_when_agentkit_missing(self) -> None:
        result = create_wallet()
        assert result is None

    @patch.dict("sys.modules", {"coinbase_agentkit": MagicMock()})
    def test_returns_wallet_when_agentkit_available(self) -> None:
        import sys

        mock_agentkit = sys.modules["coinbase_agentkit"]
        mock_wallet = MagicMock()
        mock_wallet.default_address.address_id = "0xNEW"
        mock_kit = MagicMock()
        mock_kit.wallet = mock_wallet
        mock_agentkit.AgentKit.return_value = mock_kit

        result = create_wallet()
        assert result is not None
        assert result.address == "0xNEW"
        assert result.provider == "agentkit"
        assert result.network == "base"

    def test_returns_none_on_runtime_error(self) -> None:
        """Even with agentkit importable, runtime errors return None."""
        mock_mod = MagicMock()
        mock_mod.AgentKit.side_effect = RuntimeError("API key missing")

        with patch.dict("sys.modules", {"coinbase_agentkit": mock_mod}):
            result = create_wallet()
        assert result is None


# ---------------------------------------------------------------------------
# Wallet CLI tests (wallet_app invoked directly via CliRunner)
# ---------------------------------------------------------------------------


class TestWalletCLI:
    """CLI tests for carl camp wallet subcommands."""

    def test_wallet_status_no_wallet(self) -> None:
        """wallet status shows info message when no wallet configured."""
        from carl_studio.cli.wallet import wallet_app

        with patch("carl_studio.wallet.get_wallet_info", return_value=None):
            result = wallet_runner.invoke(wallet_app, ["status"])
        assert result.exit_code == 0
        assert "wallet" in result.output.lower() or "No wallet" in result.output

    def test_wallet_status_with_wallet(self) -> None:
        """wallet status displays address when wallet exists."""
        from carl_studio.cli.wallet import wallet_app

        info = WalletInfo(address="0xTEST", network="base", provider="camp")
        with patch("carl_studio.wallet.get_wallet_info", return_value=info):
            result = wallet_runner.invoke(wallet_app, ["status"])
        assert result.exit_code == 0
        assert "0xTEST" in result.output

    def test_wallet_status_json_no_wallet(self) -> None:
        """wallet status --json outputs null fields when no wallet."""
        import json

        from carl_studio.cli.wallet import wallet_app

        with patch("carl_studio.wallet.get_wallet_info", return_value=None):
            result = wallet_runner.invoke(wallet_app, ["status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["address"] is None

    def test_wallet_status_json_with_wallet(self) -> None:
        """wallet status --json outputs wallet data."""
        import json

        from carl_studio.cli.wallet import wallet_app

        info = WalletInfo(address="0xJSON", network="base", provider="agentkit", balance="0.5 ETH")
        with patch("carl_studio.wallet.get_wallet_info", return_value=info):
            result = wallet_runner.invoke(wallet_app, ["status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["address"] == "0xJSON"
        assert data["balance"] == "0.5 ETH"

    def test_wallet_create_fails_without_agentkit(self) -> None:
        """wallet create exits 1 when create_wallet returns None."""
        from carl_studio.cli.wallet import wallet_app

        with patch("carl_studio.wallet.create_wallet", return_value=None):
            result = wallet_runner.invoke(wallet_app, ["create"])
        assert result.exit_code == 1

    def test_wallet_create_success(self) -> None:
        """wallet create exits 0 and shows address on success."""
        from carl_studio.cli.wallet import wallet_app

        info = WalletInfo(address="0xNEW", network="base", provider="agentkit")
        with patch("carl_studio.wallet.create_wallet", return_value=info):
            result = wallet_runner.invoke(wallet_app, ["create"])
        assert result.exit_code == 0
        assert "0xNEW" in result.output
