"""Tests for wallet integration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from carl_studio.wallet import (
    WalletInfo,
    check_agentkit_available,
    check_encryption_available,
    create_wallet,
    get_backend_name,
    get_wallet_info,
)

wallet_runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_wallet_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Point WalletStore at an isolated temp home and strip env/keyring."""
    home = tmp_path / "carl_home"
    home.mkdir()
    monkeypatch.setattr(
        "carl_studio.wallet_store.Path.home",
        classmethod(lambda cls: tmp_path),  # type: ignore[arg-type]
    )
    # Ensure default path becomes tmp_path/.carl
    (tmp_path / ".carl").mkdir(exist_ok=True)
    monkeypatch.delenv("CARL_WALLET_PASSPHRASE", raising=False)
    monkeypatch.setattr(
        "carl_studio.wallet_store._keyring_get_password",
        lambda service, account: None,
    )
    monkeypatch.setattr(
        "carl_studio.wallet_store._keyring_set_password",
        lambda service, account, value: False,
    )
    return tmp_path / ".carl"


# ---------------------------------------------------------------------------
# WalletInfo
# ---------------------------------------------------------------------------


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

    def test_redacted_dict_has_no_secrets(self) -> None:
        w = WalletInfo(address="0xABC", balance="1.0")
        data = w.redacted_dict()
        assert data == {
            "address": "0xABC",
            "network": "base",
            "provider": "unknown",
            "balance": "1.0",
            "metadata": {},
        }
        # A mutation on the returned copy must not propagate to the source.
        data["metadata"]["x"] = "y"
        assert w.metadata == {}


# ---------------------------------------------------------------------------
# Feature probes
# ---------------------------------------------------------------------------


class TestCheckAgentkitAvailable:
    """check_agentkit_available() without real agentkit."""

    def test_returns_false_when_not_installed(self) -> None:
        assert check_agentkit_available() is False

    @patch.dict("sys.modules", {"coinbase_agentkit": MagicMock()})
    def test_returns_true_when_installed(self) -> None:
        assert check_agentkit_available() is True


class TestCheckEncryptionAvailable:
    def test_true_when_cryptography_installed(self) -> None:
        # cryptography is a test-time dep via [wallet] install.
        assert check_encryption_available() is True


class TestBackendName:
    def test_locked_by_default(self, clean_wallet_env: Path) -> None:
        assert get_backend_name() in {"locked", "keyring", "unsupported"}

    def test_fernet_when_env_set(
        self, clean_wallet_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARL_WALLET_PASSPHRASE", "env-pass-1234")
        name = get_backend_name()
        assert name == "fernet"


# ---------------------------------------------------------------------------
# get_wallet_info chain
# ---------------------------------------------------------------------------


class TestGetWalletInfo:
    """LocalDB, fetch_camp_profile, load_x402_config are lazy-imported inside
    get_wallet_info, so we patch at their source modules."""

    def test_returns_none_when_db_unavailable(self, clean_wallet_env: Path) -> None:
        with patch("carl_studio.db.LocalDB", side_effect=Exception("no db")):
            result = get_wallet_info()
        assert result is None

    def test_returns_none_when_no_jwt(self, clean_wallet_env: Path) -> None:
        mock_db = MagicMock()
        mock_db.get_auth.return_value = None
        with patch("carl_studio.db.LocalDB", return_value=mock_db):
            result = get_wallet_info()
        assert result is None

    def test_returns_wallet_from_camp_profile(self, clean_wallet_env: Path) -> None:
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

    def test_falls_back_to_x402_config(self, clean_wallet_env: Path) -> None:
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

    def test_wallet_store_takes_priority(
        self, clean_wallet_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a WalletStore exists and has an address, it wins."""
        from carl_studio.wallet import WALLET_ADDRESS
        from carl_studio.wallet_store import WalletStore

        monkeypatch.setenv("CARL_WALLET_PASSPHRASE", "env-pass-1234")
        s = WalletStore()
        s.unlock()
        s.put(WALLET_ADDRESS, "0xSTORED")

        # Even if DB has other info, the store should be used first.
        mock_db = MagicMock()
        mock_db.get_auth.return_value = "jwt"
        with patch("carl_studio.db.LocalDB", return_value=mock_db):
            info = get_wallet_info()
        assert info is not None
        assert info.address == "0xSTORED"
        assert info.provider == "carl-store"

    def test_locked_store_does_not_block_fallbacks(
        self, clean_wallet_env: Path
    ) -> None:
        """A locked store must not prevent falling through to camp profile."""
        mock_db = MagicMock()
        mock_db.get_auth.return_value = "jwt"
        mock_db.get_config.return_value = None

        mock_profile = MagicMock()
        mock_profile.metadata = {"wallet_address": "0xCAMP2"}

        with (
            patch("carl_studio.db.LocalDB", return_value=mock_db),
            patch("carl_studio.camp.fetch_camp_profile", return_value=mock_profile),
        ):
            info = get_wallet_info()
        assert info is not None
        assert info.address == "0xCAMP2"


# ---------------------------------------------------------------------------
# create_wallet
# ---------------------------------------------------------------------------


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

    def test_persists_address_when_passphrase_given(
        self, clean_wallet_env: Path
    ) -> None:
        """Providing a passphrase encrypts and persists wallet info."""
        from carl_studio.wallet import WALLET_ADDRESS
        from carl_studio.wallet_store import WalletStore

        mock_mod = MagicMock()
        mock_wallet = MagicMock()
        mock_wallet.default_address.address_id = "0xPERSIST"
        mock_wallet.export = MagicMock(return_value="0xPRIVATE")
        mock_kit = MagicMock()
        mock_kit.wallet = mock_wallet
        mock_mod.AgentKit.return_value = mock_kit

        with patch.dict("sys.modules", {"coinbase_agentkit": mock_mod}):
            result = create_wallet(passphrase="persist-pp-1234")

        assert result is not None
        assert result.address == "0xPERSIST"

        # Reopen and verify it landed encrypted-at-rest.
        store = WalletStore()
        store.unlock(passphrase="persist-pp-1234")
        assert store.get(WALLET_ADDRESS) == "0xPERSIST"
        # Private key should have been stored too.
        assert store.get("private_key") == "0xPRIVATE"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestWalletCLI:
    """CLI tests for carl camp wallet subcommands."""

    def test_wallet_status_no_wallet(self, clean_wallet_env: Path) -> None:
        """wallet status shows info message when no wallet configured."""
        from carl_studio.cli.wallet import wallet_app

        with patch("carl_studio.wallet.get_wallet_info", return_value=None):
            result = wallet_runner.invoke(wallet_app, ["status"])
        assert result.exit_code == 0
        assert "wallet" in result.output.lower() or "No wallet" in result.output

    def test_wallet_status_with_wallet(self, clean_wallet_env: Path) -> None:
        """wallet status displays address when wallet exists."""
        from carl_studio.cli.wallet import wallet_app

        info = WalletInfo(address="0xTEST", network="base", provider="camp")
        with patch("carl_studio.wallet.get_wallet_info", return_value=info):
            result = wallet_runner.invoke(wallet_app, ["status"])
        assert result.exit_code == 0
        assert "0xTEST" in result.output

    def test_wallet_status_json_no_wallet(self, clean_wallet_env: Path) -> None:
        """wallet status --json outputs null fields when no wallet."""
        import json

        from carl_studio.cli.wallet import wallet_app

        with patch("carl_studio.wallet.get_wallet_info", return_value=None):
            result = wallet_runner.invoke(wallet_app, ["status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["address"] is None
        assert "backend" in data
        assert "locked" in data

    def test_wallet_status_json_with_wallet(self, clean_wallet_env: Path) -> None:
        """wallet status --json outputs wallet data."""
        import json

        from carl_studio.cli.wallet import wallet_app

        info = WalletInfo(
            address="0xJSON", network="base", provider="agentkit", balance="0.5 ETH"
        )
        with patch("carl_studio.wallet.get_wallet_info", return_value=info):
            result = wallet_runner.invoke(wallet_app, ["status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["address"] == "0xJSON"
        assert data["balance"] == "0.5 ETH"
        assert "backend" in data

    def test_wallet_create_fails_without_agentkit(self) -> None:
        """wallet create exits 1 when create_wallet returns None."""
        from carl_studio.cli.wallet import wallet_app

        with patch("carl_studio.wallet.create_wallet", return_value=None):
            result = wallet_runner.invoke(
                wallet_app, ["create", "--no-passphrase"]
            )
        assert result.exit_code == 1

    def test_wallet_create_success(self) -> None:
        """wallet create exits 0 and shows address on success."""
        from carl_studio.cli.wallet import wallet_app

        info = WalletInfo(address="0xNEW", network="base", provider="agentkit")
        with patch("carl_studio.wallet.create_wallet", return_value=info):
            result = wallet_runner.invoke(
                wallet_app, ["create", "--no-passphrase"]
            )
        assert result.exit_code == 0
        assert "0xNEW" in result.output

    def test_wallet_unlock_success(self, clean_wallet_env: Path) -> None:
        """wallet unlock succeeds with the right passphrase."""
        from carl_studio.cli.wallet import wallet_app
        from carl_studio.wallet_store import WalletStore

        # Seed an encrypted envelope.
        s = WalletStore()
        s.unlock(passphrase="seed-pp-12345")
        s.put("private_key", "0xXYZ")

        with patch("getpass.getpass", return_value="seed-pp-12345"):
            result = wallet_runner.invoke(wallet_app, ["unlock"])
        assert result.exit_code == 0
        assert "unlocked" in result.output.lower()

    def test_wallet_unlock_wrong_passphrase(self, clean_wallet_env: Path) -> None:
        """wallet unlock exits non-zero on wrong passphrase."""
        from carl_studio.cli.wallet import wallet_app
        from carl_studio.wallet_store import WalletStore

        s = WalletStore()
        s.unlock(passphrase="right-pp-1234")
        s.put("k", "v")

        with patch("getpass.getpass", return_value="wrong-pp-1234"):
            result = wallet_runner.invoke(wallet_app, ["unlock"])
        assert result.exit_code == 1

    def test_wallet_rotate_success(self, clean_wallet_env: Path) -> None:
        """wallet rotate prompts for old + new + confirm."""
        from carl_studio.cli.wallet import wallet_app
        from carl_studio.wallet_store import WalletStore

        s = WalletStore()
        s.unlock(passphrase="old-pp-12345")
        s.put("k", "v")

        prompts = iter(["old-pp-12345", "new-pp-12345", "new-pp-12345"])

        with patch("getpass.getpass", side_effect=lambda *_a, **_kw: next(prompts)):
            result = wallet_runner.invoke(wallet_app, ["rotate"])
        assert result.exit_code == 0

        # Verify new works.
        s2 = WalletStore()
        s2.unlock(passphrase="new-pp-12345")
        assert s2.get("k") == "v"

    def test_wallet_rotate_mismatch_confirm(self, clean_wallet_env: Path) -> None:
        from carl_studio.cli.wallet import wallet_app
        from carl_studio.wallet_store import WalletStore

        s = WalletStore()
        s.unlock(passphrase="old-pp-12345")

        prompts = iter(["old-pp-12345", "new-pp-12345", "different-pp-123"])
        with patch("getpass.getpass", side_effect=lambda *_a, **_kw: next(prompts)):
            result = wallet_runner.invoke(wallet_app, ["rotate"])
        assert result.exit_code == 1
