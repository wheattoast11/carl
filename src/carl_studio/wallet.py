"""Agent wallet integration via coinbase-agentkit.

Optional -- falls back gracefully when agentkit is not installed.
The wallet address from CampProfile is used for x402 payments.

Install wallet support::

    pip install 'carl-studio[wallet]'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WalletInfo:
    """Agent wallet state."""

    address: str
    network: str = "base"
    provider: str = "unknown"
    balance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def get_wallet_info() -> WalletInfo | None:
    """Get wallet info from camp profile or local config.

    Checks the camp profile first (wallet_address in metadata),
    then falls back to locally stored x402 config.
    Returns None if no wallet is configured.
    """
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
    except Exception:
        return None

    try:
        jwt = db.get_auth("jwt")
        if not jwt:
            return None

        # Try camp profile for wallet address
        try:
            from carl_studio.camp import (
                DEFAULT_CARL_CAMP_SUPABASE_URL,
                fetch_camp_profile,
            )

            supabase_url = db.get_config("supabase_url") or DEFAULT_CARL_CAMP_SUPABASE_URL
            profile = fetch_camp_profile(jwt, supabase_url)
            wallet_addr = profile.metadata.get("wallet_address", "")
            if wallet_addr:
                return WalletInfo(
                    address=wallet_addr,
                    network=profile.metadata.get("x402_chain", "base"),
                    provider="camp",
                )
        except Exception:
            pass

        # Fall back to local x402 config
        try:
            from carl_studio.x402 import load_x402_config

            config = load_x402_config(db=db)
            if config.wallet_address:
                return WalletInfo(
                    address=config.wallet_address,
                    network=config.chain,
                    provider="x402-local",
                )
        except Exception:
            pass

    except Exception as exc:
        logger.debug("get_wallet_info failed: %s", exc)

    return None


def check_agentkit_available() -> bool:
    """Check if coinbase-agentkit is installed."""
    try:
        import coinbase_agentkit  # noqa: F401

        return True
    except ImportError:
        return False


def create_wallet() -> WalletInfo | None:
    """Create a new agent wallet via coinbase-agentkit.

    Requires: ``pip install 'carl-studio[wallet]'``

    Returns a WalletInfo on success, None on failure.
    """
    try:
        from coinbase_agentkit import AgentKit, AgentKitConfig  # type: ignore[import-untyped]

        config = AgentKitConfig(network_id="base-mainnet")
        kit = AgentKit(config)
        wallet = kit.wallet

        return WalletInfo(
            address=wallet.default_address.address_id,
            network="base",
            provider="agentkit",
        )
    except ImportError:
        logger.info(
            "coinbase-agentkit not installed. Install: pip install 'carl-studio[wallet]'"
        )
        return None
    except Exception as e:
        logger.warning("Wallet creation failed: %s", e)
        return None
