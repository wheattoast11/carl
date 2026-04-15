"""x402 HTTP payment rail client for machine-to-machine micropayments.

The facilitator handles all on-chain work. This client sends HTTP requests
with x402 headers. No web3.py dependency — stdlib urllib only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

_X402_CONFIG_KEY = "x402_config"


class X402Error(Exception):
    """Raised when x402 payment operations fail."""


class X402Config(BaseModel):
    """x402 payment rail configuration."""

    wallet_address: str = ""
    chain: str = "base"
    facilitator_url: str = ""
    payment_token: str = "USDC"
    auto_approve_below: float = 0.0
    enabled: bool = False


class PaymentRequirement(BaseModel):
    """Parsed from a 402 response's x-payment header."""

    amount: str = "0"
    token: str = ""
    chain: str = ""
    recipient: str = ""
    facilitator: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


def _parse_x_payment_header(header: str) -> PaymentRequirement:
    """Parse an x-payment response header into a PaymentRequirement."""
    try:
        data = json.loads(header)
        return PaymentRequirement.model_validate(data)
    except (json.JSONDecodeError, Exception):
        # Fallback: key=value pairs separated by semicolons
        parts: dict[str, str] = {}
        for segment in header.split(";"):
            segment = segment.strip()
            if "=" in segment:
                k, v = segment.split("=", 1)
                parts[k.strip()] = v.strip()
        return PaymentRequirement(
            amount=parts.get("amount", "0"),
            token=parts.get("token", ""),
            chain=parts.get("chain", ""),
            recipient=parts.get("recipient", ""),
            facilitator=parts.get("facilitator", ""),
        )


class X402Client:
    """Lightweight x402 payment rail client using stdlib urllib."""

    def __init__(self, config: X402Config) -> None:
        self._config = config

    def check_x402(self, url: str, timeout: int = 10) -> PaymentRequirement | None:
        """HEAD request; parse x-payment header from 402 response.

        Returns None if the URL does not support x402.
        """
        req = urllib.request.Request(url, method="HEAD")
        try:
            urllib.request.urlopen(req, timeout=timeout)
            return None  # 200 OK — no payment required
        except urllib.error.HTTPError as exc:
            if exc.code != 402:
                return None
            header = exc.headers.get("x-payment", "")
            if not header:
                return None
            return _parse_x_payment_header(header)
        except urllib.error.URLError:
            return None

    def negotiate(
        self, requirement: PaymentRequirement, timeout: int = 10
    ) -> dict[str, Any]:
        """POST to facilitator to get a payment authorization token."""
        facilitator = requirement.facilitator or self._config.facilitator_url
        if not facilitator:
            raise X402Error("No facilitator URL configured or in payment requirement.")

        body = json.dumps({
            "amount": requirement.amount,
            "token": requirement.token or self._config.payment_token,
            "chain": requirement.chain or self._config.chain,
            "recipient": requirement.recipient,
            "payer": self._config.wallet_address,
        }).encode()

        req = urllib.request.Request(
            f"{facilitator}/negotiate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode(errors="replace") if exc.fp else str(exc)
            raise X402Error(f"Negotiation failed ({exc.code}): {msg}") from exc
        except urllib.error.URLError as exc:
            raise X402Error(f"Network error: {exc.reason}") from exc

    def execute(
        self, url: str, payment_token: str, timeout: int = 10
    ) -> bytes:
        """Replay the original request with x-payment-token header."""
        req = urllib.request.Request(
            url,
            headers={"X-Payment-Token": payment_token},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode(errors="replace") if exc.fp else str(exc)
            raise X402Error(f"Payment execution failed ({exc.code}): {msg}") from exc
        except urllib.error.URLError as exc:
            raise X402Error(f"Network error: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# CampProfile projection and persistence
# ---------------------------------------------------------------------------


def x402_config_from_profile(profile: Any) -> X402Config:
    """Project x402 flags from a CampProfile."""
    return X402Config(
        enabled=getattr(profile, "x402_enabled", False),
        wallet_address=getattr(profile, "metadata", {}).get("wallet_address", ""),
        chain=getattr(profile, "metadata", {}).get("x402_chain", "base"),
        facilitator_url=getattr(profile, "metadata", {}).get("x402_facilitator", ""),
    )


def load_x402_config(db: Any | None = None) -> X402Config:
    """Load x402 config from LocalDB."""
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()
    raw = db.get_config(_X402_CONFIG_KEY)
    if not raw:
        return X402Config()
    try:
        return X402Config.model_validate_json(raw)
    except Exception:
        return X402Config()


def save_x402_config(config: X402Config, db: Any | None = None) -> None:
    """Persist x402 config to LocalDB."""
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()
    db.set_config(_X402_CONFIG_KEY, config.model_dump_json())
