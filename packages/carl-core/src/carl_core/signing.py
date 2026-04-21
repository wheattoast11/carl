"""Public software-tier signing for EML trees.

HMAC-SHA256 over the inner tree bytes, keyed on a per-user secret.
This is the platform-verifiable tier: the carl.camp service holds
each user's ``user_secret`` and verifies on insert. No hardware
fingerprint is mixed in; for hardware-bound attestations see
``terminals_runtime.eml.sign_impl`` (private, BUSL).

Layer boundary:
  - This module is MIT, stdlib-only (hmac, hashlib). Platform-safe.
  - The private module at ``terminals_runtime.eml.sign_impl`` is
    strictly additive on top (hw fingerprint XOR user secret), and
    platform MUST NOT import it.

See ``docs/eml_signing_protocol.md`` §2 for the full contract.
"""
from __future__ import annotations

import hashlib
import hmac

from carl_core.errors import ValidationError

SIG_LEN: int = 32
"""Length of an HMAC-SHA256 signature in bytes. Protocol-stable."""

MIN_SECRET_LEN: int = 16
"""Minimum acceptable user_secret length. 16 bytes = 128 bits entropy floor."""


def _validate_secret(user_secret: bytes | bytearray) -> None:
    if len(user_secret) < MIN_SECRET_LEN:
        raise ValidationError(
            f"user_secret must be at least {MIN_SECRET_LEN} bytes",
            code="carl.eml.domain_error",
            context={"len": len(user_secret), "min": MIN_SECRET_LEN},
        )


def sign_tree_software(tree_bytes: bytes, user_secret: bytes | bytearray) -> bytes:
    """Return HMAC-SHA256(user_secret, tree_bytes).

    ``tree_bytes`` MUST be the output of ``EMLTree.to_bytes()`` — the
    inner canonical layout from ``docs/eml_signing_protocol.md`` §1.1,
    NOT the envelope layout from §1.2.
    """
    _validate_secret(user_secret)
    return hmac.new(bytes(user_secret), tree_bytes, hashlib.sha256).digest()


def verify_software_signature(
    tree_bytes: bytes,
    sig: bytes,
    user_secret: bytes | bytearray,
) -> bool:
    """Constant-time HMAC check. True iff ``sig`` was produced by
    :func:`sign_tree_software` on the same ``tree_bytes`` with the same
    ``user_secret``.

    Returns False (does NOT raise) on length mismatch or bad secret —
    the caller decides how to surface attestation failures to users.
    """
    if len(sig) != SIG_LEN:
        return False
    try:
        expected = sign_tree_software(tree_bytes, user_secret)
    except ValidationError:
        return False
    return hmac.compare_digest(expected, sig)


def sign_platform_countersig(
    content_hash_hex: str,
    purchase_tx_id: str,
    buyer_user_id: str,
    timestamp_ns: int,
    platform_secret: bytes | bytearray,
) -> bytes:
    """Platform-side countersignature for purchase delivery.

    See ``docs/eml_signing_protocol.md`` §4.2 for the payload layout.
    """
    _validate_secret(platform_secret)
    import struct

    payload = (
        b"carl-platform-countersig-v1|"
        + content_hash_hex.encode("ascii")
        + b"|"
        + purchase_tx_id.encode("ascii")
        + b"|"
        + buyer_user_id.encode("ascii")
        + b"|"
        + struct.pack("<q", int(timestamp_ns))
    )
    return hmac.new(bytes(platform_secret), payload, hashlib.sha256).digest()


def verify_platform_countersig(
    content_hash_hex: str,
    purchase_tx_id: str,
    buyer_user_id: str,
    timestamp_ns: int,
    platform_secret: bytes | bytearray,
    sig: bytes,
) -> bool:
    """Buyer-side verification of a platform countersignature."""
    if len(sig) != SIG_LEN:
        return False
    try:
        expected = sign_platform_countersig(
            content_hash_hex,
            purchase_tx_id,
            buyer_user_id,
            timestamp_ns,
            platform_secret,
        )
    except ValidationError:
        return False
    return hmac.compare_digest(expected, sig)


__all__ = [
    "SIG_LEN",
    "MIN_SECRET_LEN",
    "sign_tree_software",
    "verify_software_signature",
    "sign_platform_countersig",
    "verify_platform_countersig",
]
