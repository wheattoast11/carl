"""Carl-studio's zero-knowledge secrets toolkit (Stage B).

Platform-layer wrappers over :mod:`carl_core.secrets` primitives. Where
carl-core defines pure, dependency-free primitives (``SecretRef``,
``SecretVault``, ``seal``/``unseal``), this package integrates them
with the OS keychain, OS clipboard, and cryptographic random sources
so the agent can orchestrate secret transfer end-to-end without
reading any value.

Surface
-------

* :class:`KeychainBackend` — ``keyring`` wrapper with name-only
  semantics. The agent passes ``(service, account)`` tuples; only the
  privileged call path reaches the stored value.
* :class:`ClipboardBridge` — ``pyperclip`` wrapper with TTL auto-wipe
  and :class:`~carl_core.interaction.InteractionChain` audit.
* :class:`CryptoRandomMinter` — fresh-secret factory. Mints hex /
  base64 / UUID / Curve25519-keypair values directly into a
  :class:`~carl_core.secrets.SecretVault` without surfacing them.

Tool-dispatcher integration (registered via :func:`register_tools`):

* ``mint_secret(kind, length)`` → handle
* ``copy_to_clipboard(ref_id, ttl_s)`` → {fingerprint, expires_at}
* ``revoke_secret(ref_id)`` → bool
* ``hash_value(ref_id, algorithm)`` → {fingerprint, algorithm}
* ``list_secrets()`` → metadata-only list

All tools operate on ``ref_id`` (UUID) only. No raw values cross the
agent's context boundary.

Install: ``pip install 'carl-studio[secrets]'``
"""

from __future__ import annotations

from .clipboard import ClipboardBridge, ClipboardBridgeError
from .keychain import KeychainBackend, KeychainError
from .minter import CryptoRandomMinter
from .toolkit import SecretsToolkit


__all__ = [
    "ClipboardBridge",
    "ClipboardBridgeError",
    "CryptoRandomMinter",
    "KeychainBackend",
    "KeychainError",
    "SecretsToolkit",
]
