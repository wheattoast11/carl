"""Public resolvers for the v0.17 handle-runtime chain.

Three ship-ready backends for :class:`~carl_core.vault.Vault`:

* :class:`EnvResolver` — reads OS environment variables (``env://VAR``)
* :class:`KeyringResolver` — OS keychain (``keyring://service/account``)
* :class:`FernetFileResolver` — Fernet-encrypted local file (``fernet-file://~/.carl/vault/name``)

Each resolver is a callable instance — ``resolver(ref) -> bytes`` — so users
register it via ``vault.register_resolver(kind, resolver)``. Optional deps
(``keyring``, ``cryptography``) are lazy-imported inside the resolver's
``__call__``, not at module load — importing this module is dependency-free.

Example::

    from carl_core.secrets import SecretRef, SecretVault
    from carl_studio.handles.resolvers import EnvResolver, KeyringResolver

    vault = SecretVault()
    vault.register_resolver("env", EnvResolver())
    vault.register_resolver("keychain", KeyringResolver())

    ref = vault.put_ref_only(SecretRef(kind="env", uri="env://GITHUB_TOKEN"))
    token = vault.resolve(ref, privileged=True)  # → bytes via EnvResolver

Users bring their own 1Password / HashiCorp Vault / AWS SM adapter by
writing a callable with the same shape and registering it. Proprietary
resolvers (hardware-attested, terminals-runtime) register at admin-gate
time via ``Vault.register_runtime_resolver`` — see ``admin.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carl_core.errors import CARLError
from carl_core.vault import VaultError


__all__ = [
    "ResolverError",
    "EnvResolver",
    "KeyringResolver",
    "FernetFileResolver",
    "parse_uri",
]


class ResolverError(CARLError):
    """Base for ``carl.resolver.*`` errors.

    Does NOT inherit from :class:`VaultError` because resolvers are
    user-facing + backend-agnostic; their error taxonomy is distinct from
    the vault's. The :class:`Vault.resolve` plumbing wraps resolver
    exceptions as ``<prefix>.resolver_failed`` when it catches a plain
    ``Exception`` — :class:`ResolverError` instances also get wrapped, but
    the cause chain preserves their original taxonomy for debugging.
    """

    code = "carl.resolver"


# ---------------------------------------------------------------------------
# URI parsing
# ---------------------------------------------------------------------------


def parse_uri(uri: str, expected_scheme: str) -> str:
    """Split ``<scheme>://<body>``; return body. Raise on scheme mismatch."""
    prefix = f"{expected_scheme}://"
    if not uri.startswith(prefix):
        raise ResolverError(
            f"uri scheme mismatch: expected {expected_scheme!r}, got {uri!r}",
            code=f"carl.resolver.{expected_scheme}.scheme_mismatch",
            context={"uri": uri, "expected": expected_scheme},
        )
    return uri[len(prefix):]


# ---------------------------------------------------------------------------
# EnvResolver — OS env var → bytes
# ---------------------------------------------------------------------------


@dataclass
class EnvResolver:
    """Resolves ``env://VAR_NAME`` handles to the env-var's UTF-8 bytes.

    Missing vars raise ``carl.resolver.env.not_found``. Empty-string vars
    return ``b""`` (an explicit choice: empty is a valid value).
    """

    def __call__(self, ref: Any) -> bytes:
        var = parse_uri(ref.uri, "env")
        if var not in os.environ:
            raise ResolverError(
                f"env var {var!r} is not set",
                code="carl.resolver.env.not_found",
                context={"var": var, "ref_id": str(ref.ref_id)},
            )
        return os.environ[var].encode("utf-8")


# ---------------------------------------------------------------------------
# KeyringResolver — OS keychain → bytes
# ---------------------------------------------------------------------------


@dataclass
class KeyringResolver:
    """Resolves ``keyring://service/account`` to the OS keychain entry.

    Requires the optional ``keyring`` package (in the ``[secrets]`` extra).
    Lazy-imported inside ``__call__`` so this module imports without
    requiring keyring at all.

    Missing entries raise ``carl.resolver.keyring.not_found``. Backend
    unavailable raises ``carl.resolver.keyring.backend_unavailable``.
    """

    def __call__(self, ref: Any) -> bytes:
        body = parse_uri(ref.uri, "keyring")
        if "/" not in body:
            raise ResolverError(
                f"keyring URI must be 'keyring://service/account', got {ref.uri!r}",
                code="carl.resolver.keyring.bad_uri",
                context={"uri": ref.uri},
            )
        service, account = body.split("/", 1)

        try:
            import keyring as _keyring  # noqa: PLC0415 — intentional lazy import
        except ImportError as exc:
            raise ResolverError(
                "keyring is required. Install: pip install 'carl-studio[secrets]'",
                code="carl.resolver.keyring.backend_unavailable",
                context={"missing": "keyring"},
                cause=exc,
            ) from exc

        value = _keyring.get_password(service, account)
        if value is None:
            raise ResolverError(
                f"keyring entry not found: service={service!r}, account={account!r}",
                code="carl.resolver.keyring.not_found",
                context={"service": service, "account": account, "ref_id": str(ref.ref_id)},
            )
        return value.encode("utf-8")


# ---------------------------------------------------------------------------
# FernetFileResolver — Fernet-encrypted local file → bytes
# ---------------------------------------------------------------------------


@dataclass
class FernetFileResolver:
    """Resolves ``fernet-file://~/.carl/vault/<name>`` to decrypted bytes.

    Requires the ``cryptography`` package (in ``[wallet]`` or ``[secrets]``
    extras — typically pre-installed). The master key lives at
    ``~/.carl/vault.key`` by default; override via the ``key_path``
    constructor arg.

    Vault layout (``~/.carl/vault/<name>``):
        * Each entry is a single file containing a Fernet token.
        * The master key is 32 bytes of url-safe base64 (``Fernet.generate_key()``).
        * Key file auto-created on first resolve; mode 0o600.
    """

    key_path: Path | None = None
    vault_dir: Path | None = None

    def __call__(self, ref: Any) -> bytes:
        body = parse_uri(ref.uri, "fernet-file")
        # Accept both absolute (~/x.enc or /x.enc) and vault-relative names.
        target = Path(body).expanduser()
        if not target.is_absolute():
            target = self._vault_dir() / target

        if not target.is_file():
            raise ResolverError(
                f"fernet-file entry not found: {target}",
                code="carl.resolver.fernet_file.not_found",
                context={"path": str(target), "ref_id": str(ref.ref_id)},
            )

        try:
            from cryptography.fernet import Fernet, InvalidToken  # noqa: PLC0415
        except ImportError as exc:
            raise ResolverError(
                "cryptography is required. Install: pip install 'carl-studio[wallet]'",
                code="carl.resolver.fernet_file.backend_unavailable",
                context={"missing": "cryptography"},
                cause=exc,
            ) from exc

        try:
            key = self._load_or_create_key()
            token = target.read_bytes()
            return Fernet(key).decrypt(token)
        except InvalidToken as exc:
            raise ResolverError(
                f"fernet-file decryption failed for {target} "
                "(wrong key or corrupted file)",
                code="carl.resolver.fernet_file.decrypt_failed",
                context={"path": str(target)},
                cause=exc,
            ) from exc

    def write(self, name: str, plaintext: bytes) -> Path:
        """Encrypt + write ``plaintext`` to ``<vault_dir>/<name>``.

        Companion to :meth:`__call__` for callers that want to stage a
        Fernet entry from code. Creates the vault dir if missing. Returns
        the target path. Mode 0o600.
        """
        try:
            from cryptography.fernet import Fernet  # noqa: PLC0415
        except ImportError as exc:
            raise ResolverError(
                "cryptography is required",
                code="carl.resolver.fernet_file.backend_unavailable",
                context={"missing": "cryptography"},
                cause=exc,
            ) from exc
        vault_dir = self._vault_dir()
        vault_dir.mkdir(parents=True, exist_ok=True)
        target = vault_dir / name
        key = self._load_or_create_key()
        token = Fernet(key).encrypt(plaintext)
        target.write_bytes(token)
        target.chmod(0o600)
        return target

    def _key_path(self) -> Path:
        return self.key_path or (Path.home() / ".carl" / "vault.key")

    def _vault_dir(self) -> Path:
        return self.vault_dir or (Path.home() / ".carl" / "vault")

    def _load_or_create_key(self) -> bytes:
        p = self._key_path()
        if p.exists():
            return p.read_bytes()
        try:
            from cryptography.fernet import Fernet  # noqa: PLC0415
        except ImportError as exc:
            raise ResolverError(
                "cryptography is required",
                code="carl.resolver.fernet_file.backend_unavailable",
                context={"missing": "cryptography"},
                cause=exc,
            ) from exc
        p.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        p.write_bytes(key)
        p.chmod(0o600)
        return key


# Keep VaultError import live for consumers that catch both ResolverError
# and VaultError in a single `except` — trivial, pyright's unused-import
# check fires without a sentinel.
_ = VaultError
