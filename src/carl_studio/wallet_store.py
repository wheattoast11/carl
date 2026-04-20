"""Encrypted wallet storage with keyring fallback.

Stores wallet secrets at ``~/.carl/wallet.enc`` as a Fernet-encrypted envelope
keyed off a user-supplied passphrase (or an OS keychain entry when the
``keyring`` library is installed).

Policy:
- Never log the passphrase or derived key.
- File mode is 0o600 on POSIX.
- Plaintext fallback requires an explicit opt-in AND no other material found.
- The ``cryptography`` and ``keyring`` libraries are *lazy-imported*; callers
  who touch secrets simply receive a :class:`WalletLocked` with a pip hint when
  the ``wallet`` extra is not installed.

Install::

    pip install 'carl-studio[wallet]'
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from carl_core.errors import CARLError, CredentialError, ValidationError

from carl_studio.settings import carl_home

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from carl_studio.config_registry import ConfigRegistry
    from carl_studio.db import LocalDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — envelope format + KDF parameters
# ---------------------------------------------------------------------------

#: On-disk envelope version. Bump on any layout change.
ENVELOPE_VERSION = 1

#: OWASP-recommended floor for PBKDF2-HMAC-SHA256 (as of 2026).
PBKDF2_ITERATIONS = 600_000

#: Salt length in bytes. 16 is standard for PBKDF2 salts.
SALT_LENGTH = 16

#: Default storage path components.
DEFAULT_HOME_SUBDIR = ".carl"
ENCRYPTED_FILENAME = "wallet.enc"
PLAINTEXT_FILENAME = "wallet.json"

#: Keyring service/account names used when OS keychain fallback is active.
KEYRING_SERVICE = "carl-studio"
KEYRING_ACCOUNT = "wallet-master-key"

#: Install hint surfaced when ``cryptography`` is not present.
INSTALL_HINT = "pip install 'carl-studio[wallet]'"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WalletLocked(CARLError):
    """Raised when the wallet store is not unlocked or the passphrase is wrong."""

    code = "carl.wallet.locked"


class WalletCorrupted(CARLError):
    """Raised when the on-disk envelope fails integrity checks."""

    code = "carl.wallet.corrupted"


# ---------------------------------------------------------------------------
# Non-secret metadata — persisted via ConfigRegistry
# ---------------------------------------------------------------------------


class WalletMetadata(BaseModel):
    """Non-secret, durable metadata about the on-disk wallet envelope.

    This model is intentionally *narrow*: it covers only the
    parameters needed to describe where and how the encrypted payload
    is stored. No passphrase, derived-key, or ciphertext material ever
    flows through here — those stay on the filesystem encryption path.

    Stored under ``carl.wallet.walletmetadata`` in
    :class:`~carl_studio.db.LocalDB` via :class:`ConfigRegistry`.
    """

    #: Filename (not full path) of the encrypted envelope on disk.
    encrypted_filename: str = Field(default=ENCRYPTED_FILENAME)
    #: PBKDF2 iteration count used to derive the Fernet key. Mirrors
    #: :data:`PBKDF2_ITERATIONS`; persisted so a future rotation can
    #: detect the originally-used value.
    kdf_iterations: int = Field(default=PBKDF2_ITERATIONS, ge=1)
    #: Envelope schema version — matches :data:`ENVELOPE_VERSION`.
    envelope_version: int = Field(default=ENVELOPE_VERSION, ge=1)
    #: Which backend the wallet is currently recorded as using. One of
    #: ``"fernet"``, ``"keyring"``, ``"plaintext"``, or ``"locked"``.
    backend: str = Field(default="locked")


def wallet_metadata_registry(db: LocalDB) -> ConfigRegistry[WalletMetadata]:
    """Return a typed :class:`ConfigRegistry` for :class:`WalletMetadata`.

    Thin helper so callers don't repeat the namespace literal. Reads
    and writes route through the shared ``LocalDB`` — never through
    the filesystem encryption path.
    """
    return db.config_registry(WalletMetadata, namespace="carl.wallet")


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------


def _load_cryptography() -> tuple[Any, Any, Any, Any]:
    """Lazy-import ``cryptography`` primitives. Raises :class:`WalletLocked` on miss."""
    try:
        from cryptography.fernet import Fernet, InvalidToken
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as exc:  # pragma: no cover - exercised via integration
        raise WalletLocked(
            "Wallet encryption requires the 'cryptography' package.",
            context={"hint": INSTALL_HINT},
            cause=exc,
        ) from exc
    return Fernet, InvalidToken, hashes, PBKDF2HMAC


def _load_keyring() -> Any | None:
    """Lazy-import ``keyring``. Returns ``None`` when unavailable."""
    try:
        import keyring  # type: ignore[import-not-found]
    except ImportError:
        return None
    except Exception as exc:  # pragma: no cover - backend issues
        logger.debug("keyring import failed: %s", exc)
        return None
    return keyring


def _keyring_get_password(service: str, account: str) -> str | None:
    """Safely query keyring. Returns ``None`` on any error or missing entry."""
    kr = _load_keyring()
    if kr is None:
        return None
    try:
        value = cast("str | None", kr.get_password(service, account))
    except Exception as exc:  # pragma: no cover - backend-specific
        logger.debug("keyring.get_password failed: %s", exc)
        return None
    return value


def _keyring_set_password(service: str, account: str, value: str) -> bool:
    """Attempt to write to keyring. Returns True on success."""
    kr = _load_keyring()
    if kr is None:
        return False
    try:
        kr.set_password(service, account, value)
    except Exception as exc:  # pragma: no cover - backend-specific
        logger.debug("keyring.set_password failed: %s", exc)
        return False
    return True


# ---------------------------------------------------------------------------
# WalletStore
# ---------------------------------------------------------------------------


class WalletStore:
    """Encrypted wallet storage.

    Data shape: an envelope JSON file containing a Fernet ciphertext of a
    secrets dict. Keys are arbitrary short names (e.g. ``"private_key"``);
    values are strings. Callers should not put non-string values through this
    store — use their own serialization upstream.

    Thread-safety: not thread-safe. Each caller should hold its own instance.
    """

    def __init__(self, home: Path | None = None) -> None:
        resolved = Path(home) if home is not None else carl_home()
        self._home: Path = resolved
        self._home.mkdir(parents=True, exist_ok=True)
        self._enc_path: Path = self._home / ENCRYPTED_FILENAME
        self._plain_path: Path = self._home / PLAINTEXT_FILENAME
        self._fernet: Any | None = None
        self._salt: bytes | None = None
        self._data: dict[str, str] = {}
        self._backend: str = "locked"
        self._unlocked: bool = False
        self._allow_plaintext: bool = False

    # -- Properties --------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """True when no key material has been loaded."""
        return not self._unlocked

    @property
    def backend(self) -> str:
        """Human-readable backend name: ``"fernet"``, ``"keyring"``, ``"plaintext"``, or ``"locked"``."""
        return self._backend

    @property
    def path(self) -> Path:
        """Path to the encrypted envelope (may or may not exist)."""
        return self._enc_path

    # -- Unlock flow -------------------------------------------------------

    def unlock(
        self,
        *,
        passphrase: str | None = None,
        allow_plaintext: bool = False,
    ) -> None:
        """Unlock the store.

        Resolution order:
          1. If ``passphrase`` provided -> derive key, load envelope if present.
          2. Else if ``CARL_WALLET_PASSPHRASE`` env var is set -> use it.
          3. Else if keyring has a stored master key -> use it.
          4. Else if ``allow_plaintext`` AND a plaintext file exists (or will be
             created) -> unlocked with plaintext backend.
          5. Else -> :class:`WalletLocked`.
        """
        env_pass = os.environ.get("CARL_WALLET_PASSPHRASE")
        resolved_pass = passphrase if passphrase is not None else env_pass

        if resolved_pass is not None:
            self._validate_passphrase(resolved_pass)
            self._unlock_with_passphrase(resolved_pass)
            self._backend = "fernet"
            self._unlocked = True
            self._allow_plaintext = False
            return

        # Try keyring master key.
        keyring_pass = _keyring_get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        if keyring_pass:
            self._unlock_with_passphrase(keyring_pass)
            self._backend = "keyring"
            self._unlocked = True
            self._allow_plaintext = False
            return

        # Plaintext fallback — only when explicitly opted in AND no encrypted
        # envelope exists (we do not silently downgrade a previously-encrypted
        # store).
        if allow_plaintext and not self._enc_path.exists():
            self._unlock_plaintext()
            self._backend = "plaintext"
            self._unlocked = True
            self._allow_plaintext = True
            return

        raise WalletLocked(
            "Wallet is locked. Provide a passphrase, set CARL_WALLET_PASSPHRASE, "
            "or store a master key in the OS keyring.",
            context={"hint": "carl camp wallet unlock"},
        )

    # -- Public read/write ------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return a secret value, or ``None`` if absent. Raises when locked."""
        self._require_unlocked()
        self._validate_key(key)
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        """Persist a secret value. Raises when locked or inputs invalid."""
        self._require_unlocked()
        self._validate_key(key)
        if type(value) is not str:
            raise ValidationError("Wallet value must be a string.")
        self._data[key] = value
        self._save()

    def delete(self, key: str) -> bool:
        """Remove a secret. Returns True if something was removed."""
        self._require_unlocked()
        self._validate_key(key)
        existed = key in self._data
        if existed:
            del self._data[key]
            self._save()
        return existed

    @staticmethod
    def _validate_key(key: Any) -> None:
        """Raise ValidationError on empty / non-str keys. Accepts Any for runtime safety."""
        if type(key) is not str or not key:
            raise ValidationError("Wallet key must be a non-empty string.")

    def keys(self) -> list[str]:
        """List stored key names (not values)."""
        self._require_unlocked()
        return sorted(self._data.keys())

    def rotate_passphrase(self, old: str, new: str) -> None:
        """Re-encrypt under a new passphrase.

        ``old`` must successfully decrypt the current envelope (or match a
        currently unlocked in-memory state derived from ``old``). ``new`` is
        validated and then used to rebuild the envelope with a fresh salt.
        """
        if type(old) is not str or type(new) is not str:
            raise ValidationError("Passphrases must be strings.")
        self._validate_passphrase(new)

        # Force a re-derivation path from ``old`` so we fail loudly on mismatch.
        probe = WalletStore(home=self._home)
        try:
            probe.unlock(passphrase=old)
        except WalletLocked as exc:
            raise WalletLocked(
                "Old passphrase did not unlock the wallet.",
                cause=exc,
            ) from exc

        # Copy current (in-memory) data onto the probe in case caller mutated
        # after a successful unlock — probe already contains the persisted data,
        # so `self._data` is authoritative only for unsaved edits. Prefer the
        # live store's data to avoid losing in-flight puts.
        probe._data = dict(self._data)

        # Derive new key material and rewrite the envelope atomically.
        probe._derive_key(new, salt=None)
        probe._save()

        # Adopt the probe's Fernet + salt so future reads/writes use the new key.
        self._fernet = probe._fernet
        self._salt = probe._salt
        self._data = probe._data
        self._backend = "fernet"
        self._unlocked = True

    # -- Internal ---------------------------------------------------------

    def _require_unlocked(self) -> None:
        if not self._unlocked:
            raise WalletLocked(
                "Wallet store is locked; call unlock() first.",
                context={"hint": "carl camp wallet unlock"},
            )

    def _validate_passphrase(self, passphrase: Any) -> None:
        if type(passphrase) is not str:
            raise ValidationError("Passphrase must be a string.")
        if len(passphrase) < 8:
            raise ValidationError(
                "Passphrase must be at least 8 characters.",
                context={"hint": "use a stronger passphrase"},
            )

    def _derive_key(self, passphrase: str, *, salt: bytes | None) -> None:
        """Derive a Fernet key via PBKDF2-HMAC-SHA256. Stores salt + Fernet."""
        _Fernet, _InvalidToken, hashes, PBKDF2HMAC = _load_cryptography()
        use_salt = salt if salt is not None else secrets.token_bytes(SALT_LENGTH)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=use_salt,
            iterations=PBKDF2_ITERATIONS,
        )
        raw_key = kdf.derive(passphrase.encode("utf-8"))
        fernet_key = base64.urlsafe_b64encode(raw_key)
        self._fernet = _Fernet(fernet_key)
        self._salt = use_salt
        # Defense-in-depth: scrub local references to raw key material.
        del raw_key
        del fernet_key

    def _unlock_with_passphrase(self, passphrase: str) -> None:
        """Derive key from passphrase; load + decrypt envelope if present."""
        _Fernet, InvalidToken, _h, _k = _load_cryptography()

        if self._enc_path.exists():
            envelope = self._read_envelope()
            salt = self._decode_salt(envelope["salt"])
            self._derive_key(passphrase, salt=salt)
            assert self._fernet is not None
            try:
                plaintext = self._fernet.decrypt(envelope["ct"].encode("utf-8"))
            except InvalidToken as exc:
                # Wrong passphrase or tampered ciphertext.
                raise WalletLocked(
                    "Failed to decrypt wallet — wrong passphrase.",
                ) from exc
            try:
                loaded = json.loads(plaintext.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WalletCorrupted(
                    "Decrypted payload was not valid JSON.",
                    cause=exc,
                ) from exc
            if not isinstance(loaded, dict):
                raise WalletCorrupted("Decrypted payload was not a JSON object.")
            # Coerce to str/str map; reject anything else.
            data: dict[str, str] = {}
            for k, v in cast(dict[Any, Any], loaded).items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise WalletCorrupted("Wallet payload contained non-string fields.")
                data[k] = v
            self._data = data
        else:
            self._derive_key(passphrase, salt=None)
            self._data = {}

    def _unlock_plaintext(self) -> None:
        """Load the plaintext fallback file (creating it empty if missing)."""
        if self._plain_path.exists():
            try:
                loaded = json.loads(self._plain_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise WalletCorrupted(
                    "Plaintext wallet file could not be parsed.",
                    cause=exc,
                ) from exc
            if not isinstance(loaded, dict):
                raise WalletCorrupted("Plaintext wallet file was not a JSON object.")
            data: dict[str, str] = {}
            for k, v in cast(dict[Any, Any], loaded).items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise WalletCorrupted("Plaintext wallet contained non-string fields.")
                data[k] = v
            self._data = data
        else:
            self._data = {}

    def _read_envelope(self) -> dict[str, Any]:
        try:
            text = self._enc_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise WalletCorrupted(
                "Failed to read encrypted wallet file.",
                cause=exc,
            ) from exc
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WalletCorrupted(
                "Wallet envelope was not valid JSON.",
                cause=exc,
            ) from exc
        if not isinstance(envelope, dict):
            raise WalletCorrupted("Wallet envelope was not a JSON object.")
        env = cast(dict[str, Any], envelope)
        if env.get("v") != ENVELOPE_VERSION:
            raise WalletCorrupted(
                f"Unsupported wallet envelope version: {env.get('v')!r}",
                context={"expected": ENVELOPE_VERSION},
            )
        if not isinstance(env.get("salt"), str) or not isinstance(env.get("ct"), str):
            raise WalletCorrupted("Wallet envelope missing required fields.")
        return env

    @staticmethod
    def _decode_salt(salt_b64: str) -> bytes:
        try:
            salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise WalletCorrupted("Wallet salt was not valid base64.", cause=exc) from exc
        if len(salt) < 8:
            raise WalletCorrupted("Wallet salt is too short.")
        return salt

    def _save(self) -> None:
        """Persist current ``_data``. Dispatches on backend."""
        if self._backend == "plaintext":
            self._save_plaintext()
            return
        # Encrypted path (fernet or keyring).
        if self._fernet is None or self._salt is None:
            raise CredentialError(
                "Wallet store missing key material; refusing to write.",
                context={"backend": self._backend},
            )
        payload = json.dumps(self._data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ct = self._fernet.encrypt(payload).decode("utf-8")
        envelope: dict[str, Any] = {
            "v": ENVELOPE_VERSION,
            "salt": base64.urlsafe_b64encode(self._salt).decode("ascii"),
            "ct": ct,
        }
        self._atomic_write(self._enc_path, json.dumps(envelope, indent=2))

    def _save_plaintext(self) -> None:
        self._atomic_write(
            self._plain_path,
            json.dumps(self._data, sort_keys=True, indent=2),
        )

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write ``text`` to ``path`` atomically with 0o600 permissions."""
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp = parent / f".{path.name}.tmp.{os.getpid()}"
        # POSIX: O_WRONLY|O_CREAT|O_TRUNC with 0o600 mode.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(tmp), flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        # Enforce 0o600 even if the umask / FS ignored the O_CREAT mode.
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as exc:  # pragma: no cover - non-POSIX / read-only
            logger.debug("chmod %s failed: %s", tmp, exc)
        os.replace(tmp, path)
        # Reassert mode on the destination too (rename can preserve old perms).
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as exc:  # pragma: no cover
            logger.debug("chmod %s failed: %s", path, exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def set_keyring_master_key(passphrase: str) -> bool:
    """Install ``passphrase`` as the OS-keychain master key for carl-studio.

    Returns True when stored successfully, False when ``keyring`` is missing
    or the backend refused the write.
    """
    if type(passphrase) is not str or len(passphrase) < 8:
        raise ValidationError("Keyring master key must be at least 8 characters.")
    return _keyring_set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, passphrase)


def keyring_available() -> bool:
    """True when ``keyring`` is importable AND a master key is stored."""
    return _keyring_get_password(KEYRING_SERVICE, KEYRING_ACCOUNT) is not None


__all__ = [
    "ENVELOPE_VERSION",
    "INSTALL_HINT",
    "KEYRING_ACCOUNT",
    "KEYRING_SERVICE",
    "PBKDF2_ITERATIONS",
    "SALT_LENGTH",
    "WalletCorrupted",
    "WalletLocked",
    "WalletMetadata",
    "WalletStore",
    "keyring_available",
    "set_keyring_master_key",
    "wallet_metadata_registry",
]
