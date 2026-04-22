"""CARL admin unlock — hardware-gated private repo access.

The admin layer is double-locked:
  1. CARL_ADMIN_SECRET env var (never committed, Tej's machines only)
  2. Hardware fingerprint (machine-specific HMAC derived from system serial)

Unlock on a new machine:
  export CARL_ADMIN_SECRET=<secret>  # set from your secure store
  carl admin unlock                  # writes ~/.carl/admin.key

Without both keys, this module is a no-op. No private code exposed.

Private repo: wheattoast11/carl-private (private HF dataset repo)
Public stubs call load_private() which returns the real implementation only
when is_admin() passes. Otherwise raises ImportError with access instructions.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

_ADMIN_KEY_PATH = Path.home() / ".carl" / "admin.key"
_PRIVATE_REPO = "wheattoast11/carl-private"


@lru_cache(maxsize=1)
def _hw_fingerprint() -> bytes:
    """Derive a deterministic hardware fingerprint for this machine."""
    parts: list[bytes] = []

    # macOS: IOPlatformSerialNumber (unique per physical machine)
    try:
        out = subprocess.check_output(
            ["ioreg", "-l", "-p", "IOService", "-r", "IOPlatformExpertDevice"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        for line in out.decode(errors="ignore").splitlines():
            if "IOPlatformSerialNumber" in line:
                serial = line.split('"')[-2]
                parts.append(serial.encode())
                break
    except Exception:
        pass

    # Fallback / supplemental: processor + hostname
    import platform
    import socket

    parts.append(platform.processor().encode())
    parts.append(socket.gethostname().encode())

    if not parts:
        # Last resort: fixed salt (no security, just prevents empty input)
        parts.append(b"carl-admin-fallback")

    return hashlib.sha256(b"|".join(parts)).digest()


def _derive_key(secret: str) -> str:
    """Derive the admin key for this machine from the admin secret."""
    return hmac.new(
        secret.encode(),
        _hw_fingerprint(),
        hashlib.sha256,
    ).hexdigest()


def is_admin() -> bool:
    """Check if admin mode is unlocked on this machine.

    Returns True only if:
      1. CARL_ADMIN_SECRET is set in environment
      2. ~/.carl/admin.key exists
      3. The stored key matches HMAC(hw_fingerprint, secret)
    """
    secret = os.environ.get("CARL_ADMIN_SECRET", "")
    if not secret or not _ADMIN_KEY_PATH.exists():
        return False
    try:
        stored = _ADMIN_KEY_PATH.read_text().strip()
        return hmac.compare_digest(stored, _derive_key(secret))
    except Exception:
        return False


def generate_admin_key() -> str:
    """Generate the admin key for this machine.

    Requires CARL_ADMIN_SECRET in environment.
    Returns the hex key string to write to ~/.carl/admin.key.
    """
    secret = os.environ.get("CARL_ADMIN_SECRET", "")
    if not secret:
        raise RuntimeError(
            "CARL_ADMIN_SECRET not set. "
            "Export it from your secure credential store before running 'carl admin unlock'."
        )
    return _derive_key(secret)


def write_admin_key() -> Path:
    """Generate and persist the admin key for this machine.

    Writes to ~/.carl/admin.key (mode 0o600).
    Call this once per machine with CARL_ADMIN_SECRET set.
    """
    key = generate_admin_key()
    _ADMIN_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ADMIN_KEY_PATH.write_text(key)
    _ADMIN_KEY_PATH.chmod(0o600)
    return _ADMIN_KEY_PATH


def clear_admin_key() -> None:
    """Remove the admin key from this machine."""
    if _ADMIN_KEY_PATH.exists():
        _ADMIN_KEY_PATH.unlink()


def admin_status() -> dict[str, str]:
    """Return admin status dict for display."""
    hw = _hw_fingerprint().hex()[:12]
    key_exists = _ADMIN_KEY_PATH.exists()
    secret_set = bool(os.environ.get("CARL_ADMIN_SECRET", ""))
    unlocked = is_admin()
    return {
        "status": "UNLOCKED" if unlocked else "LOCKED",
        "hw_fingerprint": f"{hw}...",
        "key_file": str(_ADMIN_KEY_PATH),
        "key_exists": str(key_exists),
        "secret_in_env": str(secret_set),
    }


def load_private(module_name: str) -> Any:
    """Dynamically import a module from the private CARL runtime.

    Resolution order (v0.17):

    1. **Local ``resonance`` package.** Terminals-team machines have the
       private repo at ``/Users/.../models/resonance/`` pip-installed
       editable. Try ``importlib.import_module(f"resonance.{module_name}")``
       first — fastest path, works offline, supports dotted submodules
       (e.g. ``"signals.constitutional"``).
    2. **HuggingFace dataset fallback.** For distributed access, download
       a single ``.py`` file from ``wheattoast11/carl-private``. Works only
       for flat module names (no dots) since the dataset has flat files.

    Only succeeds when :func:`is_admin` returns True. Without admin unlock
    raises ``ImportError`` with access instructions.

    Usage in stubs::

        from carl_studio.admin import load_private, is_admin
        if is_admin():
            mod = load_private("signals.constitutional")  # local resonance
            impl = mod.ConstitutionalLedgerImpl(...)
        else:
            raise ImportError("Requires admin unlock or resonance runtime")
    """
    if not is_admin():
        raise ImportError(
            f"Module '{module_name}' is in the private CARL runtime. "
            "Either install the resonance package or unlock admin mode: "
            "'carl admin unlock'. Contact support@terminals.tech for access."
        )

    # (1) Fast path — local resonance package
    try:
        import importlib

        return importlib.import_module(f"resonance.{module_name}")
    except ImportError:
        # Fall through to HF dataset path
        pass

    # (2) HF dataset fallback — flat module names only
    if "." in module_name:
        raise ImportError(
            f"Module '{module_name}' requires the local resonance package. "
            f"The HuggingFace dataset fallback only supports flat module names "
            f"(no dotted paths). Install the resonance package for access."
        )
    try:
        import importlib.util

        from huggingface_hub import hf_hub_download

        path: str = hf_hub_download(  # pyright: ignore[reportUnknownVariableType]
            repo_id=_PRIVATE_REPO,
            filename=f"{module_name}.py",
            repo_type="dataset",
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {module_name}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    except ImportError:
        raise
    except Exception as exc:
        raise ImportError(
            f"Failed to load private module '{module_name}' from {_PRIVATE_REPO}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# AdminToken — capability proof of admin-gate traversal
# ---------------------------------------------------------------------------
#
# The admin-gate seam (v0.17 plan §4.A8 / §9) wants a way for private runtime
# modules to register proprietary resolvers into a public `Vault` without
# carl-studio needing to import `resonance` at module load. Pattern:
#
#   1. `resonance.signals.foo` calls `admin.issue_token()` on import.
#   2. It passes the token + resolver to `vault.register_runtime_resolver`.
#   3. The public Vault class verifies the token is current + hardware-matches.
#
# This is a VISIBILITY guard, not a cryptographic one. Holding an AdminToken
# means "I went through the admin gate at this point in time on this machine."
# The real enforcement is the CI moat-boundary check (F7).


from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402


@dataclass(frozen=True)
class AdminToken:
    """Proof of admin-gate traversal. Issued by :func:`issue_token` only.

    Carry this opaque value to private-runtime entry points (e.g.
    ``Vault.register_runtime_resolver``) as a visibility marker that the
    registration flows from an admin-unlocked context. Tokens expire after
    one hour to bound replay damage if a token leaks.
    """

    issued_at: datetime
    hw_fingerprint_prefix: str  # first 12 hex of the current hw fingerprint
    _max_age_s: int = 3600  # 1 hour

    def is_expired(self, *, now: datetime | None = None) -> bool:
        current = now if now is not None else datetime.now(timezone.utc)
        return current >= self.issued_at + timedelta(seconds=self._max_age_s)

    def verify(self) -> None:
        """Raise :class:`ImportError` if the token is stale or mismatches the
        current hardware fingerprint.
        """
        if self.is_expired():
            raise ImportError(
                "AdminToken has expired (>1h old); re-issue via admin.issue_token()"
            )
        current_fp = _hw_fingerprint().hex()[:12]
        if self.hw_fingerprint_prefix != current_fp:
            raise ImportError(
                "AdminToken hardware fingerprint mismatch — token was issued "
                "on a different machine. Re-issue via admin.issue_token()."
            )


def issue_token() -> AdminToken:
    """Issue a fresh :class:`AdminToken`. Only succeeds when admin unlocked.

    Raises :class:`ImportError` if the admin gate is locked. Expected caller
    is the private runtime (``resonance``) at its own module-import time —
    the token is handed to the vault's ``register_runtime_resolver`` entry
    point as proof the registration came from an admin-gated context.
    """
    if not is_admin():
        raise ImportError(
            "issue_token() requires admin unlock. Run 'carl admin unlock' first."
        )
    return AdminToken(
        issued_at=datetime.now(timezone.utc),
        hw_fingerprint_prefix=_hw_fingerprint().hex()[:12],
    )
