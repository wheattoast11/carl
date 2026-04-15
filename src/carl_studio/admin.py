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
    """Dynamically import a module from the private CARL repo.

    Only succeeds when is_admin() returns True.
    Modules live in wheattoast11/carl-private as dataset files.

    Usage in stubs::

        from carl_studio.admin import load_private, is_admin
        if is_admin():
            mod = load_private("slot")          # downloads slot.py
            impl = mod.SLOTOptimizerImpl(...)
        else:
            raise ImportError("Requires admin unlock or terminals-runtime")
    """
    if not is_admin():
        raise ImportError(
            f"Module '{module_name}' is in the private CARL runtime. "
            "Either install terminals-runtime or unlock admin mode: 'carl admin unlock'. "
            "Contact support@terminals.tech for access."
        )
    try:
        from huggingface_hub import hf_hub_download
        import importlib.util

        path = hf_hub_download(
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
