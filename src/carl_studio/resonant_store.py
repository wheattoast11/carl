"""Local storage + user_secret management for Resonants.

File layout (all paths rooted at ``~/.carl``):

    credentials/user_secret       32 bytes raw, mode 0600, auto-generated
    resonants/<name>/tree.emlt    envelope: EMLT magic | VERSION | inner tree | 32-byte sig
    resonants/<name>/projection.npy   numpy float64, shape (k, d)
    resonants/<name>/readout.npy      numpy float64, shape (a, k)
    resonants/<name>/metadata.json    free-form annotations + content_hash

The identity fingerprint ``sha256(user_secret)[:16]`` is a one-way
derivative used as a stable public handle for the user on the
marketplace. Upgrade path: in v0.10 it becomes the ed25519 pubkey hex.
Schema unchanged.

See ``docs/eml_signing_protocol.md`` §5 for the full contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from carl_core.eml import EMLTree
from carl_core.errors import ValidationError
from carl_core.resonant import Resonant, make_resonant
from carl_core.signing import sign_tree_software

CARL_HOME: Path = Path.home() / ".carl"
CREDENTIALS_DIR: Path = CARL_HOME / "credentials"
RESONANTS_DIR: Path = CARL_HOME / "resonants"
USER_SECRET_PATH: Path = CREDENTIALS_DIR / "user_secret"
USER_SECRET_BYTES: int = 32

_ENVELOPE_MAGIC: bytes = b"EMLT"
_ENVELOPE_VERSION: int = 1
_SIG_LEN: int = 32

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# ---------------------------------------------------------------------------
# user_secret management
# ---------------------------------------------------------------------------


def _ensure_credentials_dir() -> None:
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CREDENTIALS_DIR, 0o700)
    except OSError:
        pass


def read_or_create_user_secret() -> bytes:
    """Return the user's 32-byte raw secret, creating it if absent.

    Writes ``~/.carl/credentials/user_secret`` with mode 0600 on first
    call. Never logs the bytes. Callers MUST NOT print or persist the
    returned value.
    """
    _ensure_credentials_dir()
    if USER_SECRET_PATH.exists():
        data = USER_SECRET_PATH.read_bytes()
        if len(data) < 16:
            raise ValidationError(
                f"user_secret at {USER_SECRET_PATH} is too short ({len(data)} bytes); "
                "delete the file to regenerate",
                code="carl.credential.invalid",
                context={"path": str(USER_SECRET_PATH), "length": len(data)},
            )
        return data
    secret = os.urandom(USER_SECRET_BYTES)
    tmp = USER_SECRET_PATH.with_suffix(".tmp")
    tmp.write_bytes(secret)
    os.chmod(tmp, 0o600)
    tmp.replace(USER_SECRET_PATH)
    return secret


def identity_fingerprint(secret: bytes | None = None) -> str:
    """Return ``sha256(secret)[:16]`` hex — 32 chars, one-way, stable per secret.

    When ``secret`` is None, reads (or creates) the on-disk user_secret.
    Safe to log; does NOT reveal the secret.
    """
    s = secret if secret is not None else read_or_create_user_secret()
    return hashlib.sha256(s).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Envelope helpers (inlined to avoid importing the private terminals-runtime
# codec; matches docs/eml_signing_protocol.md §1.2 exactly)
# ---------------------------------------------------------------------------


def encode_envelope(tree_bytes: bytes, sig: bytes | None = None) -> bytes:
    """Wrap inner ``tree_bytes`` in the EMLT header + optional sig tail."""
    if sig is not None and len(sig) != _SIG_LEN:
        raise ValidationError(
            f"signature must be {_SIG_LEN} bytes; got {len(sig)}",
            code="carl.eml.domain_error",
            context={"sig_len": len(sig)},
        )
    body = bytearray(_ENVELOPE_MAGIC)
    body.append(_ENVELOPE_VERSION)
    body += tree_bytes
    if sig is not None:
        body += sig
    return bytes(body)


def decode_envelope(data: bytes) -> tuple[bytes, bytes | None]:
    """Inverse of :func:`encode_envelope`. Returns ``(inner_tree_bytes, sig_or_None)``."""
    if len(data) < len(_ENVELOPE_MAGIC) + 1:
        raise ValidationError(
            "envelope too short", code="carl.eml.decode_error",
            context={"size": len(data)},
        )
    if data[: len(_ENVELOPE_MAGIC)] != _ENVELOPE_MAGIC:
        raise ValidationError(
            "bad envelope magic", code="carl.eml.decode_error",
            context={"magic": data[: len(_ENVELOPE_MAGIC)].hex()},
        )
    version = data[len(_ENVELOPE_MAGIC)]
    if version != _ENVELOPE_VERSION:
        raise ValidationError(
            f"unsupported envelope version {version}",
            code="carl.eml.decode_error",
            context={"version": version, "supported": _ENVELOPE_VERSION},
        )
    remainder = data[len(_ENVELOPE_MAGIC) + 1 :]
    if len(remainder) > _SIG_LEN:
        # Try stripping trailing 32 bytes as signature. If what's left parses
        # as a valid inner tree, accept as signed. Otherwise treat as unsigned.
        candidate_inner = remainder[:-_SIG_LEN]
        candidate_sig = remainder[-_SIG_LEN:]
        try:
            EMLTree.from_bytes(candidate_inner)
        except ValidationError:
            # Stripping breaks parse → signature-less; fall through
            pass
        else:
            try:
                EMLTree.from_bytes(remainder)
            except ValidationError:
                return candidate_inner, candidate_sig
            return remainder, None
    return remainder, None


# ---------------------------------------------------------------------------
# Resonant storage
# ---------------------------------------------------------------------------


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValidationError(
            f"invalid resonant name {name!r}; must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}",
            code="carl.validation.name_invalid",
            context={"name": name},
        )


def resonant_path(name: str) -> Path:
    _validate_name(name)
    return RESONANTS_DIR / name


def save_resonant(
    name: str,
    resonant: Resonant,
    user_secret: bytes | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a Resonant to ``~/.carl/resonants/<name>/``.

    Writes the signed envelope, projection + readout matrices, and a
    metadata JSON. Overwrites any existing entry with the same name.

    ``user_secret`` defaults to the on-disk secret (auto-generated on
    first use). The HMAC signature is computed over the inner tree
    bytes only, per the signing protocol.
    """
    _validate_name(name)
    secret = user_secret if user_secret is not None else read_or_create_user_secret()

    dir_path = RESONANTS_DIR / name
    dir_path.mkdir(parents=True, exist_ok=True)

    inner = resonant.tree.to_bytes()
    sig = sign_tree_software(inner, secret)
    envelope = encode_envelope(inner, sig)

    (dir_path / "tree.emlt").write_bytes(envelope)
    np.save(dir_path / "projection.npy", np.asarray(resonant.projection, dtype=np.float64))
    np.save(dir_path / "readout.npy", np.asarray(resonant.readout, dtype=np.float64))

    meta = dict(resonant.metadata)
    if metadata:
        meta.update(metadata)
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    meta["name"] = name
    meta["identity"] = resonant.identity
    meta["tree_hash"] = resonant.tree.hash()
    meta["input_dim"] = int(resonant.tree.input_dim)
    meta["output_dim"] = int(resonant.readout.shape[0])
    meta["latent_dim"] = int(resonant.projection.shape[0])
    meta["depth"] = int(resonant.tree.depth())
    meta["sig_public_component"] = identity_fingerprint(secret)
    (dir_path / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    return dir_path


def load_resonant(name: str) -> tuple[Resonant, bytes, dict[str, Any]]:
    """Reconstitute a saved Resonant.

    Returns ``(resonant, envelope_bytes, metadata)``. The envelope bytes
    are the on-wire form suitable for ``POST /api/resonants``.
    """
    _validate_name(name)
    dir_path = RESONANTS_DIR / name
    if not dir_path.is_dir():
        raise ValidationError(
            f"no resonant named {name!r} at {dir_path}",
            code="carl.validation.not_found",
            context={"name": name, "path": str(dir_path)},
        )

    envelope = (dir_path / "tree.emlt").read_bytes()
    inner, _sig = decode_envelope(envelope)
    tree = EMLTree.from_bytes(inner)
    projection = np.load(dir_path / "projection.npy")
    readout = np.load(dir_path / "readout.npy")
    meta_raw = (dir_path / "metadata.json").read_text()
    meta = json.loads(meta_raw)

    resonant = make_resonant(
        tree, projection, readout,
        metadata={k: v for k, v in meta.items() if k not in {"identity", "tree_hash", "sig_public_component"}},
    )
    return resonant, envelope, meta


def list_resonants() -> list[dict[str, Any]]:
    """Enumerate local Resonants with lightweight metadata.

    Returns a list of ``{name, identity, tree_hash, input_dim, output_dim,
    depth, created_at}`` dicts, sorted by name.
    """
    if not RESONANTS_DIR.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for child in sorted(RESONANTS_DIR.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(
            {
                "name": meta.get("name", child.name),
                "identity": meta.get("identity", ""),
                "tree_hash": meta.get("tree_hash", ""),
                "input_dim": meta.get("input_dim"),
                "output_dim": meta.get("output_dim"),
                "depth": meta.get("depth"),
                "created_at": meta.get("created_at", ""),
                "path": str(child),
            }
        )
    return entries


def delete_resonant(name: str) -> bool:
    """Remove a saved Resonant. Returns True if anything was deleted."""
    _validate_name(name)
    dir_path = RESONANTS_DIR / name
    if not dir_path.exists():
        return False
    for child in dir_path.iterdir():
        child.unlink(missing_ok=True)
    dir_path.rmdir()
    return True


__all__ = [
    "CARL_HOME",
    "CREDENTIALS_DIR",
    "RESONANTS_DIR",
    "USER_SECRET_PATH",
    "USER_SECRET_BYTES",
    "decode_envelope",
    "delete_resonant",
    "encode_envelope",
    "identity_fingerprint",
    "list_resonants",
    "load_resonant",
    "read_or_create_user_secret",
    "resonant_path",
    "save_resonant",
]
