"""JWS-signed AgentCard identity for A2A v1.0.

The A2A spec mandates that the ``agent/getAuthenticatedExtendedCard`` method
returns an AgentCard bound to a cryptographic identity — a JWS compact
serialization so the receiving party can prove the card truly came from the
declared issuer without having to trust the transport.

This module owns keypair generation, persistence, signing, and verification.

Key algorithm
-------------
ES256 (ECDSA over the NIST P-256 curve with SHA-256) by default. Chosen
because:

* It's mandatory in the A2A v1.0 spec's "minimum viable identity" section.
* Keys are small enough to drop into a filesystem or OS keychain without
  pagination issues.
* The ``cryptography`` library (already in our ``[wallet]`` extra) supplies
  every primitive we need — no ``python-jose`` / ``PyJWT`` dependency.

Key storage
-----------
Default: ``~/.carl/keys/agent.key`` (PEM-encoded private key, mode ``0o600``)
and ``~/.carl/keys/agent.pub`` (PEM-encoded public key, mode ``0o644``).

Fallback: when the OS keyring ``keyring`` package is installed, we ALSO
persist the private key under service ``carl-studio-a2a``, user ``agent-key``
so other tools (desktop apps, agent-wallet extensions) can pick it up. This
is additive — the filesystem remains the source of truth because keyring
access requires a UI unlock in most OSes and agents need unattended access.

Everything ``cryptography`` / ``keyring`` related is imported lazily so
``import carl_studio.a2a.identity`` stays cheap without the ``[wallet]``
extra.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carl_core.connection import (
    ConnectionAuthError,
    ConnectionUnavailableError,
)

# ---------------------------------------------------------------------------
# Supported signing algorithms.
# ---------------------------------------------------------------------------

_SUPPORTED_ALGORITHMS: frozenset[str] = frozenset({"ES256"})


def _require_cryptography() -> Any:
    """Lazy-import ``cryptography``. Raise :class:`ConnectionUnavailableError`
    with the correct install-hint when the ``[wallet]`` extra is not
    present."""
    try:
        import cryptography  # type: ignore[import-not-found] # noqa: F401
        from cryptography.hazmat.primitives import hashes, serialization  # type: ignore[import-not-found]
        from cryptography.hazmat.primitives.asymmetric import ec  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ConnectionUnavailableError(
            "cryptography not installed, pip install carl-studio[wallet]",
            context={"hint": "pip install carl-studio[wallet]"},
        ) from exc
    return ec, hashes, serialization


def _default_keystore_dir() -> Path:
    """``~/.carl/keys`` by default. Honors ``CARL_HOME`` if set (for tests)."""
    home = os.environ.get("CARL_HOME")
    if home:
        return Path(home).expanduser() / "keys"
    return Path.home() / ".carl" / "keys"


# ---------------------------------------------------------------------------
# Base64URL helpers (JWS uses RFC 4648 Section 5, unpadded).
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


# ---------------------------------------------------------------------------
# AgentIdentity — keypair wrapper.
# ---------------------------------------------------------------------------


@dataclass
class AgentIdentity:
    """Wraps the agent's cryptographic identity.

    Holds the PEM-encoded private and/or public key bytes plus the algorithm
    tag. Instances are cheap to construct but the PEM parse happens lazily
    on the first ``sign_card`` / ``verify_card`` call.

    Most callers go through :meth:`AgentIdentity.load` which handles key
    discovery (filesystem, then keyring) and generation-on-miss.
    """

    private_key_pem: bytes | None = None
    public_key_pem: bytes | None = None
    algorithm: str = "ES256"
    kid: str = "carl-studio-agent-es256"
    keystore_dir: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"unsupported JWS algorithm: {self.algorithm}. "
                f"Supported: {sorted(_SUPPORTED_ALGORITHMS)}"
            )
        if self.private_key_pem is None and self.public_key_pem is None:
            raise ValueError(
                "AgentIdentity requires at least one of private_key_pem "
                "or public_key_pem",
            )

    # -- discovery / persistence ----------------------------------------

    @classmethod
    def load(
        cls,
        keystore_dir: Path | None = None,
        *,
        algorithm: str = "ES256",
        create_if_missing: bool = True,
    ) -> AgentIdentity:
        """Load (or generate) the agent identity from disk.

        Search order:

        1. ``{keystore_dir}/agent.key`` + ``agent.pub`` (or the default
           ``~/.carl/keys/``).
        2. OS keyring under service ``carl-studio-a2a``, key ``agent-key``.
        3. If both miss and ``create_if_missing`` is True, generate a fresh
           ES256 keypair, persist it to disk with ``0o600`` perms, and
           return.
        """
        directory = (keystore_dir or _default_keystore_dir()).expanduser()
        priv_path = directory / "agent.key"
        pub_path = directory / "agent.pub"

        priv_bytes: bytes | None = None
        pub_bytes: bytes | None = None

        if priv_path.exists():
            try:
                priv_bytes = priv_path.read_bytes()
            except OSError as exc:
                raise ConnectionAuthError(
                    f"failed to read private key: {exc}",
                    context={"path": str(priv_path)},
                ) from exc
        if pub_path.exists():
            try:
                pub_bytes = pub_path.read_bytes()
            except OSError as exc:
                raise ConnectionAuthError(
                    f"failed to read public key: {exc}",
                    context={"path": str(pub_path)},
                ) from exc

        if priv_bytes is None:
            ring_bytes = _try_keyring_load()
            if ring_bytes is not None:
                priv_bytes = ring_bytes

        if priv_bytes is None and pub_bytes is None:
            if not create_if_missing:
                raise ConnectionAuthError(
                    "no agent identity found and create_if_missing=False",
                    context={"keystore_dir": str(directory)},
                )
            priv_bytes, pub_bytes = _generate_es256_pem()
            _persist_keypair(directory, priv_bytes, pub_bytes)
            _try_keyring_save(priv_bytes)

        # Derive + persist pub if we only recovered priv (e.g. from keyring).
        if priv_bytes is not None and pub_bytes is None:
            _ec, _hashes, serialization = _require_cryptography()
            priv_obj = serialization.load_pem_private_key(priv_bytes, password=None)
            derived_pub: bytes = priv_obj.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            pub_bytes = derived_pub
            try:
                _persist_keypair(directory, priv_bytes, derived_pub)
            except ConnectionAuthError:
                # Persistence failure is tolerable — we still return a
                # working in-memory identity.
                pass

        return cls(
            private_key_pem=priv_bytes,
            public_key_pem=pub_bytes,
            algorithm=algorithm,
            keystore_dir=directory,
        )

    # -- public-key derivation ------------------------------------------

    def public_key_bytes(self) -> bytes:
        """Return the PEM-encoded public key, deriving from the private key
        if necessary."""
        if self.public_key_pem is not None:
            return self.public_key_pem
        if self.private_key_pem is None:
            raise ConnectionAuthError(
                "AgentIdentity has no key material to derive public key from",
            )
        _ec, _hashes, serialization = _require_cryptography()
        priv = serialization.load_pem_private_key(self.private_key_pem, password=None)
        pub = priv.public_key()
        pem: bytes = pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.public_key_pem = pem
        return pem

    # -- signing --------------------------------------------------------

    def sign_card(self, card: Any) -> str:
        """Return a JWS compact serialization of the card's canonical JSON.

        ``card`` is anything that either exposes ``to_json`` (as
        :class:`CARLAgentCard` does), supports ``dataclasses.asdict``, or is
        already a dict. The payload is canonicalized (sorted keys, no
        whitespace) so the signature binds to a deterministic byte sequence.
        """
        if self.private_key_pem is None:
            raise ConnectionAuthError(
                "cannot sign: AgentIdentity has no private key",
                context={"kid": self.kid},
            )
        ec, hashes, serialization = _require_cryptography()

        payload_bytes = _canonical_card_bytes(card)
        header = {"alg": self.algorithm, "typ": "JWT", "kid": self.kid}
        header_b64 = _b64url_encode(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        payload_b64 = _b64url_encode(payload_bytes)
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

        try:
            priv = serialization.load_pem_private_key(
                self.private_key_pem, password=None
            )
        except Exception as exc:
            raise ConnectionAuthError(
                "failed to load private key for signing",
                context={"kid": self.kid, "reason": type(exc).__name__},
            ) from exc

        der_sig: bytes = priv.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        # JWS requires the signature in (R || S) raw form, not ASN.1 DER.
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature  # type: ignore[import-not-found]

        r, s = decode_dss_signature(der_sig)
        raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        sig_b64 = _b64url_encode(raw_sig)
        return f"{header_b64}.{payload_b64}.{sig_b64}"

    # -- verification ---------------------------------------------------

    def verify_card(
        self,
        jws_token: str,
        expected_kid: str | None = None,
    ) -> Any:
        """Verify ``jws_token``, returning the parsed :class:`CARLAgentCard`.

        Raises :class:`ConnectionAuthError` on any tamper / malformed input.
        """
        from carl_studio.a2a.agent_card import CARLAgentCard

        ec, hashes, serialization = _require_cryptography()

        try:
            header_b64, payload_b64, sig_b64 = jws_token.split(".")
        except ValueError as exc:
            raise ConnectionAuthError(
                "malformed JWS token — expected three dot-separated parts",
                context={"kid": expected_kid},
            ) from exc

        try:
            header_obj: Any = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        except Exception as exc:
            raise ConnectionAuthError(
                "JWS header is not valid JSON",
                context={"kid": expected_kid},
            ) from exc

        if not isinstance(header_obj, dict):
            raise ConnectionAuthError(
                "JWS header must be a JSON object",
                context={"kid": expected_kid},
            )
        header: dict[str, Any] = dict(header_obj)  # type: ignore[arg-type]
        alg_any: Any = header.get("alg")
        if alg_any != self.algorithm:
            raise ConnectionAuthError(
                f"JWS alg mismatch: expected {self.algorithm}, got {alg_any}",
                context={"kid": expected_kid, "alg": alg_any},
            )
        if expected_kid is not None and header.get("kid") != expected_kid:
            raise ConnectionAuthError(
                "JWS kid does not match expected value",
                context={
                    "expected_kid": expected_kid,
                    "actual_kid": header.get("kid"),
                },
            )

        try:
            raw_sig = _b64url_decode(sig_b64)
            if len(raw_sig) != 64:
                raise ConnectionAuthError(
                    "JWS signature length != 64 bytes (expected ES256 R||S)",
                    context={"len": len(raw_sig)},
                )
            from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature  # type: ignore[import-not-found]

            r = int.from_bytes(raw_sig[:32], "big")
            s = int.from_bytes(raw_sig[32:], "big")
            der_sig: bytes = encode_dss_signature(r, s)
        except ConnectionAuthError:
            raise
        except Exception as exc:
            raise ConnectionAuthError(
                "failed to decode JWS signature",
                context={"reason": type(exc).__name__},
            ) from exc

        try:
            pub_pem = self.public_key_bytes()
            pub = serialization.load_pem_public_key(pub_pem)
        except Exception as exc:
            raise ConnectionAuthError(
                "failed to load public key for verification",
                context={"reason": type(exc).__name__},
            ) from exc

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        try:
            pub.verify(der_sig, signing_input, ec.ECDSA(hashes.SHA256()))
        except Exception as exc:
            raise ConnectionAuthError(
                "JWS signature verification failed",
                context={"kid": header.get("kid")},
            ) from exc

        try:
            payload_bytes = _b64url_decode(payload_b64)
            payload_any: Any = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            raise ConnectionAuthError(
                "JWS payload is not valid JSON",
            ) from exc
        if not isinstance(payload_any, dict):
            raise ConnectionAuthError(
                "JWS payload must be a JSON object (the AgentCard)",
            )
        card_dict: dict[str, Any] = dict(payload_any)  # type: ignore[arg-type]
        # CARLAgentCard.from_json wants a string; feed canonical bytes.
        return CARLAgentCard.from_json(json.dumps(card_dict))


# ---------------------------------------------------------------------------
# Key generation / persistence helpers.
# ---------------------------------------------------------------------------


def _generate_es256_pem() -> tuple[bytes, bytes]:
    """Generate a fresh ES256 keypair, returning (priv_pem, pub_pem)."""
    ec, _hashes, serialization = _require_cryptography()
    priv = ec.generate_private_key(ec.SECP256R1())
    priv_pem: bytes = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem: bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


def _persist_keypair(directory: Path, priv_pem: bytes, pub_pem: bytes) -> None:
    """Write keypair to ``directory`` with ``0o600`` perms for the private key."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        priv_path = directory / "agent.key"
        pub_path = directory / "agent.pub"
        priv_path.write_bytes(priv_pem)
        try:
            os.chmod(priv_path, 0o600)
        except OSError:
            # Best-effort. Windows / unusual filesystems won't accept chmod —
            # the file is still written; subsequent ``load`` calls work.
            pass
        pub_path.write_bytes(pub_pem)
        try:
            os.chmod(pub_path, 0o644)
        except OSError:
            pass
    except OSError as exc:
        raise ConnectionAuthError(
            f"failed to persist agent identity: {exc}",
            context={"directory": str(directory)},
        ) from exc


def _try_keyring_load() -> bytes | None:
    """Load the private key PEM from the OS keyring, if ``keyring`` is
    installed. Returns ``None`` on any failure."""
    try:
        import keyring  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        raw = keyring.get_password("carl-studio-a2a", "agent-key")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return raw.encode("utf-8")
    except Exception:
        return None


def _try_keyring_save(priv_pem: bytes) -> None:
    """Persist the private key to the OS keyring best-effort."""
    try:
        import keyring  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        keyring.set_password(
            "carl-studio-a2a", "agent-key", priv_pem.decode("utf-8")
        )
    except Exception:
        # Keyring failures are non-fatal — filesystem is source of truth.
        pass


def _canonical_card_bytes(card: Any) -> bytes:
    """Return the canonical JSON bytes for a card-shaped object.

    Canonicalization rules (JCS-ish, since we control both ends):

    * ``dict`` keys sorted.
    * No trailing whitespace.
    * ``datetime`` / ``Enum`` coerced to string.
    * ``None`` preserved (the spec does not reject nulls).
    """
    raw: Any = None
    to_json_attr: Any = getattr(card, "to_json", None)
    if callable(to_json_attr):
        try:
            rendered: Any = to_json_attr()
            if isinstance(rendered, (str, bytes, bytearray)):
                raw = json.loads(rendered)
        except Exception:
            raw = None
    if raw is None:
        if dataclasses.is_dataclass(card) and not isinstance(card, type):
            raw = dataclasses.asdict(card)
        elif isinstance(card, dict):
            raw = dict(card)  # type: ignore[arg-type]
        else:
            model_dump: Any = getattr(card, "model_dump", None)  # pydantic v2
            if callable(model_dump):
                dumped: Any = model_dump()
                raw = dumped
            else:
                raise ValueError(
                    f"cannot canonicalize card of type {type(card).__name__}"
                )
    encoded: bytes = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return encoded


__all__ = [
    "AgentIdentity",
]
