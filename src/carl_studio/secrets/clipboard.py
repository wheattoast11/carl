"""ClipboardBridge — TTL-bounded handle-based clipboard.

Writes a vault-resolved value to the OS clipboard, records a
``CLIPBOARD_WRITE`` Step with the 12-hex fingerprint (not the value),
and schedules a background wipe after ``ttl_s`` seconds. The agent
never sees the bytes; the OS clipboard does, and an external target
(browser form, desktop app) can paste.

Wrap behavior
-------------

* ``write_from_ref(ref, ttl_s=30)`` — pulls the value via privileged
  ``vault.resolve``, ``pyperclip.copy``-s it, schedules a
  ``threading.Timer`` to wipe after ``ttl_s``.
* ``wipe()`` — explicit clear. Idempotent.
* ``was_modified_since(t0)`` — integrity probe; if the clipboard was
  modified externally between the ref write and now, the scheduled
  wipe is a no-op (OS already changed the contents).

Install: ``pip install 'carl-studio[secrets]'`` pulls ``pyperclip>=1.9``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from carl_core.errors import CARLError
from carl_core.interaction import ActionType, InteractionChain
from carl_core.secrets import SecretRef, SecretVault


__all__ = ["ClipboardBridge", "ClipboardBridgeError"]


class ClipboardBridgeError(CARLError):
    """Base for ``carl.clipboard.*`` errors."""

    code = "carl.clipboard"


class ClipboardBridge:
    """Scoped clipboard writer with TTL auto-wipe and audit trail.

    Thread-safe for the auto-wipe Timer. Every write records a
    ``CLIPBOARD_WRITE`` Step; the scheduled wipe emits a
    ``SECRET_REVOKE``-named Step on the chain so the audit trail shows
    both boundaries explicitly.

    Integrity: if ``pyperclip.paste()`` returns a different value at
    wipe time than what we wrote, we assume the user (or another app)
    overwrote the clipboard, and we skip the wipe. The audit Step
    records ``modified_externally=True`` in that case.
    """

    def __init__(
        self,
        vault: SecretVault,
        *,
        chain: InteractionChain | None = None,
        default_ttl_s: int = 30,
    ) -> None:
        if default_ttl_s < 1:
            raise ValueError(f"default_ttl_s must be >= 1, got {default_ttl_s}")
        self.vault = vault
        self.chain = chain
        self.default_ttl_s = default_ttl_s
        self._lock = threading.RLock()
        self._active_timer: threading.Timer | None = None
        self._active_signature: str | None = None  # fingerprint of last write
        self._active_written_at: datetime | None = None

    # -- availability --------------------------------------------------

    @staticmethod
    def available() -> bool:
        """True iff ``pyperclip`` is importable + has a working backend."""
        try:
            import pyperclip

            # pyperclip.paste() raises if no backend — probe is safe.
            pyperclip.paste()
            return True
        except Exception:
            return False

    # -- primary operations --------------------------------------------

    def write_from_ref(
        self,
        ref: SecretRef,
        *,
        ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Copy a vault-held value to the OS clipboard with auto-wipe.

        Returns a metadata dict (``fingerprint`` / ``expires_at`` /
        ``ttl_s``) — never the value. Cancels any previously scheduled
        wipe so there's always at most one pending Timer.
        """
        pyperclip = self._pyperclip_or_raise()
        effective_ttl = int(ttl_s if ttl_s is not None else self.default_ttl_s)
        if effective_ttl < 1:
            raise ClipboardBridgeError(
                f"ttl_s must be >= 1, got {effective_ttl}",
                code="carl.clipboard.invalid_ttl",
                context={"ttl_s": effective_ttl},
            )

        value_bytes = self.vault.resolve(ref, privileged=True)
        try:
            value_str = value_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ClipboardBridgeError(
                "clipboard values must be UTF-8 strings",
                code="carl.clipboard.unsupported_value",
                context={"ref_id": str(ref.ref_id)},
                cause=exc,
            ) from exc

        fingerprint_hex = self.vault.fingerprint_of(ref)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=effective_ttl)

        with self._lock:
            # Cancel any pending wipe from a prior write.
            if self._active_timer is not None:
                self._active_timer.cancel()
                self._active_timer = None

            pyperclip.copy(value_str)

            self._active_signature = fingerprint_hex
            self._active_written_at = now

            timer = threading.Timer(
                float(effective_ttl),
                self._auto_wipe,
                kwargs={"expected_signature": fingerprint_hex},
            )
            timer.daemon = True
            timer.start()
            self._active_timer = timer

        # Audit boundary — write event.
        if self.chain is not None:
            self.chain.record(
                ActionType.CLIPBOARD_WRITE,
                name="clipboard.write_from_ref",
                input={"ref_id": str(ref.ref_id), "ttl_s": effective_ttl},
                output={
                    "fingerprint": fingerprint_hex,
                    "expires_at": expires_at.isoformat(),
                    "ttl_s": effective_ttl,
                },
            )

        return {
            "fingerprint": fingerprint_hex,
            "expires_at": expires_at.isoformat(),
            "ttl_s": effective_ttl,
        }

    def wipe(self) -> bool:
        """Explicit clipboard clear. Returns True iff the clipboard was cleared.

        Idempotent — calling wipe() on an already-empty clipboard is fine.
        Emits a ``SECRET_REVOKE``-named Step (name=``clipboard.wipe``).
        """
        pyperclip = self._pyperclip_or_raise()
        with self._lock:
            if self._active_timer is not None:
                self._active_timer.cancel()
                self._active_timer = None
            try:
                pyperclip.copy("")
            except Exception:
                return False
            signature = self._active_signature
            self._active_signature = None
            self._active_written_at = None

        if self.chain is not None:
            self.chain.record(
                ActionType.SECRET_REVOKE,
                name="clipboard.wipe",
                input={},
                output={
                    "prior_fingerprint": signature,
                    "trigger": "explicit",
                },
            )
        return True

    def was_modified_since(self, t0: datetime) -> bool:
        """True iff the clipboard content changed between ``t0`` and now.

        Reads the current clipboard value and fingerprints it locally;
        compares against the fingerprint we stored at the last
        :meth:`write_from_ref`. Does NOT log the live clipboard value.
        """
        if self._active_signature is None or self._active_written_at is None:
            return False
        if self._active_written_at > t0:
            return False
        pyperclip = self._pyperclip_or_raise()
        try:
            current = pyperclip.paste()
        except Exception:
            return True  # can't read → assume changed
        from carl_core.hashing import fingerprint as _fp

        return _fp(current) != self._active_signature

    # -- internals ------------------------------------------------------

    def _auto_wipe(self, *, expected_signature: str) -> None:
        """Scheduled callback: wipe only if nobody else has touched the clipboard.

        Runs in the Timer thread. Never raises.
        """
        try:
            pyperclip = self._pyperclip_or_raise()
        except ClipboardBridgeError:
            return
        try:
            current = pyperclip.paste()
        except Exception:
            current = ""
        from carl_core.hashing import fingerprint as _fp

        current_signature = _fp(current) if current else None
        modified_externally = (
            current_signature is not None and current_signature != expected_signature
        )

        with self._lock:
            # Clear state regardless; the Timer has fired.
            self._active_timer = None
            self._active_signature = None
            self._active_written_at = None

            if not modified_externally:
                try:
                    pyperclip.copy("")
                except Exception:
                    pass

        if self.chain is not None:
            self.chain.record(
                ActionType.SECRET_REVOKE,
                name="clipboard.auto_wipe",
                input={"expected_signature": expected_signature},
                output={
                    "trigger": "ttl",
                    "modified_externally": modified_externally,
                },
            )

    @staticmethod
    def _pyperclip_or_raise() -> Any:
        try:
            import pyperclip

            return pyperclip
        except ImportError as exc:
            raise ClipboardBridgeError(
                "pyperclip is not installed. Install with: "
                "pip install 'carl-studio[secrets]'",
                code="carl.clipboard.unavailable",
                cause=exc,
            ) from exc
