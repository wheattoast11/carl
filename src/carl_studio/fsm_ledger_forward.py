"""carl-studio constitutional ledger HTTP forwarder (Phase H-S4a).

Forwards signed :class:`carl_core.constitutional.LedgerBlock` instances
to carl.camp's existing ``/api/ledger/append`` route (Phase H-B6a hardened
the receiving table). Always persists locally to
``~/.carl/constitutional_ledger.jsonl`` with 10MB rotation; HTTP forward
is opt-in via ``consent.telemetry``.

Replay safety: locally-persisted blocks are JSONL-tagged with the
signature_hex so :meth:`ConstitutionalForwarder.replay_pending` can walk
the file and retry any blocks that don't yet carry an ``acked: true``
annotation.

Decision ladder
---------------

1. Always ``_append_local(block)`` first — local persistence is the
   source of truth; the carl.camp ledger is a secondary mirror.
2. Forward only when ``consent.telemetry == True``. Default-off,
   matching the privacy-first consent doctrine.
3. Skip the POST when no bearer token is resolvable (FREE / unauth
   path); this is a no-op, not an error.
4. On HTTP success, mark the persisted line with ``acked: true``.
5. On HTTP failure, swallow the error in :meth:`forward_block` (the
   caller's local persist already succeeded). The structured
   ``reason`` field in the return dict surfaces the failure to
   instrumentation; replay_pending() picks the block back up later.

Privacy
-------

- Bearer tokens never leave the resolver; the request header is built
  inline and discarded after the call.
- ``LedgerBlock`` payloads are already the constitutional signing
  surface — they don't carry the source action's input/output text,
  only the 25-dim feature digest + verdict + signature.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import httpx

from carl_core.constitutional import LedgerBlock
from carl_core.errors import ConstitutionalForwardFailedError
from carl_core.interaction import ActionType, InteractionChain

logger = logging.getLogger(__name__)


DEFAULT_CARL_CAMP_BASE: str = "https://carl.camp"
APPEND_PATH: str = "/api/ledger/append"
DEFAULT_LEDGER_PATH: Path = Path.home() / ".carl" / "constitutional_ledger.jsonl"
ROTATION_BYTES: int = 10 * 1024 * 1024  # 10 MB per active file.
MAX_ROTATIONS: int = 3                  # Keep .1, .2, .3 — total ceiling 40 MB.
LEDGER_FILE_MODE: int = 0o600           # private; never let other users read.

# Sentinel exposed for instrumentation: the error code paired with this
# error class is the canonical taxonomy entry for an exhausted forward.
FORWARD_FAILED_CODE: str = ConstitutionalForwardFailedError.code


# ---------------------------------------------------------------------------
# ConstitutionalForwarder
# ---------------------------------------------------------------------------


class ConstitutionalForwarder:
    """HTTP forwarder for signed :class:`LedgerBlock` instances.

    Always persists locally; forwards to carl.camp only when consent
    allows AND a bearer token is resolvable. Errors during forward
    NEVER disrupt local persistence — that is the source of truth.

    Parameters
    ----------
    base_url
        Override the carl.camp base URL. Defaults to ``CARL_CAMP_BASE``
        env var, then ``https://carl.camp``.
    ledger_path
        Override the on-disk JSONL location. Defaults to
        ``~/.carl/constitutional_ledger.jsonl``. The directory is
        created with ``mkdir -p`` semantics on first construct.
    bearer_token_resolver
        Callable returning a ``str | None`` JWT. Defaults to
        :func:`_default_bearer_resolver` which reuses the
        ``CARL_CAMP_TOKEN`` → ``~/.carl/camp_token`` → ``LocalDB jwt``
        chain documented in CLAUDE.md.
    consent_check
        Callable returning a bool — ``True`` iff
        ``consent.telemetry`` is granted. Defaults to
        :func:`_default_consent_check`.
    http
        Optional pre-built ``httpx.Client`` for tests / custom timeout.
    chain
        Optional :class:`~carl_core.interaction.InteractionChain` to
        record EXTERNAL steps against (one per forward attempt). When
        omitted, no chain steps are recorded.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        ledger_path: Path | None = None,
        bearer_token_resolver: Callable[[], str | None] | None = None,
        consent_check: Callable[[], bool] | None = None,
        http: httpx.Client | None = None,
        chain: InteractionChain | None = None,
    ) -> None:
        env_base = os.environ.get("CARL_CAMP_BASE")
        self._base_url: str = (base_url or env_base or DEFAULT_CARL_CAMP_BASE).rstrip("/")
        self._ledger_path: Path = ledger_path or DEFAULT_LEDGER_PATH
        self._bearer_resolver: Callable[[], str | None] = (
            bearer_token_resolver or _default_bearer_resolver
        )
        self._consent_check: Callable[[], bool] = (
            consent_check or _default_consent_check
        )
        self._http: httpx.Client = http or httpx.Client(timeout=httpx.Timeout(15.0))
        self._chain: InteractionChain | None = chain
        # Ensure parent dir exists with default permissions; the file
        # itself is created with mode 0600 on first append.
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ledger_path(self) -> Path:
        """The on-disk JSONL path. Public for inspection / tests."""
        return self._ledger_path

    @property
    def base_url(self) -> str:
        """Resolved carl.camp base URL (no trailing slash)."""
        return self._base_url

    def forward_block(self, block: LedgerBlock) -> dict[str, Any]:
        """Persist locally + (optionally) POST to ``/api/ledger/append``.

        Always returns a dict with the keys:

        * ``persisted: bool`` — did the local JSONL append succeed.
          ``True`` on the happy path; the call raises if local
          persistence itself fails (the caller's safety net is gone in
          that case and we want loud failure).
        * ``forwarded: bool`` — was the HTTP POST attempted AND
          accepted by carl.camp.
        * ``response: dict | None`` — parsed JSON response body, when
          carl.camp returned ``application/json``.
        * ``reason: str | None`` — when ``forwarded=False``, a short
          machine-friendly tag (``consent_off`` / ``no_bearer`` /
          ``network: ...`` / ``http_<status>: <body[:200]>``).

        The block is NOT re-signed in transit. The signature ships
        as-is in the body so carl.camp can re-verify against the
        same signing key chain that produced the local persist.
        """
        # 1. Always persist locally first. Failure here propagates —
        #    we'd rather surface a disk problem than silently lose audit.
        self._append_local(block)

        # 2. Forward gate: consent must be ON.
        try:
            consent_on = bool(self._consent_check())
        except Exception:
            logger.exception("constitutional.forward: consent check failed")
            consent_on = False

        if not consent_on:
            self._record_step(
                "constitutional.forward",
                output={"forwarded": False, "reason": "consent_off"},
                success=True,
            )
            return {
                "persisted": True,
                "forwarded": False,
                "response": None,
                "reason": "consent_off",
            }

        # 3. Bearer gate: no token = silent no-op (FREE / unauth path).
        try:
            token = self._bearer_resolver()
        except Exception:
            logger.exception("constitutional.forward: bearer resolver failed")
            token = None
        if not token:
            self._record_step(
                "constitutional.forward",
                output={"forwarded": False, "reason": "no_bearer"},
                success=True,
            )
            return {
                "persisted": True,
                "forwarded": False,
                "response": None,
                "reason": "no_bearer",
            }

        # 4. Build payload + POST.
        url = f"{self._base_url}{APPEND_PATH}"
        body: dict[str, Any] = {
            "block": block.to_dict(),
            "signature_hex": block.signature.hex(),
        }

        try:
            resp = self._http.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            reason = f"network: {type(exc).__name__}"
            self._record_step(
                "constitutional.forward",
                output={"forwarded": False, "reason": reason},
                success=False,
            )
            return {
                "persisted": True,
                "forwarded": False,
                "response": None,
                "reason": reason,
            }

        status = int(getattr(resp, "status_code", 0) or 0)
        if status >= 400:
            text_excerpt = ""
            with contextlib.suppress(Exception):
                text_excerpt = resp.text[:200]
            reason = f"http_{status}: {text_excerpt}"
            self._record_step(
                "constitutional.forward",
                output={"forwarded": False, "reason": reason, "status": status},
                success=False,
            )
            return {
                "persisted": True,
                "forwarded": False,
                "response": None,
                "reason": reason,
            }

        # 5. Mark this block acked in the local JSONL (best-effort).
        sig_hex = block.signature.hex()
        try:
            self._mark_acked(sig_hex)
        except Exception:
            # Acking is a convenience; replay_pending() also tolerates
            # missing ack lines by re-POSTing — at worst the carl.camp
            # idempotency uniq-index dedupes.
            logger.exception(
                "constitutional.forward: ack annotation failed for sig=%s",
                sig_hex[:12],
            )

        parsed: dict[str, Any] | None = None
        ctype = resp.headers.get("content-type", "") if resp.headers else ""
        if ctype.startswith("application/json"):
            try:
                raw_any: Any = resp.json()
            except Exception:
                raw_any = None
            if isinstance(raw_any, dict):
                parsed = cast(dict[str, Any], raw_any)

        self._record_step(
            "constitutional.forward",
            output={"forwarded": True, "block_id": block.block_id, "status": status},
            success=True,
        )
        return {
            "persisted": True,
            "forwarded": True,
            "response": parsed,
            "reason": None,
        }

    def replay_pending(self) -> int:
        """Walk the local JSONL, retry forward for any non-acked entries.

        Returns the number of blocks newly acked. ``0`` is non-error:
        an empty queue, missing ledger file, missing bearer, or
        consent-off all return ``0``.

        Acks are computed in-memory then a single rewrite collapses
        all newly-acked lines at the end — never rewrites mid-loop.
        """
        if not self._ledger_path.exists():
            return 0

        try:
            consent_on = bool(self._consent_check())
        except Exception:
            consent_on = False
        if not consent_on:
            return 0

        try:
            token = self._bearer_resolver()
        except Exception:
            token = None
        if not token:
            return 0

        records: list[dict[str, Any]] = self._read_all_records()
        if not records:
            return 0

        url = f"{self._base_url}{APPEND_PATH}"
        new_acks: set[str] = set()
        new_ack_count = 0

        for rec in records:
            if rec.get("acked") is True:
                continue
            block_dict = rec.get("block")
            sig_hex = rec.get("signature_hex")
            if not isinstance(block_dict, dict) or not isinstance(sig_hex, str):
                continue

            payload: dict[str, Any] = {
                "block": cast(dict[str, Any], block_dict),
                "signature_hex": sig_hex,
            }
            try:
                resp = self._http.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
            except httpx.HTTPError:
                # Skip this one; it'll get another shot on the next replay.
                continue

            status = int(getattr(resp, "status_code", 0) or 0)
            if status < 400:
                new_acks.add(sig_hex)
                new_ack_count += 1
            # 400-class is left non-acked too — caller may inspect logs
            # to determine whether the block is genuinely rejected (in
            # which case manual remediation is the next step).

        if new_acks:
            for rec in records:
                sig = rec.get("signature_hex")
                if isinstance(sig, str) and sig in new_acks:
                    rec["acked"] = True
            self._rewrite_records(records)

        return new_ack_count

    # ------------------------------------------------------------------
    # Local-persist primitives
    # ------------------------------------------------------------------

    def _append_local(self, block: LedgerBlock) -> None:
        """Append a JSON line for *block* to the local ledger file.

        Rotates if the file would exceed :data:`ROTATION_BYTES` AFTER
        the append. Uses ``os.open`` with explicit mode so the file is
        never world-readable, even at unusual umasks.
        """
        # Rotate first if we're already at-or-over the ceiling — keeps
        # the active file <= ROTATION_BYTES at the moment of write.
        if (
            self._ledger_path.exists()
            and self._ledger_path.stat().st_size >= ROTATION_BYTES
        ):
            self._rotate()

        record: dict[str, Any] = {
            "at": datetime.now(tz=timezone.utc).isoformat(),
            "block": block.to_dict(),
            "signature_hex": block.signature.hex(),
            "acked": False,
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"

        # Create with strict mode if missing; subsequent appends keep
        # the existing mode. ``O_APPEND`` ensures atomic append on POSIX.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(self._ledger_path), flags, LEDGER_FILE_MODE)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    def _rotate(self) -> None:
        """Rotate ``ledger.jsonl`` → ``ledger.jsonl.1`` → ``.2`` → ``.3``.

        Drops anything that would exceed :data:`MAX_ROTATIONS`. The
        active file becomes ``.1`` after rotation; subsequent appends
        recreate the active file.
        """
        # Walk from highest to lowest; promote each archive one slot.
        # Anything at MAX_ROTATIONS gets unlinked.
        for i in range(MAX_ROTATIONS, 0, -1):
            src = self._archive_path(i)
            if i == MAX_ROTATIONS:
                if src.exists():
                    with contextlib.suppress(OSError):
                        src.unlink()
                continue
            dst = self._archive_path(i + 1)
            if src.exists():
                with contextlib.suppress(OSError):
                    src.rename(dst)

        # Move the live file to .1.
        target = self._archive_path(1)
        if self._ledger_path.exists():
            with contextlib.suppress(OSError):
                shutil.move(str(self._ledger_path), str(target))

    def _archive_path(self, n: int) -> Path:
        """Return the archive path for slot *n* (``ledger.jsonl.N``)."""
        return self._ledger_path.with_name(f"{self._ledger_path.name}.{n}")

    def _mark_acked(self, signature_hex: str) -> None:
        """Best-effort: rewrite the file with ``acked: true`` on the
        line whose ``signature_hex`` matches.

        Acceptable cost given typical write rate of <10 blocks/sec; if
        write rates ever climb we'd switch to a sidecar ack journal.
        """
        records = self._read_all_records()
        if not records:
            return
        changed = False
        for rec in records:
            if rec.get("signature_hex") == signature_hex and rec.get("acked") is not True:
                rec["acked"] = True
                changed = True
        if changed:
            self._rewrite_records(records)

    def _read_all_records(self) -> list[dict[str, Any]]:
        """Parse the JSONL file into a list of records.

        Tolerant: malformed lines are skipped (logged at debug). The
        file may be empty or absent — both yield ``[]``.
        """
        if not self._ledger_path.exists():
            return []
        try:
            with self._ledger_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                obj_any: Any = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                logger.debug("constitutional.forward: skipping malformed JSONL line")
                continue
            if isinstance(obj_any, dict):
                out.append(cast(dict[str, Any], obj_any))
        return out

    def _rewrite_records(self, records: list[dict[str, Any]]) -> None:
        """Atomic file replace — write all records to a tmp file,
        chmod, then ``os.replace`` over the live path.

        The tmp file is created with mode 0600 BEFORE data is written.
        """
        path = self._ledger_path
        path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(8)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{token}")

        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, LEDGER_FILE_MODE)
        try:
            buf: list[str] = []
            for rec in records:
                buf.append(json.dumps(rec, separators=(",", ":")))
                buf.append("\n")
            os.write(fd, "".join(buf).encode("utf-8"))
        finally:
            os.close(fd)

        try:
            os.chmod(tmp, LEDGER_FILE_MODE)
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------
    # Audit chain helpers
    # ------------------------------------------------------------------

    def _record_step(
        self,
        name: str,
        *,
        output: dict[str, Any] | None = None,
        success: bool = True,
    ) -> None:
        """Best-effort step record; failures never propagate to caller."""
        if self._chain is None:
            return
        try:
            self._chain.record(
                ActionType.EXTERNAL,
                name,
                output=output,
                success=success,
            )
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def aclose(self) -> None:
        """Close the underlying httpx client. Idempotent."""
        with contextlib.suppress(Exception):
            self._http.close()


# ---------------------------------------------------------------------------
# Default-resolver helpers (lazy imports so the module stays light)
# ---------------------------------------------------------------------------


def _default_bearer_resolver() -> str | None:
    """Reuse the canonical bearer-token chain from
    :func:`carl_studio.tier._resolve_bearer_token_for_verify`.

    Order: ``CARL_CAMP_TOKEN`` env → ``~/.carl/camp_token`` →
    ``LocalDB.get_auth("jwt")``. Lazy-imported so this module stays
    cheap to import.
    """
    try:
        from carl_studio.tier import (
            _resolve_bearer_token_for_verify,  # pyright: ignore[reportPrivateUsage]
        )

        return _resolve_bearer_token_for_verify()
    except Exception:
        logger.exception("constitutional.forward: bearer resolver failed")
        return None


def _default_consent_check() -> bool:
    """Read ``consent.telemetry`` via :class:`carl_studio.consent.ConsentManager`.

    Returns ``False`` on any exception (fail-closed).
    """
    try:
        from carl_studio.consent import ConsentManager

        return ConsentManager().is_granted("telemetry")
    except Exception:
        logger.exception("constitutional.forward: consent check failed")
        return False


# ---------------------------------------------------------------------------
# Singleton wiring
# ---------------------------------------------------------------------------

_DEFAULT_FORWARDER: ConstitutionalForwarder | None = None


def install_default_forwarder(
    *, base_url: str | None = None
) -> ConstitutionalForwarder:
    """Construct (or return) the process-wide default forwarder.

    Idempotent: subsequent calls return the same instance. Useful for
    threading the same forwarder through every ``evaluate_action(...)``
    call without re-opening an httpx client per block.
    """
    global _DEFAULT_FORWARDER
    if _DEFAULT_FORWARDER is None:
        _DEFAULT_FORWARDER = ConstitutionalForwarder(  # pyright: ignore[reportConstantRedefinition]
            base_url=base_url
        )
    return _DEFAULT_FORWARDER


def reset_default_forwarder() -> None:
    """Test helper: clear the singleton + close its client."""
    global _DEFAULT_FORWARDER
    if _DEFAULT_FORWARDER is not None:
        with contextlib.suppress(Exception):
            _DEFAULT_FORWARDER.aclose()
    _DEFAULT_FORWARDER = None  # pyright: ignore[reportConstantRedefinition]


__all__ = [
    "APPEND_PATH",
    "ConstitutionalForwarder",
    "DEFAULT_CARL_CAMP_BASE",
    "DEFAULT_LEDGER_PATH",
    "FORWARD_FAILED_CODE",
    "LEDGER_FILE_MODE",
    "MAX_ROTATIONS",
    "ROTATION_BYTES",
    "install_default_forwarder",
    "reset_default_forwarder",
]
