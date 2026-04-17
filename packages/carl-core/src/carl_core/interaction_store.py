"""InteractionStore — JSONL append-only persistence for InteractionChain traces.

Each chain is stored at ``<root>/<chain_id>.jsonl`` where the first line is a
header row (``{"chain_id", "started_at", "context"}``) and subsequent lines are
one :class:`carl_core.interaction.Step` per line (the output of ``Step.to_dict``).

Appends are serialized with :func:`fcntl.flock` on POSIX and
:mod:`msvcrt.locking` on Windows so that multiple threads or processes writing
to the same chain do not interleave partial records. Reads lock in shared mode
where the platform supports it.

The store depends on the standard library only — no external deps.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from carl_core.interaction import ActionType, InteractionChain, Step

__all__ = ["InteractionStore"]


# ---------------------------------------------------------------------------
# Cross-platform file lock helpers
# ---------------------------------------------------------------------------


if sys.platform == "win32":  # pragma: no cover - platform specific
    import msvcrt

    def _lock_exclusive(fileno: int) -> None:
        # LK_LOCK blocks for ~10s then raises; retry loop handles longer waits.
        deadline = time.monotonic() + 30.0
        while True:
            try:
                msvcrt.locking(fileno, msvcrt.LK_LOCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def _lock_shared(fileno: int) -> None:
        _lock_exclusive(fileno)

    def _unlock(fileno: int) -> None:
        try:
            msvcrt.locking(fileno, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock_exclusive(fileno: int) -> None:
        fcntl.flock(fileno, fcntl.LOCK_EX)

    def _lock_shared(fileno: int) -> None:
        fcntl.flock(fileno, fcntl.LOCK_SH)

    def _unlock(fileno: int) -> None:
        try:
            fcntl.flock(fileno, fcntl.LOCK_UN)
        except OSError:
            pass


# Per-process in-memory lock: avoids threads contending on the same fd.
_PROCESS_LOCK = threading.Lock()


class InteractionStore:
    """JSONL append-only store for InteractionChain traces.

    Path convention: ``<root>/<chain_id>.jsonl``. The store never truncates
    existing files: :meth:`append` opens in append mode and writes a single
    line under an exclusive advisory lock, so concurrent appends from multiple
    threads or processes produce a well-formed JSONL file.

    The first append for a new chain also writes the chain header row so that
    :meth:`load` can recover chain context even when the caller only has step
    rows.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _validate_chain_id(self, chain_id: str) -> str:
        if not chain_id or not isinstance(chain_id, str):
            raise ValueError(f"chain_id must be a non-empty string, got {chain_id!r}")
        # Disallow path separators and nul bytes to prevent path traversal.
        if any(c in chain_id for c in ("/", "\\", "\x00", "..")):
            raise ValueError(f"chain_id contains illegal characters: {chain_id!r}")
        return chain_id

    def path_for(self, chain_id: str) -> Path:
        cid = self._validate_chain_id(chain_id)
        return self.root / f"{cid}.jsonl"

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        chain_id: str,
        step: Step,
        *,
        header: dict[str, Any] | None = None,
    ) -> Path:
        """Append a step (and optionally a header) to ``<chain_id>.jsonl``.

        If the file does not yet exist, writes a header row first. The
        caller can pass ``header`` to override the default (``chain_id`` +
        ``started_at`` only). Returns the resolved path.
        """
        self._ensure_root()
        path = self.path_for(chain_id)
        payload = json.dumps(step.to_dict(), ensure_ascii=False)

        with _PROCESS_LOCK:
            needs_header = not path.exists() or path.stat().st_size == 0
            with open(path, "ab") as fh:
                try:
                    _lock_exclusive(fh.fileno())
                    if needs_header:
                        hdr = header if header is not None else {
                            "chain_id": chain_id,
                            "started_at": step.started_at.isoformat(),
                            "context": {},
                        }
                        fh.write((json.dumps(hdr, ensure_ascii=False) + "\n").encode("utf-8"))
                    fh.write((payload + "\n").encode("utf-8"))
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    _unlock(fh.fileno())
        return path

    def append_chain(self, chain: InteractionChain) -> Path:
        """Persist the entire chain atomically (header + every step).

        Overwrites any existing file for this chain_id. Useful at the end of
        a run when you prefer a single snapshot over streaming appends.
        """
        self._ensure_root()
        path = self.path_for(chain.chain_id)
        with _PROCESS_LOCK:
            with open(path, "wb") as fh:
                try:
                    _lock_exclusive(fh.fileno())
                    fh.write((chain.to_jsonl() + "\n").encode("utf-8"))
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    _unlock(fh.fileno())
        return path

    def load(self, chain_id: str) -> InteractionChain:
        """Reconstruct a chain from its JSONL file.

        Missing file -> returns an empty chain with the requested id so
        callers can treat it like a fresh chain without special-casing.
        """
        cid = self._validate_chain_id(chain_id)
        path = self.path_for(cid)
        if not path.exists():
            return InteractionChain(chain_id=cid)

        header: dict[str, Any] | None = None
        steps_raw: list[dict[str, Any]] = []

        with _PROCESS_LOCK:
            with open(path, "rb") as fh:
                try:
                    _lock_shared(fh.fileno())
                    data = fh.read()
                finally:
                    _unlock(fh.fileno())

        text = data.decode("utf-8", errors="replace")
        for lineno, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                # Skip corrupt lines but keep going — partial traces are
                # better than nothing and we rebuild via ``from_dict``.
                continue
            if not isinstance(parsed, dict):
                continue
            if lineno == 0 and "action" not in parsed:
                header = parsed
                continue
            if "action" in parsed:
                steps_raw.append(parsed)

        payload: dict[str, Any] = {}
        if header is not None:
            payload.update(header)
        payload["chain_id"] = cid
        payload["steps"] = steps_raw
        return InteractionChain.from_dict(payload)

    def list_chains(self) -> list[str]:
        """Return every chain_id found under ``root``. Empty if root missing."""
        if not self.root.exists():
            return []
        out: list[str] = []
        for entry in sorted(self.root.iterdir()):
            if entry.is_file() and entry.suffix == ".jsonl":
                out.append(entry.stem)
        return out

    def delete(self, chain_id: str) -> bool:
        """Remove a chain file. Returns ``True`` when a file was deleted."""
        path = self.path_for(chain_id)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False


# Re-export for convenience.
__all__ += ["ActionType", "InteractionChain", "Step"]
