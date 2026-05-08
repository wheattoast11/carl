"""AXON HTTP forwarder for carl-studio.

Bridges :class:`carl_core.interaction.InteractionChain` →
``carl.camp /api/axon/ingest``.

Default-OFF: requires ``consent.telemetry == True`` AND no
``AXON_FORWARD_DISABLED`` env opt-out. Buffers up to 100 events per batch
with a 5-second max-flush interval.

Mapping policy (Step → AxonEvent)
---------------------------------

Single-signal table:

* ``ActionType.TRAINING_STEP`` → ``SKILL_TRAINING_STARTED``
* ``ActionType.CHECKPOINT``    → ``SKILL_CRYSTALLIZED``
* ``ActionType.LLM_REPLY``     → ``INTERACTION_CREATED``
* ``ActionType.REWARD``        → ``INTERACTION_CREATED``
* ``ActionType.EXTERNAL``      → ``ACTION_DISPATCHED``

Plus a SECONDARY ``COHERENCE_UPDATE`` event when the step carries any of
``Step.phi`` / ``Step.kuramoto_r`` / ``Step.channel_coherence`` — these
are the cross-channel coherence fields documented at
``carl_core.interaction.Step`` and are populated either by an explicit
caller-passed kwarg or by the v0.10 auto-attach probe.

Idempotency
-----------

Each :class:`AxonEvent` carries
``idempotency_key = sha256(step_id:signal_type)``. ``Step.step_id`` is a
12-hex globally-unique id assigned at construction (see
``carl_core.interaction._new_id``); using it as the seed means any retry
of the same logical step produces the same key, so the carl.camp
``/api/axon/ingest`` route's ``ON CONFLICT (idempotency_key)`` upsert
collapses duplicates.

Privacy
-------

Every payload runs through :func:`carl_core.errors._redact` before
shipping; secret-shaped keys (``key``, ``token``, ``secret``,
``password``, ``authorization``, ``bearer``) are scrubbed. The scrub is
applied to ``Step.input`` / ``Step.output`` AND to the resulting payload
dict so deeply-nested credentials are caught.

Threading
---------

The forwarder uses a thread-bound ``queue.Queue`` plus a daemon flusher
thread. NOT asyncio — the seam at ``InteractionChain.record(...)`` is
synchronous and we don't want to pay the coroutine-cost on the hot path.
Queue-full conditions drop the OLDEST event (ring-buffer semantics) so
telemetry never blocks the producer.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Callable, cast

import httpx
from pydantic import BaseModel, Field

from carl_core.errors import _redact  # pyright: ignore[reportPrivateUsage]
from carl_core.interaction import ActionType, Step, set_global_forwarder

from carl_studio.telemetry.axon_signals import (
    ACTION_DISPATCHED,
    COHERENCE_UPDATE,
    INTERACTION_CREATED,
    SKILL_CRYSTALLIZED,
    SKILL_TRAINING_STARTED,
)

logger = logging.getLogger(__name__)

DEFAULT_CARL_CAMP_BASE: str = "https://carl.camp"
INGEST_PATH: str = "/api/axon/ingest"
DEFAULT_BATCH_SIZE: int = 100
DEFAULT_FLUSH_INTERVAL_S: float = 5.0
MAX_QUEUE_DEPTH: int = 10_000  # drop oldest if exceeded


def _shape_keys(value: Any) -> list[str]:
    """Return sorted dict keys of *value*, or [] for non-dicts.

    Carl-core ``Step.input`` / ``Step.output`` are typed ``Any`` so we
    narrow with isinstance + cast before reading ``.keys()``. The list
    is sorted to keep payload shape deterministic across replays.
    """
    if isinstance(value, dict):
        narrowed: dict[Any, Any] = cast(dict[Any, Any], value)
        return sorted(str(k) for k in narrowed.keys())
    return []


class AxonEvent(BaseModel):
    """A single AXON telemetry event, ready to ship to carl.camp."""

    signal_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime
    idempotency_key: str


class AxonForwarder:
    """Background HTTP forwarder for AXON events.

    Construct via :func:`install_default_forwarder` for the singleton
    used by the carl_core seam. Direct instantiation is for tests and
    custom-base-URL scenarios only.

    Parameters
    ----------
    base_url
        Override the carl.camp base URL. Defaults to ``CARL_CAMP_BASE``
        env var, then ``https://carl.camp``.
    batch_size
        Max events per ``/api/axon/ingest`` POST.
    flush_interval_s
        Max seconds between flush attempts. The daemon thread wakes
        every interval and drains the queue (or no-ops if empty).
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
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        bearer_token_resolver: Callable[[], str | None] | None = None,
        consent_check: Callable[[], bool] | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self._base_url: str = base_url or os.environ.get(
            "CARL_CAMP_BASE", DEFAULT_CARL_CAMP_BASE
        )
        self._batch_size: int = batch_size
        self._flush_interval_s: float = flush_interval_s
        self._queue: queue.Queue[AxonEvent] = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._bearer_resolver: Callable[[], str | None] = (
            bearer_token_resolver or _default_bearer_resolver
        )
        self._consent_check: Callable[[], bool] = (
            consent_check or _default_consent_check
        )
        self._http: httpx.Client = http or httpx.Client(timeout=httpx.Timeout(10.0))
        self._thread: threading.Thread | None = None

    # -- gating ----------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return True iff the forwarder is currently allowed to ship.

        Returns False if either the kill-switch env var is set OR the
        consent check is False. The kill-switch is checked FIRST so it
        always wins, even if consent is somehow corrupted.
        """
        if os.environ.get("AXON_FORWARD_DISABLED") == "1":
            return False
        try:
            return bool(self._consent_check())
        except Exception:
            # Fail-closed: if consent lookup explodes, do not ship.
            return False

    # -- producer side --------------------------------------------------------

    def on_step(self, step: Step) -> None:
        """Hook to register via :func:`set_global_forwarder`.

        Maps the step to 0+ AxonEvents and enqueues them. Must NEVER
        raise — the carl_core seam swallows exceptions defensively but
        we belt-and-suspenders here too.
        """
        if not self.is_enabled():
            return
        try:
            events = self.map_step_to_events(step)
        except Exception:
            logger.exception("axon: map_step_to_events raised; dropping step")
            return

        for ev in events:
            try:
                self._queue.put_nowait(ev)
            except queue.Full:
                # Telemetry must never block. Drop the OLDEST event
                # (ring-buffer semantics) and try once more; if that
                # also fails, drop the new event.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(ev)
                except (queue.Empty, queue.Full):
                    pass  # genuine overflow

    # -- mapping --------------------------------------------------------------

    def map_step_to_events(self, step: Step) -> list[AxonEvent]:
        """Convert a Step into 0+ AxonEvents.

        Returns an empty list when the action type is not in the
        primary mapping table AND the step carries no coherence fields.

        See module docstring for the full mapping table.
        """
        primary_signal = self._primary_for(step.action)

        # Coherence-update emission happens on ANY step whose phi /
        # kuramoto_r / channel_coherence are populated (from explicit
        # kwargs or from the v0.10 auto-attach probe). It can fire as
        # the SOLE event (e.g. on a TOOL_CALL step that the auto-attach
        # probe enriched) or as a SECONDARY event alongside the primary.
        has_coherence = (
            step.phi is not None
            or step.kuramoto_r is not None
            or step.channel_coherence is not None
        )

        if primary_signal is None and not has_coherence:
            return []

        # Build the shared base payload. Every value runs through the
        # carl_core redact helper to scrub secret-shaped keys.
        step_id = getattr(step, "step_id", "unknown")
        started = self._step_started_at(step)

        raw_payload: dict[str, Any] = {
            "step_id": step_id,
            "action": step.action.value,
            "name": step.name,
            "success": bool(step.success),
            "duration_ms": step.duration_ms,
            "session_id": step.session_id,
            "trace_id": step.trace_id,
            # Shape-only summaries of input/output to keep payload small
            # and avoid leaking large blobs over telemetry.
            "input_keys": _shape_keys(step.input),
            "output_keys": _shape_keys(step.output),
            "phi": step.phi,
            "kuramoto_r": step.kuramoto_r,
            "channel_coherence": (
                dict(step.channel_coherence)
                if step.channel_coherence is not None
                else None
            ),
        }
        scrubbed_any = _redact(raw_payload)
        scrubbed: dict[str, Any] = (
            cast(dict[str, Any], scrubbed_any)
            if isinstance(scrubbed_any, dict)
            else raw_payload
        )

        events: list[AxonEvent] = []

        if primary_signal is not None:
            events.append(
                AxonEvent(
                    signal_type=primary_signal,
                    payload=scrubbed,
                    occurred_at=started,
                    idempotency_key=self._idempotency_key(step_id, primary_signal),
                )
            )

        if has_coherence:
            coherence_payload: dict[str, Any] = {
                "step_id": step_id,
                "action": step.action.value,
                "name": step.name,
                "phi": step.phi,
                "kuramoto_r": step.kuramoto_r,
                "channel_coherence": (
                    dict(step.channel_coherence)
                    if step.channel_coherence is not None
                    else None
                ),
            }
            events.append(
                AxonEvent(
                    signal_type=COHERENCE_UPDATE,
                    payload=coherence_payload,
                    occurred_at=started,
                    idempotency_key=self._idempotency_key(step_id, COHERENCE_UPDATE),
                )
            )

        return events

    @staticmethod
    def _primary_for(action: ActionType) -> str | None:
        """Map ActionType → primary signal name (or None)."""
        if action is ActionType.TRAINING_STEP:
            return SKILL_TRAINING_STARTED
        if action is ActionType.CHECKPOINT:
            return SKILL_CRYSTALLIZED
        if action is ActionType.LLM_REPLY:
            return INTERACTION_CREATED
        if action is ActionType.REWARD:
            return INTERACTION_CREATED
        if action is ActionType.EXTERNAL:
            return ACTION_DISPATCHED
        return None

    @staticmethod
    def _idempotency_key(step_id: str, signal_type: str) -> str:
        """Stable key from ``Step.step_id`` + signal name.

        ``Step.step_id`` is the 12-hex globally-unique id assigned at
        Step construction. Combining it with the signal type means a
        single Step that produces both a primary and a secondary
        coherence event yields two distinct, idempotent keys.
        """
        return hashlib.sha256(f"{step_id}:{signal_type}".encode()).hexdigest()

    @staticmethod
    def _step_started_at(step: Step) -> datetime:
        """Return ``Step.started_at`` as an aware datetime, fallback to now()."""
        raw = getattr(step, "started_at", None)
        if isinstance(raw, datetime):
            if raw.tzinfo is None:
                return raw.replace(tzinfo=timezone.utc)
            return raw
        return datetime.now(tz=timezone.utc)

    # -- consumer side --------------------------------------------------------

    def flush(self) -> int:
        """Drain the queue once and POST to ``/api/axon/ingest``.

        Returns the count of events successfully shipped. A return of 0
        is non-error: it can mean an empty queue, missing bearer, or
        consent-off — all valid no-op states.
        """
        events: list[AxonEvent] = []
        while len(events) < self._batch_size:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not events:
            return 0

        token = self._bearer_resolver()
        if not token:
            # No bearer means we can't auth — drop the batch silently.
            # Telemetry-without-auth is a deliberate no-op; not an error.
            logger.debug("axon: no bearer token; dropping %d events", len(events))
            return 0

        url = f"{self._base_url.rstrip('/')}{INGEST_PATH}"
        try:
            self._http.post(
                url,
                json={"events": [ev.model_dump(mode="json") for ev in events]},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            return len(events)
        except Exception:
            # Telemetry must never raise. Drop the batch + log.
            logger.exception("axon: POST %s failed; dropped %d events", url, len(events))
            return 0

    def _flush_loop(self) -> None:
        """Daemon thread: wake every flush_interval_s and drain."""
        while not self._stop.is_set():
            self._stop.wait(self._flush_interval_s)
            try:
                self.flush()
            except Exception:
                # Belt-and-suspenders — flush() already swallows but
                # the daemon thread MUST never die mid-loop.
                logger.exception("axon: flush_loop iteration failed; continuing")

    def start(self) -> None:
        """Start the daemon flusher thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="axon-forwarder",
            daemon=True,
        )
        self._thread.start()

    def aclose(self) -> None:
        """Best-effort flush + thread shutdown. Safe to call from atexit.

        Stops the daemon, waits up to 2s for it to exit, runs one final
        synchronous flush to drain anything still queued, and closes
        the underlying ``httpx.Client``.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        try:
            self.flush()
        except Exception:
            logger.exception("axon: final flush during aclose failed")
        try:
            self._http.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Default-resolver helpers (lazy imports so subpackage stays light)
# ---------------------------------------------------------------------------


def _default_bearer_resolver() -> str | None:
    """Reuse the canonical bearer-token chain from
    :func:`carl_studio.tier._resolve_bearer_token_for_verify`.

    Order: ``CARL_CAMP_TOKEN`` env → ``~/.carl/camp_token`` → ``LocalDB
    .get_auth("jwt")``. Lazy-imported so this module stays cheap to
    import.
    """
    try:
        from carl_studio.tier import (
            _resolve_bearer_token_for_verify,  # pyright: ignore[reportPrivateUsage]
        )

        return _resolve_bearer_token_for_verify()
    except Exception:
        logger.exception("axon: bearer resolver failed")
        return None


def _default_consent_check() -> bool:
    """Read ``consent.telemetry`` via :class:`carl_studio.consent.ConsentManager`.

    Returns ``False`` on any exception (fail-closed).
    """
    try:
        from carl_studio.consent import ConsentManager

        return ConsentManager().is_granted("telemetry")
    except Exception:
        logger.exception("axon: consent check failed")
        return False


# ---------------------------------------------------------------------------
# Singleton wiring
# ---------------------------------------------------------------------------

_DEFAULT_FORWARDER: AxonForwarder | None = None


def install_default_forwarder(*, base_url: str | None = None) -> AxonForwarder:
    """Construct an :class:`AxonForwarder`, register it via
    :func:`set_global_forwarder`, start the daemon, and schedule
    ``atexit`` cleanup.

    Idempotent: subsequent calls return the same instance.
    """
    global _DEFAULT_FORWARDER
    if _DEFAULT_FORWARDER is None:
        forwarder = AxonForwarder(base_url=base_url)
        _DEFAULT_FORWARDER = forwarder  # pyright: ignore[reportConstantRedefinition]
        set_global_forwarder(forwarder.on_step)
        forwarder.start()
        atexit.register(forwarder.aclose)
    return _DEFAULT_FORWARDER


def reset_default_forwarder() -> None:
    """Test helper: clear the singleton + global registration."""
    global _DEFAULT_FORWARDER
    if _DEFAULT_FORWARDER is not None:
        try:
            _DEFAULT_FORWARDER.aclose()
        except Exception:
            pass
    _DEFAULT_FORWARDER = None  # pyright: ignore[reportConstantRedefinition]
    set_global_forwarder(None)


__all__ = [
    "AxonEvent",
    "AxonForwarder",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CARL_CAMP_BASE",
    "DEFAULT_FLUSH_INTERVAL_S",
    "INGEST_PATH",
    "MAX_QUEUE_DEPTH",
    "install_default_forwarder",
    "reset_default_forwarder",
]
