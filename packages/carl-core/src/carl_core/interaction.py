"""InteractionChain — typed trace of every interaction Carl participates in.

Every CLI command, tool call, LLM reply, permission gate, and external call
emits a `Step` into an `InteractionChain`. The chain is the raw material for:

  - Replay and debug (`carl replay <session>`)
  - Training signal (which interaction shapes drove success)
  - Contract / witness trace (see `carl_studio.contract`)

This primitive lives in carl-core so it's a dependency-free lingua franca
across every Carl package. Only depends on the standard library.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Enumerates the shapes of interactions we trace."""

    USER_INPUT = "user_input"         # text typed by user in chat or CLI
    TOOL_CALL = "tool_call"           # CARLAgent calls a registered tool
    LLM_REPLY = "llm_reply"           # model streamed a reply
    CLI_CMD = "cli_cmd"               # top-level CLI command dispatch
    GATE = "gate"                     # credential / permission prompt
    EXTERNAL = "external"             # HTTP request, file write, subprocess
    PAYMENT = "payment"               # x402 / wallet payment flow
    TRAINING_STEP = "training_step"   # periodic training progress marker
    EVAL_PHASE = "eval_phase"         # eval phase start/end boundary
    REWARD = "reward"                 # reward-aggregation snapshot
    CHECKPOINT = "checkpoint"         # trainer checkpoint / model save
    MEMORY_READ = "memory_read"       # memory recall (WORKING/LONG retrieval)
    MEMORY_WRITE = "memory_write"     # memory commit (write to any layer)
    HEARTBEAT_CYCLE = "heartbeat_cycle"  # full sticky-note cycle boundary (start/end)
    STICKY_NOTE = "sticky_note"       # note append / dequeue / status transition


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Step:
    """One atomic unit in an InteractionChain.

    Attributes
    ----------
    action
        The shape of this interaction. See `ActionType`.
    name
        Short human-readable label, e.g. ``"require:ANTHROPIC_API_KEY"``,
        ``"carl chat"``, ``"ingest_source"``.
    input
        What went in (prompt string, args dict, credential name — never the
        credential value itself).
    output
        What came out (response text, tool result, ``True``/``False`` for gates).
    success
        Whether the step succeeded. Failures persist too — they are signal.
    started_at / duration_ms
        Timing, populated by the caller.
    parent_id
        Step id of the parent when nested or dispatched in parallel.
    step_id
        Stable id for this step (12-hex chars, generated here).
    phi
        Optional coherence-field snapshot for this step. When populated
        (by training / eval / chat-agent / connection hooks), a chain
        becomes a true cross-channel witness: :meth:`coherence_trajectory`
        reduces the whole chain to a phi-vs-step series.
    kuramoto_r
        Optional Kuramoto order-parameter (R) snapshot in [0, 1] — the
        phase-locking correlate of phi. Sourced from
        :meth:`carl_core.coherence_trace.CoherenceTrace.kuramoto_R` when
        available.
    channel_coherence
        When this step was emitted by a
        :class:`carl_core.connection.BaseConnection` transact, the
        connection's :class:`~carl_core.connection.ChannelCoherence` at
        event time, serialized via
        :meth:`~carl_core.connection.ChannelCoherence.as_dict`.
    """

    action: ActionType
    name: str
    input: Any = None
    output: Any = None
    success: bool = True
    started_at: datetime = field(default_factory=_utcnow)
    duration_ms: float | None = None
    parent_id: str | None = None
    step_id: str = field(default_factory=_new_id)
    session_id: str | None = None
    trace_id: str | None = None
    phi: float | None = None
    kuramoto_r: float | None = None
    channel_coherence: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action.value,
            "name": self.name,
            "input": _json_safe(self.input),
            "output": _json_safe(self.output),
            "success": self.success,
            "started_at": self.started_at.isoformat(),
            "duration_ms": self.duration_ms,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "phi": self.phi,
            "kuramoto_r": self.kuramoto_r,
            "channel_coherence": (
                dict(self.channel_coherence)
                if self.channel_coherence is not None
                else None
            ),
        }


@dataclass
class InteractionChain:
    """An ordered log of `Step`s plus a free-form context dict.

    The chain is append-only; once a step lands, it doesn't get mutated.
    Serialize with `to_dict()` / `to_jsonl()`, reload with `from_dict()`.
    """

    chain_id: str = field(default_factory=_new_id)
    started_at: datetime = field(default_factory=_utcnow)
    steps: list[Step] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    # -- core mutation -----------------------------------------------------

    def append(self, step: Step, result: Any = None) -> Step:
        """Append a step. If `result` is given, it overwrites `step.output`."""
        if result is not None:
            step.output = result
        self.steps.append(step)
        return step

    def record(
        self,
        action: ActionType,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        success: bool = True,
        duration_ms: float | None = None,
        parent_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        phi: float | None = None,
        kuramoto_r: float | None = None,
        channel_coherence: dict[str, float] | None = None,
    ) -> Step:
        """Build and append a step in one call — the common case.

        ``phi`` / ``kuramoto_r`` / ``channel_coherence`` are optional
        cross-channel coherence fields; see :class:`Step` for semantics.
        """
        step = Step(
            action=action,
            name=name,
            input=input,
            output=output,
            success=success,
            duration_ms=duration_ms,
            parent_id=parent_id,
            session_id=session_id,
            trace_id=trace_id,
            phi=phi,
            kuramoto_r=kuramoto_r,
            channel_coherence=channel_coherence,
        )
        self.steps.append(step)
        return step

    # -- query -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    def last(self) -> Step | None:
        return self.steps[-1] if self.steps else None

    def by_action(self, action: ActionType) -> list[Step]:
        return [s for s in self.steps if s.action == action]

    def success_rate(self) -> float:
        """Fraction of steps where success=True. Empty chain returns 1.0."""
        if not self.steps:
            return 1.0
        return sum(1 for s in self.steps if s.success) / len(self.steps)

    def coherence_trajectory(self) -> list[tuple[str, float | None]]:
        """Return the phi-vs-step series (timestamp, phi) across the whole chain.

        Steps without phi are included with ``None`` — caller decides
        whether to filter (visualization) or impute (training signal).
        This is what turns a flat log into a cross-channel coherence
        witness: the trajectory is well-defined across training,
        eval, chat, and connection transacts because every step can
        attach a coherence snapshot with the same units.
        """
        return [(s.started_at.isoformat(), s.phi) for s in self.steps]

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "started_at": self.started_at.isoformat(),
            "steps": [s.to_dict() for s in self.steps],
            "context": _json_safe(self.context),
        }

    def to_jsonl(self) -> str:
        """Serialize as newline-delimited JSON: header row + one row per step."""
        header = json.dumps({"chain_id": self.chain_id, "started_at": self.started_at.isoformat(), "context": _json_safe(self.context)})
        rows = [json.dumps(s.to_dict()) for s in self.steps]
        return "\n".join([header, *rows])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InteractionChain:
        chain = cls(
            chain_id=d.get("chain_id", _new_id()),
            context=dict(d.get("context", {})),
        )
        if started := d.get("started_at"):
            chain.started_at = datetime.fromisoformat(started)
        for raw in d.get("steps", []):
            cc_raw = raw.get("channel_coherence")
            cc: dict[str, float] | None
            if cc_raw is None:
                cc = None
            else:
                cc = {str(k): float(v) for k, v in cc_raw.items()}
            chain.steps.append(
                Step(
                    action=ActionType(raw["action"]),
                    name=raw["name"],
                    input=raw.get("input"),
                    output=raw.get("output"),
                    success=bool(raw.get("success", True)),
                    started_at=datetime.fromisoformat(raw["started_at"]) if raw.get("started_at") else _utcnow(),
                    duration_ms=raw.get("duration_ms"),
                    parent_id=raw.get("parent_id"),
                    step_id=raw.get("step_id", _new_id()),
                    session_id=raw.get("session_id"),
                    trace_id=raw.get("trace_id"),
                    phi=raw.get("phi"),
                    kuramoto_r=raw.get("kuramoto_r"),
                    channel_coherence=cc,
                )
            )
        return chain


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Coerce a value to a JSON-serializable form. Falls back to repr()."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return repr(value)
