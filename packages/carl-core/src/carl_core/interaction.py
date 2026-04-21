"""InteractionChain — typed trace of every interaction Carl participates in.

Every CLI command, tool call, LLM reply, permission gate, and external call
emits a `Step` into an `InteractionChain`. The chain is the raw material for:

  - Replay and debug (`carl replay <session>`)
  - Training signal (which interaction shapes drove success)
  - Contract / witness trace (see `carl_studio.contract`)

This primitive lives in carl-core so it's a dependency-free lingua franca
across every Carl package. Only depends on the standard library.

Secret redaction
----------------
The chain persists durably (``~/.carl/interactions/*.jsonl``) and can be
fed back into training, so raw secrets must never land in a step. The
``_json_safe`` helper below walks the step input/output/context payloads
and:

* Replaces dict values whose key name is secret-shaped
  (``*key*``, ``*token*``, ``*secret*``, ``*password*``, ``*authorization*``,
  ``*bearer*``) with the literal ``"<redacted>"``.
* Scrubs JWTs, OpenAI-style ``sk-...`` keys, Anthropic ``sk-ant-...`` keys,
  HF ``hf_...`` tokens, and EVM wallet addresses out of any string
  payload, preserving the first few characters so the shape is still
  debuggable.

This runs at ``to_dict`` / ``to_jsonl`` serialization time; in-memory
``Step`` objects retain the original values for programmatic use in the
same process.
"""
from __future__ import annotations

import json
import re
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
    GATE_CHECK = "gate_check"         # structured allow/deny predicate check (consent, tier, ...)
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
    # v0.16 secrets toolkit — zero-knowledge value-transfer lifecycle. Every
    # op emits a Step with a fingerprint, never the value. See
    # docs/v16_secrets_toolkit_design.md for the capability-security model.
    SECRET_MINT = "secret_mint"         # new value minted / stored into the vault
    SECRET_RESOLVE = "secret_resolve"   # privileged deref of a handle
    SECRET_REVOKE = "secret_revoke"     # handle invalidated
    CLIPBOARD_WRITE = "clipboard_write" # scoped clipboard bridge emit


# v0.10 W10: actions eligible for coherence auto-attach via a registered
# probe. These are the "informational" boundaries where a coherence
# snapshot carries the most training/eval value.
_AUTO_ATTACH_ACTIONS = frozenset({
    ActionType.LLM_REPLY,
    ActionType.TOOL_CALL,
    ActionType.TRAINING_STEP,
    ActionType.EVAL_PHASE,
    ActionType.REWARD,
})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _short_sha256(payload: str) -> str:
    """Return the first 12 hex chars of sha256(payload) — audit fingerprint.

    12 hex chars = 48 bits, plenty for collision resistance in an
    audit-trail context where the full payload is never re-derivable
    from the digest (the digest is a witness, not a secret).
    """
    import hashlib

    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:12]


def _digest_inputs(action: "ActionType", name: str, input: Any, output: Any) -> str:
    """Fingerprint the probe inputs without persisting them."""
    try:
        import json as _json

        payload = _json.dumps(
            [action.value if hasattr(action, "value") else str(action), name, repr(input), repr(output)],
            sort_keys=True,
            default=str,
        )
    except Exception:
        payload = f"{action}:{name}"
    return _short_sha256(payload)


def _digest_snap(snap: dict[str, Any]) -> str:
    """Fingerprint the probe return dict without persisting payloads."""
    try:
        import json as _json

        payload = _json.dumps(snap, sort_keys=True, default=str)
    except Exception:
        payload = repr(snap)
    return _short_sha256(payload)


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
    # v0.11 Fano V5 witnessability: when a coherence probe supplied the
    # phi/kuramoto_r/channel_coherence fields, this dict records a
    # fingerprint of that probe invocation so downstream consumers can
    # audit whether the coherence numbers came from a trusted source.
    # Shape (all optional):
    #   {"probe_name": str, "inputs_sha256": hex12, "output_sha256": hex12,
    #    "populated": list[str]}
    # Digests (not full payloads) preserve BITC axiom 1 (finite support).
    probe_call: dict[str, Any] | None = None
    # v0.9 EML audit hook: when a step was emitted by a code path that
    # composed the EML kernel (exp(x) - ln(y)) — the smooth coherence
    # gate, an Adam-trainable reward tree, an eval scoring tree — the
    # caller can attach the tree's structural snapshot here so downstream
    # consumers can replay, audit, or retrain it. Shape is left to the
    # producer: typically ``{"op": "eml", "children": [...], "score":
    # float, ...}``. Absent by default so legacy serializations stay
    # byte-identical (see ``to_dict`` — the key is omitted when None).
    eml_tree: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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
            "probe_call": (
                dict(self.probe_call) if self.probe_call is not None else None
            ),
        }
        # Only add eml_tree when populated so pre-v0.9 serializations
        # stay byte-identical (no empty key proliferation in the jsonl).
        if self.eml_tree is not None:
            d["eml_tree"] = _json_safe(self.eml_tree)
        return d


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
    # v0.10 W10: auto-attach coherence fields on LLM_REPLY / TOOL_CALL via a
    # caller-registered probe. When the probe is set AND the action type is
    # in ``_AUTO_ATTACH_ACTIONS`` AND the step was not passed explicit
    # coherence values, the probe is invoked and the result populates
    # phi / kuramoto_r / channel_coherence. Opt-in — default None = no probe.
    _coherence_probe: Any = field(default=None, repr=False, compare=False)

    # -- coherence auto-attach --------------------------------------------

    def register_coherence_probe(self, probe: Any) -> None:
        """Register a callable ``probe(action, name, input, output) -> dict`` that
        returns a dict with optional ``phi`` / ``kuramoto_r`` / ``channel_coherence``
        keys. Called at :meth:`record` time for action types in
        ``_AUTO_ATTACH_ACTIONS`` when explicit values aren't passed.

        Set ``probe=None`` to disable.
        """
        self._coherence_probe = probe

    def clear_coherence_probe(self) -> None:
        self._coherence_probe = None

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

        v0.10: when a coherence probe has been registered via
        :meth:`register_coherence_probe` and the action is in
        ``_AUTO_ATTACH_ACTIONS`` (LLM_REPLY, TOOL_CALL), any coherence
        fields not passed explicitly will be populated from the probe's
        return dict. Probe failures are swallowed so observability
        never kills the record path.
        """
        probe_call: dict[str, Any] | None = None
        if (
            self._coherence_probe is not None
            and action in _AUTO_ATTACH_ACTIONS
            and (phi is None and kuramoto_r is None and channel_coherence is None)
        ):
            try:
                snap = self._coherence_probe(action=action, name=name, input=input, output=output)
            except Exception:
                snap = None
            if isinstance(snap, dict):
                populated: list[str] = []
                if phi is None and "phi" in snap:
                    phi = snap.get("phi")
                    populated.append("phi")
                if kuramoto_r is None and "kuramoto_r" in snap:
                    kuramoto_r = snap.get("kuramoto_r")
                    populated.append("kuramoto_r")
                if channel_coherence is None and "channel_coherence" in snap:
                    channel_coherence = snap.get("channel_coherence")
                    populated.append("channel_coherence")
                if populated:
                    probe_call = {
                        "probe_name": getattr(
                            self._coherence_probe, "__name__", "<lambda>"
                        ),
                        "inputs_sha256": _digest_inputs(action, name, input, output),
                        "output_sha256": _digest_snap(snap),
                        "populated": populated,
                    }

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
            probe_call=probe_call,
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
            eml_tree_raw = raw.get("eml_tree")
            eml_tree = dict(eml_tree_raw) if isinstance(eml_tree_raw, dict) else None
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
                    eml_tree=eml_tree,
                )
            )
        return chain


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_REDACTED = "<redacted>"

from carl_core.errors import _is_sensitive as _is_sensitive_key


# Literal-secret shapes worth scrubbing from free-form strings. Order matters:
# more-specific patterns (Anthropic sk-ant-...) come before less-specific ones
# (generic sk-...).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # JWT: three base64url segments separated by '.'. Real-world header
    # segments are often ~18 base64url chars (e.g. ``eyJhbGciOiJIUzI1NiJ9``
    # → "{alg: HS256}" = 18 chars), so the first segment only requires
    # 10+ chars; the payload / signature requirement stays at 20+ so the
    # pattern is still distinctive. The ``\bey[A-Za-z0-9_-]...`` word
    # boundary prevents collisions with identifiers that happen to end in
    # ``...ey``.
    re.compile(
        r"\bey[A-Za-z0-9_-]{10,}\."
        r"[A-Za-z0-9_-]{20,}"
        r"(?:\.[A-Za-z0-9_-]+)?"
    ),
    # Anthropic API key — sk-ant-... — MUST be checked before the generic
    # sk-... pattern so we preserve the "sk-ant" prefix in the redacted
    # preview instead of stopping at "sk-" + 20 arbitrary chars.
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    # OpenAI-style bearer keys (sk-, sk-proj-, sk-svcacct-, ...).
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    # Hugging Face user / org tokens.
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    # EVM wallet address — 20 hex bytes after 0x, bounded so random
    # 40-hex substrings elsewhere don't collide.
    re.compile(r"\b0x[0-9a-fA-F]{40}\b"),
)


# Cheap sentinel probe — any secret shape in _SECRET_PATTERNS contains at least
# one of these substrings. Scanning for them in C via `in` is ~100x faster than
# running five regex .sub() calls on a string that doesn't contain any secret.
_SECRET_SENTINELS: tuple[str, ...] = ("sk-", "hf_", "0x", "ey")


def _scrub_secrets(text: str) -> str:
    """Replace literal secret shapes in *text* with a redacted preview.

    The first six characters of the original match (e.g. ``"eyJhbG"``,
    ``"sk-ant"``, ``"0x742d"``) are kept as a debug aid — enough to
    confirm the shape that was found without leaking the credential.
    """
    if len(text) < 20 or not any(s in text for s in _SECRET_SENTINELS):
        return text

    def _sub(match: "re.Match[str]") -> str:
        matched = match.group(0)
        prefix = matched[:6] if len(matched) >= 6 else matched
        return f"{prefix}{_REDACTED}"

    scrubbed = text
    for pattern in _SECRET_PATTERNS:
        scrubbed = pattern.sub(_sub, scrubbed)
    return scrubbed


def _json_safe(value: Any) -> Any:
    """Coerce a value to a JSON-serializable form. Falls back to repr().

    Applies two layers of secret redaction:

    1. **Name-based redaction** (dict keys): when the key name is secret-
       shaped (``*key*``, ``*token*``, ``*secret*``, ``*password*``,
       ``*authorization*``, ``*bearer*``), the value is replaced wholesale
       with ``<redacted>`` — we never recurse into the original.
    2. **Shape-based redaction** (string values): JWTs, OpenAI/Anthropic
       API keys, Hugging Face tokens, and EVM wallet addresses are
       replaced with ``<first-6-chars>+<redacted>`` so the shape stays
       debuggable while the credential does not persist.

    Both layers run on every recursive call so deeply nested tool-input
    payloads (``{"ctx": {"auth": {"token": "..."}}}``) are also covered.
    Non-string scalars, datetimes, and enums pass through unchanged.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _scrub_secrets(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _is_sensitive_key(key):
                out[key] = _REDACTED
            else:
                out[key] = _json_safe(v)
        return out
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return _scrub_secrets(repr(value))
