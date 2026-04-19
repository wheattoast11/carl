"""JIT runtime context extraction.

On the user's FIRST input (which is a response to the env-baked intro
greeting), we parse intent, domain, and task type, then produce a
``JITContext`` that primes the WorkFrame and system prompt for the
heartbeat loop and CARLAgent. No network, no LLM call — pure local
pattern matching.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskIntent(str, Enum):
    EXPLORE = "explore"
    TRAIN = "train"
    EVAL = "eval"
    STICKY = "sticky"
    FREE = "free"


_INTENT_PATTERNS: dict[TaskIntent, list[re.Pattern[str]]] = {
    TaskIntent.EXPLORE: [
        re.compile(r"\b(explore|scan|map|survey|discover|what['’]?s in|show me)\b", re.I),
        re.compile(r"\b(repo|codebase|project structure)\b", re.I),
    ],
    TaskIntent.TRAIN: [
        re.compile(r"\b(train|fine[- ]?tune|sft|grpo|lora|qlora|finetune)\b", re.I),
        re.compile(r"\b(teach|learn|optimi[sz]e)\b", re.I),
    ],
    TaskIntent.EVAL: [
        re.compile(r"\b(eval|evaluate|test|benchmark|score|measure)\b", re.I),
        re.compile(r"\b(how well|performance|accuracy)\b", re.I),
    ],
    TaskIntent.STICKY: [
        re.compile(r"^/?sticky\b", re.I),
        re.compile(r"^(queue|remind|note)\b", re.I),
    ],
}


_MOVE_KEY_TO_INTENT: dict[str, TaskIntent] = {
    "e": TaskIntent.EXPLORE,
    "t": TaskIntent.TRAIN,
    "v": TaskIntent.EVAL,
    "s": TaskIntent.STICKY,
}


class JITContext(BaseModel):
    """Parsed runtime context derived from the user's first message.

    The object is intentionally small and JSON-serializable so the
    heartbeat loop can persist it alongside an ``InteractionChain`` step
    of type :class:`~carl_core.interaction.ActionType.USER_INPUT`.
    """

    intent: TaskIntent
    domain: str = ""
    task_type: str = ""
    raw_input: str
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    frame_patch: dict[str, Any] = Field(default_factory=dict)
    primer: str = ""

    def system_prompt_extension(self) -> str:
        """Text to append to the system prompt on turn 1."""
        if self.primer:
            return self.primer
        return (
            f"[jit-context] intent={self.intent.value} "
            f"domain={self.domain or 'unspecified'} "
            f"task_type={self.task_type or 'free-form'}"
        )


def extract(raw: str, move_key: str | None = None) -> JITContext:
    """Derive a :class:`JITContext` from raw user input + optional move key.

    - If ``move_key`` is one of e/t/v/s, intent short-circuits.
    - Otherwise, regex sweep over ``_INTENT_PATTERNS`` picks the first match.
    - ``domain`` is a crude noun-phrase extraction (first tokens of the
      input). No external NLP dep.
    """
    # Defensive: callers may originate from dynamic CLI input where the type
    # guarantee from the annotation doesn't actually hold. Checking __class__
    # avoids pyright's ``reportUnnecessaryIsInstance`` while preserving the
    # runtime guard.
    if raw.__class__ is not str:
        raise TypeError(f"raw must be str, got {type(raw).__name__}")

    normalized_raw = raw.strip()

    intent = _MOVE_KEY_TO_INTENT.get((move_key or "").lower().strip(), TaskIntent.FREE)
    if intent is TaskIntent.FREE:
        for candidate, patterns in _INTENT_PATTERNS.items():
            if any(p.search(normalized_raw) for p in patterns):
                intent = candidate
                break

    domain = _extract_domain(normalized_raw, intent)
    task_type = _extract_task_type(normalized_raw, intent)
    primer = _build_primer(normalized_raw, intent, domain, task_type)
    frame_patch = _build_frame_patch(intent, domain, task_type)

    return JITContext(
        intent=intent,
        domain=domain,
        task_type=task_type,
        raw_input=normalized_raw,
        frame_patch=frame_patch,
        primer=primer,
    )


def _extract_domain(raw: str, intent: TaskIntent) -> str:
    """First 5 identifier-ish tokens, truncated at 80 chars."""
    tokens = re.findall(r"[A-Za-z_][\w\-./]*", raw)
    if not tokens:
        return ""
    candidate = " ".join(tokens[:5]).strip()
    return candidate[:80]


def _extract_task_type(raw: str, intent: TaskIntent) -> str:
    """Map keywords in ``raw`` to a coarse task type. Fall back to intent."""
    hits: dict[str, str] = {
        "curriculum": r"\bcurric",
        "reward": r"\breward\b",
        "checkpoint": r"\b(checkpoint|ckpt)\b",
        "dataset": r"\b(dataset|corpus)\b",
        "model": r"\b(model|weights)\b",
    }
    for label, pattern in hits.items():
        if re.search(pattern, raw, re.I):
            return label
    return intent.value


def _build_primer(raw: str, intent: TaskIntent, domain: str, task_type: str) -> str:
    return (
        f"[intro-response] You greeted the user; they answered: {raw!r}. "
        f"Intent classified as {intent.value}. "
        f"Domain hint: {domain or '(unspecified)'}. "
        f"Task type hint: {task_type or 'free-form'}. "
        f"Prime the conversation to this intent immediately; do not re-ask basic setup."
    )


def _build_frame_patch(intent: TaskIntent, domain: str, task_type: str) -> dict[str, Any]:
    """Build a dict suitable for :meth:`WorkFrame.model_copy(update=...)`.

    WorkFrame fields (see ``carl_studio.frame``): ``domain``, ``function``,
    ``role``, ``objectives``, ``constraints``, ``entities``, ``metrics``,
    ``context``. We only populate ``domain``, ``function``, and
    ``objectives`` so the patch stays narrow and predictable.
    """
    patch: dict[str, Any] = {}
    if domain:
        patch["domain"] = domain
    intent_to_function: dict[TaskIntent, str] = {
        TaskIntent.EXPLORE: "exploration",
        TaskIntent.TRAIN: "training",
        TaskIntent.EVAL: "evaluation",
        TaskIntent.STICKY: "task-management",
        TaskIntent.FREE: "general",
    }
    patch["function"] = intent_to_function.get(intent, "general")
    if task_type:
        patch["objectives"] = [f"{intent.value}:{task_type}"]
    return patch


__all__ = [
    "JITContext",
    "TaskIntent",
    "extract",
]
