"""Screen / text redaction helpers (Stage C-2 scaffold).

Inspired by the openadapt-privacy redaction rules: scrub PII, credentials,
and financial identifiers from strings before they land in a
:class:`~carl_studio.handles.data.DataToolkit` bytes handle.

The :class:`~carl_core.interaction.InteractionChain` already scrubs
credential shapes at serialization time (JWTs, ``sk-ant-*``, ``hf_*``,
EVM wallets). This module layers on **content-level** scrubbing that
makes sense BEFORE content reaches the audit trail — phone numbers,
SSN-shaped strings, credit card numbers, email addresses the chain
wouldn't otherwise scrub.

Contract:

* :func:`redact_text` — sync, string in / string out. Redacted spans
  are replaced with ``<REDACTED:<category>>`` so the shape stays
  debuggable without the value.
* :func:`redact_preview_spans` — returns structured spans (for UI layers
  that want to highlight or count redactions per category).

Scope for Stage C-2: US-centric regexes, deliberately conservative
(low false-positive rate; false-negative acceptable because the chain
still runs its own pass). A future pass can wire in openadapt's
ML-assisted redactor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "RedactionCategory",
    "RedactionSpan",
    "redact_text",
    "redact_preview_spans",
]


RedactionCategory = Literal[
    "email",
    "phone",
    "ssn",
    "credit_card",
    "ipv4",
    "date_of_birth",
]


@dataclass(frozen=True)
class RedactionSpan:
    """One redacted span inside the input string."""

    start: int
    end: int
    category: RedactionCategory
    preview: str  # first 2 chars of the matched value (still audit-safe)


# Patterns are deliberately conservative. Order matters: more specific
# patterns (SSN) come before broader ones (phone) to prevent overlap.
_PATTERNS: tuple[tuple[RedactionCategory, re.Pattern[str]], ...] = (
    # Email — simple RFC 5322 subset, good enough for content logs.
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    # Social Security Number — 3-2-4 digits with dashes.
    (
        "ssn",
        re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    ),
    # Credit card — Visa / MC / Amex / Discover lengths. No Luhn check
    # (false-positive risk is acceptable; if you want Luhn, plug it in).
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    ),
    # US phone — (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, +1...
    (
        "phone",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
    # IPv4 — strict dotted quad.
    (
        "ipv4",
        re.compile(
            r"\b(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}\b"
        ),
    ),
    # Date of birth — MM/DD/YYYY or YYYY-MM-DD in a narrow year range.
    (
        "date_of_birth",
        re.compile(
            r"\b(?:(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-]"
            r"(?:19|20)\d{2}|(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-"
            r"(?:0[1-9]|[12]\d|3[01]))\b"
        ),
    ),
)


def redact_preview_spans(text: str) -> list[RedactionSpan]:
    """Return non-overlapping redaction spans in order."""
    found: list[RedactionSpan] = []
    occupied: list[tuple[int, int]] = []
    for category, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(not (end <= a or start >= b) for a, b in occupied):
                continue
            preview = match.group(0)[:2]
            found.append(
                RedactionSpan(start=start, end=end, category=category, preview=preview)
            )
            occupied.append((start, end))
    found.sort(key=lambda s: s.start)
    return found


def redact_text(text: str) -> str:
    """Return ``text`` with every redaction span replaced by a marker.

    Example: ``"call 555-123-4567"`` → ``"call <REDACTED:phone>"``.
    """
    spans = redact_preview_spans(text)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for span in spans:
        out.append(text[cursor:span.start])
        out.append(f"<REDACTED:{span.category}>")
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)
