"""Env-baked pre-coded intro for bare ``carl``.

Zero-latency greeting rendered before the first ``input()``. Proposes 4
keyed moves (e/t/v/s) plus free-form. Shipped as a constant so there
is no I/O, no network, no LLM call before the user acts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from carl_studio.console import CampConsole

INTRO_MOVES: Final[list[tuple[str, str, str]]] = [
    ("e", "explore",  "map your repo and surface a starting goal"),
    ("t", "train",    "start or continue a training run"),
    ("v", "evaluate", "run an eval against your latest checkpoint"),
    ("s", "sticky",   "queue a note for the heartbeat loop"),
]

INTRO_TITLE: Final[str] = "carl · coherence-aware reinforcement learning"


def render_intro(c: CampConsole) -> None:
    """Print the intro greeting + moves panel.

    Uses the existing ``CampConsole`` primitives (``rule``/``info``/``kv``)
    so the output flows through the configured theme. No color/icon logic
    lives here — it's all in the theme layer.
    """
    c.rule()
    c.info(INTRO_TITLE)
    c.info("what shall we do?")
    for key, label, desc in INTRO_MOVES:
        c.kv(f"[{key}] {label}", desc)
    c.info("  or type anything — i'll pick up the thread.")
    c.rule()


def parse_intro_selection(raw: str) -> str | None:
    """Return the canonical move key (e/t/v/s) for a selection.

    Returns ``None`` when the input is free-form (no keyed intent match).
    Accepts either the single-letter shortcut (``e``) or the full word
    (``explore``). Whitespace and case are ignored.
    """
    if not raw:
        return None
    token = raw.strip().lower()
    if not token:
        return None
    for key, label, _ in INTRO_MOVES:
        if token == key or token == label:
            return key
    return None


__all__ = [
    "INTRO_MOVES",
    "INTRO_TITLE",
    "parse_intro_selection",
    "render_intro",
]
