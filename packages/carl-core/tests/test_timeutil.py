"""Tests for carl_core.timeutil — the single source of truth for the
ISO-8601 "Z" format used across ~/.carl state rows and JSONL traces.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from carl_core import ISO8601_Z, now_iso, now_iso_ms
from carl_core.timeutil import now_iso as now_iso_submodule


# Canonical shape for the second-resolution variant used by every migrated
# production call site (sticky_notes.created_at, camp_profile_cached_at,
# consent.changed_at, carlito.updated_at, ...).
_ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_ISO_Z_MS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def test_now_iso_format_matches_iso_z() -> None:
    """now_iso() output matches ``YYYY-MM-DDTHH:MM:SSZ`` exactly.

    No microseconds, no offset, ``Z`` suffix — byte-identical to every
    ``.strftime("%Y-%m-%dT%H:%M:%SZ")`` call site we replaced.
    """
    ts = now_iso()
    assert _ISO_Z_RE.fullmatch(ts) is not None, f"shape mismatch: {ts!r}"
    assert ts.endswith("Z")
    assert "+" not in ts
    assert "." not in ts
    assert ISO8601_Z == "%Y-%m-%dT%H:%M:%SZ"


def test_now_iso_is_utc() -> None:
    """now_iso() is grounded in UTC, not the host's local time.

    Regression: parse the timestamp back as UTC, compare to a fresh UTC
    ``datetime.now`` — they must be within a small wall-clock window
    regardless of the ``TZ`` environment variable on the test host.
    """
    ts = now_iso()
    parsed = datetime.strptime(ts, ISO8601_Z).replace(tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    delta = abs((now_utc - parsed).total_seconds())
    # Allow 5s slack for slow CI; enough to catch a TZ-offset bug (which
    # would produce deltas in the thousands of seconds).
    assert delta < 5.0, f"now_iso={ts} drifted {delta}s from UTC"


def test_now_iso_second_resolution() -> None:
    """Confirm sub-second precision is truncated — SQLite rows + JSONL
    traces rely on this for text-comparable ordering.
    """
    ts = now_iso()
    # Length is exactly 20 chars (19 char body + the ``Z``).
    assert len(ts) == 20, f"expected 20 chars, got {len(ts)}: {ts!r}"
    # Two calls within the same second MUST be byte-identical (that's the
    # whole point of second resolution; the dequeue/complete/cutoff math
    # in sticky.py depends on it).
    first = now_iso()
    second = now_iso()
    # They may differ if the test rolls a second boundary between calls,
    # but in that case the later one must be strictly greater.
    assert first <= second


def test_now_iso_ms_format_matches() -> None:
    """now_iso_ms() adds three-digit millisecond precision and preserves
    the ``Z`` suffix.
    """
    ts = now_iso_ms()
    assert _ISO_Z_MS_RE.fullmatch(ts) is not None, f"shape mismatch: {ts!r}"
    # Exactly 24 chars: "YYYY-MM-DDTHH:MM:SS.mmmZ".
    assert len(ts) == 24
    assert ts.endswith("Z")


def test_now_iso_distinct_calls_monotonic_or_equal() -> None:
    """Across a handful of distinct calls, output is non-decreasing.

    Regression: prevents accidentally switching the clock source to
    something that jumps backward (e.g. ``time.time()`` after a
    daylight-saving flip would still be fine because we use UTC, but a
    future refactor that swaps in a local-time source would not).
    """
    samples = [now_iso()]
    for _ in range(5):
        time.sleep(0.001)
        samples.append(now_iso())
    assert samples == sorted(samples), f"non-monotonic: {samples}"


def test_now_iso_reexport_matches_submodule() -> None:
    """The carl_core top-level re-export is the same callable as the one
    in carl_core.timeutil — no accidental shadow, no stale rebind.
    """
    assert now_iso is now_iso_submodule


def test_iso8601_z_constant_roundtrip() -> None:
    """``ISO8601_Z`` is the exact strftime format used by now_iso().

    Consumers like ``sticky.reclaim_stale`` format cutoff datetimes with
    this constant; the whole migration rests on the string being stable.
    """
    fixed = datetime(2026, 4, 19, 12, 34, 56, tzinfo=timezone.utc)
    assert fixed.strftime(ISO8601_Z) == "2026-04-19T12:34:56Z"
