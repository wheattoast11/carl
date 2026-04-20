"""Smoke test: the README Quick-start snippet must actually run.

If a user pastes the first ```python``` block under ``## Quick start`` into
a REPL they should get output, not a traceback. This test extracts that
block verbatim, ``exec``\'s it in an isolated namespace, and asserts the
expected lines land on stdout.
"""
from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def _extract_quickstart_snippet() -> str:
    """Return the first python fenced code block under ``## Quick start``.

    The matcher is deliberately tight: it anchors on the heading so edits to
    unrelated sections of the README cannot silently steal the test target.
    """
    text = README.read_text(encoding="utf-8")
    # Locate the Quick-start section and everything until the next level-2
    # heading, so we never accidentally scoop a later fenced block.
    section_match = re.search(
        r"^## Quick start\s*$(?P<body>.*?)^## ",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section_match is not None, "README is missing a '## Quick start' section"
    body = section_match.group("body")

    code_match = re.search(
        r"```python\s*\n(?P<code>.*?)```",
        body,
        flags=re.DOTALL,
    )
    assert code_match is not None, (
        "README '## Quick start' section must contain a python fenced block"
    )
    return code_match.group("code")


def test_quickstart_snippet_is_runnable() -> None:
    snippet = _extract_quickstart_snippet()

    buf = io.StringIO()
    namespace: dict[str, object] = {"__name__": "__readme_quickstart__"}

    try:
        with redirect_stdout(buf):
            exec(compile(snippet, str(README), "exec"), namespace)
    except Exception as exc:  # pragma: no cover - surfaces as a test failure
        pytest.fail(f"README quickstart snippet raised: {exc!r}\n---\n{snippet}")

    output = buf.getvalue()

    # Load-bearing identifiers: the snippet must touch CoherenceProbe and the
    # published conservation constants so the example remains honest about
    # what CARL measures.
    assert "CoherenceProbe" in snippet
    assert "KAPPA" in snippet
    assert "SIGMA" in snippet

    # Output contract — the snippet prints phi_mean and the KAPPA horizon.
    # We match loosely on the labels so minor formatting tweaks don't
    # break the test, but the semantic contract stays pinned.
    assert "phi_mean" in output
    assert "horizon" in output


def test_quickstart_snippet_has_reasonable_length() -> None:
    """The snippet should stay compact — this is the first thing users see."""
    snippet = _extract_quickstart_snippet()
    nonblank_lines = [line for line in snippet.splitlines() if line.strip()]
    assert 3 <= len(nonblank_lines) <= 20, (
        f"Quick-start snippet should be 3-20 non-blank lines; got {len(nonblank_lines)}"
    )
