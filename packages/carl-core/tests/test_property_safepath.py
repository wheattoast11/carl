"""Property tests for carl_core.safepath.

Invariants:
- Any safe relative path accepted under a fresh tmp sandbox.
- Any path containing ``..`` that escapes is rejected.
- Null byte is always rejected.
- Symlinks at any position in the chain are rejected with follow_symlinks=False.

``tmp_path`` is function-scoped; Hypothesis will not reset function-scoped
fixtures between generated examples. We therefore create a fresh sandbox
directory inside every test body with ``tempfile.TemporaryDirectory``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.safepath import PathEscape, safe_resolve, within

from hypothesis_strategies import st_safe_name, st_safe_relpath


@given(rel=st_safe_relpath)
@settings(max_examples=100, deadline=None)
def test_safe_relpath_accepted(rel: str) -> None:
    """Any syntactically safe multi-segment relative path is accepted."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        resolved = safe_resolve(rel, root)
        assert str(resolved).startswith(str(root.resolve()))
        assert within(rel, root) is True


@given(
    prefix_dots=st.integers(min_value=1, max_value=5),
    trail=st_safe_name,
)
@settings(max_examples=100, deadline=None)
def test_parent_traversal_rejected(prefix_dots: int, trail: str) -> None:
    """Any path with leading ``../`` segments escapes and must be rejected."""
    rel = "/".join([".."] * prefix_dots + [trail])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with pytest.raises(PathEscape):
            safe_resolve(rel, root)
        assert within(rel, root) is False


@given(
    pre=st.text(max_size=8),
    post=st.text(max_size=8),
)
@settings(max_examples=50, deadline=None)
def test_null_byte_rejected(pre: str, post: str) -> None:
    """Any path containing a null byte is rejected with PathEscape."""
    if "\x00" in pre or "\x00" in post:
        return
    rel = f"{pre}\x00{post}"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with pytest.raises(PathEscape):
            safe_resolve(rel, root)
        assert within(rel, root) is False


@given(link_name=st_safe_name)
@settings(max_examples=30, deadline=None)
def test_symlink_blocked_by_default(link_name: str) -> None:
    """A symlink in the sandbox is rejected when follow_symlinks=False."""
    if link_name == "real.txt":
        return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        real = root / "real.txt"
        real.write_text("hi")
        link = root / link_name
        link.symlink_to(real)
        with pytest.raises(PathEscape):
            safe_resolve(link_name, root)
