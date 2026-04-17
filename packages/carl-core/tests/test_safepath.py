"""Tests for ``carl_core.safepath`` — sandboxed path resolution.

The primitive replaces ad-hoc ``Path.resolve() + startswith`` checks across
the codebase. Symlink traversal is the headline security property: realpath
follows symlinks, so a ``startswith`` check against a resolved path can be
bypassed by swapping a symlink mid-chain.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from carl_core.errors import ValidationError
from carl_core.safepath import (
    PathEscape,
    SandboxedPath,
    safe_resolve,
    within,
)


def test_legit_relative_path_resolves_inside_sandbox(tmp_path: Path) -> None:
    target = tmp_path / "data" / "output.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    resolved = safe_resolve("data/output.json", tmp_path)
    assert resolved == tmp_path / "data" / "output.json"


def test_absolute_path_inside_sandbox_ok(tmp_path: Path) -> None:
    inside = tmp_path / "sub" / "file.txt"
    inside.parent.mkdir()
    inside.write_text("x")
    resolved = safe_resolve(str(inside), tmp_path)
    assert resolved == inside


def test_dotdot_escape_raises_path_escape(tmp_path: Path) -> None:
    with pytest.raises(PathEscape) as excinfo:
        safe_resolve("../etc/passwd", tmp_path)
    assert excinfo.value.code == "carl.path_escape"
    assert "escapes sandbox root" in str(excinfo.value)


def test_mixed_dotdot_that_stays_inside_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    resolved = safe_resolve("a/b/../b", tmp_path)
    assert resolved == tmp_path / "a" / "b"


def test_absolute_path_outside_sandbox_raises(tmp_path: Path) -> None:
    with pytest.raises(PathEscape) as excinfo:
        safe_resolve("/etc/passwd", tmp_path)
    assert excinfo.value.code == "carl.path_escape"


def test_prefix_adjacent_sandbox_is_rejected(tmp_path: Path) -> None:
    sibling = tmp_path.parent / (tmp_path.name + "_evil") / "file"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("bad")
    try:
        with pytest.raises(PathEscape):
            safe_resolve(str(sibling), tmp_path)
    finally:
        sibling.unlink()
        sibling.parent.rmdir()


def test_null_byte_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathEscape) as excinfo:
        safe_resolve("ok\x00/file", tmp_path)
    assert "null byte" in str(excinfo.value)
    assert excinfo.value.code == "carl.path_escape"


def test_symlink_escape_blocked(tmp_path: Path) -> None:
    # Create a sandbox and an outside target; symlink the target into sandbox.
    outside = tmp_path.parent / "outside_evil"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("classified")

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    link = sandbox / "link_to_evil"
    link.symlink_to(secret)

    try:
        with pytest.raises(PathEscape) as excinfo:
            safe_resolve("link_to_evil", sandbox)
        assert excinfo.value.code == "carl.path_escape"
        # The message identifies the symlink, not a containment mismatch —
        # because the rejection fires BEFORE containment even runs.
        assert "symlink" in str(excinfo.value)
    finally:
        link.unlink()
        secret.unlink()
        outside.rmdir()


def test_symlink_inside_sandbox_still_blocked_by_default(tmp_path: Path) -> None:
    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    real_target = sandbox / "real.txt"
    real_target.write_text("hi")
    link = sandbox / "link.txt"
    link.symlink_to(real_target)

    # Even though target is in-sandbox, a symlink at the tail is rejected
    # because it can be swapped after the check (TOCTOU).
    with pytest.raises(PathEscape) as excinfo:
        safe_resolve("link.txt", sandbox)
    assert "symlink" in str(excinfo.value)


def test_symlink_allowed_with_follow_symlinks_true(tmp_path: Path) -> None:
    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    real_target = sandbox / "real.txt"
    real_target.write_text("hi")
    link = sandbox / "link.txt"
    link.symlink_to(real_target)

    resolved = safe_resolve("link.txt", sandbox, follow_symlinks=True)
    # With follow_symlinks=True the realpath lands on the target.
    assert resolved == real_target


def test_symlink_allowed_follow_still_enforces_containment(tmp_path: Path) -> None:
    outside = tmp_path.parent / "evil_area_follow"
    outside.mkdir()
    secret = outside / "s.txt"
    secret.write_text("x")
    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    link = sandbox / "escape"
    link.symlink_to(secret)
    try:
        with pytest.raises(PathEscape):
            safe_resolve("escape", sandbox, follow_symlinks=True)
    finally:
        link.unlink()
        secret.unlink()
        outside.rmdir()


def test_within_returns_true_for_inside(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    assert within("x", tmp_path) is True


def test_within_returns_false_for_outside(tmp_path: Path) -> None:
    assert within("../etc/passwd", tmp_path) is False
    assert within("/etc/passwd", tmp_path) is False


def test_within_returns_false_for_null_byte(tmp_path: Path) -> None:
    assert within("bad\x00/name", tmp_path) is False


def test_sandboxed_path_binds_root(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    sb = SandboxedPath(tmp_path)
    assert sb.root == tmp_path.resolve()
    assert sb.follow_symlinks is False
    resolved = sb.resolve("data")
    assert resolved == tmp_path / "data"
    with pytest.raises(PathEscape):
        sb.resolve("../evil")


def test_sandboxed_path_contains_helper(tmp_path: Path) -> None:
    sb = SandboxedPath(tmp_path)
    (tmp_path / "inside").mkdir()
    assert sb.contains("inside") is True
    assert sb.contains("/etc") is False


def test_sandboxed_path_follow_symlinks_flag_carried(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("y")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    strict = SandboxedPath(tmp_path)
    lax = SandboxedPath(tmp_path, follow_symlinks=True)
    with pytest.raises(PathEscape):
        strict.resolve("link.txt")
    assert lax.resolve("link.txt") == real


def test_must_exist_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(PathEscape) as excinfo:
        safe_resolve("nonexistent.json", tmp_path, must_exist=True)
    assert "does not exist" in str(excinfo.value)


def test_must_exist_raises_on_missing_with_follow(tmp_path: Path) -> None:
    with pytest.raises(PathEscape) as excinfo:
        safe_resolve(
            "nonexistent.json", tmp_path, follow_symlinks=True, must_exist=True
        )
    assert "does not exist" in str(excinfo.value)


def test_must_exist_false_allows_missing(tmp_path: Path) -> None:
    resolved = safe_resolve("will_be_written.json", tmp_path, must_exist=False)
    assert resolved == tmp_path / "will_be_written.json"


def test_non_absolute_sandbox_root_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Force Path(".").resolve() to return a non-absolute path by stubbing.
    # Easier: SandboxedPath surfaces the same check via its constructor.
    class FakeStr:
        """Path-like that yields a relative path after resolve."""

        def __fspath__(self) -> str:
            return "relative-root"

    # For safe_resolve, simulate resolve returning relative. Use a relative
    # path; Path("relative").resolve() actually returns absolute based on cwd,
    # so monkeypatch cwd to preserve the rule test by patching resolve on the
    # specific Path instance via its class — simpler: simulate by patching
    # ``Path.resolve`` with a wrapper that keeps the relative form when the
    # input equals our sentinel.
    original_resolve = Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:  # type: ignore[override]
        if str(self) == "relative-sentinel":
            return Path("still-relative")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(ValidationError) as excinfo:
        safe_resolve("x", "relative-sentinel")
    assert "absolute" in str(excinfo.value)


def test_empty_and_dot_stay_inside_root(tmp_path: Path) -> None:
    # '.' normalizes to the sandbox root itself, which is inside.
    resolved = safe_resolve(".", tmp_path)
    assert resolved == tmp_path
    # '' joined with root is the root.
    resolved_empty = safe_resolve("", tmp_path)
    assert resolved_empty == tmp_path


def test_deeply_nested_path_works(tmp_path: Path) -> None:
    parts = [f"d{i}" for i in range(10)]
    rel = "/".join(parts)
    (tmp_path.joinpath(*parts)).mkdir(parents=True)
    resolved = safe_resolve(rel, tmp_path)
    assert resolved == tmp_path.joinpath(*parts)


def test_windows_style_backslash_on_posix(tmp_path: Path) -> None:
    # On POSIX, backslash is a regular filename char — it must NOT be
    # interpreted as a separator, and must not escape the sandbox.
    if os.sep == "\\":  # pragma: no cover — POSIX-only test
        pytest.skip("POSIX-only: backslash is a path separator on Windows")
    (tmp_path / "a\\b").touch()
    resolved = safe_resolve("a\\b", tmp_path)
    assert str(resolved).startswith(str(tmp_path))
    # A path with backslash traversal should NOT escape on POSIX.
    assert within("..\\etc\\passwd", tmp_path) is True  # filename, not escape


def test_repeated_slashes_collapse(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    resolved = safe_resolve("a//b", tmp_path)
    assert resolved == tmp_path / "a" / "b"


def test_path_escape_is_validation_subclass() -> None:
    assert issubclass(PathEscape, ValidationError)
    err = PathEscape("x")
    assert err.code == "carl.path_escape"


def test_safe_resolve_accepts_pathlike(tmp_path: Path) -> None:
    rel: Path = Path("sub") / "x"
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x").write_text("ok")
    resolved = safe_resolve(rel, tmp_path)
    assert resolved == tmp_path / "sub" / "x"


def test_symlink_midchain_blocked(tmp_path: Path) -> None:
    # Symlink at a non-tail position must also be rejected.
    outside = tmp_path.parent / "outer_midchain"
    outside.mkdir()
    (outside / "inner").mkdir()
    (outside / "inner" / "file.txt").write_text("hi")

    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    (sandbox / "mid").symlink_to(outside)

    try:
        with pytest.raises(PathEscape) as excinfo:
            safe_resolve("mid/inner/file.txt", sandbox)
        assert "symlink" in str(excinfo.value)
    finally:
        (sandbox / "mid").unlink()
        (outside / "inner" / "file.txt").unlink()
        (outside / "inner").rmdir()
        outside.rmdir()


def test_sandboxed_path_repr(tmp_path: Path) -> None:
    sb = SandboxedPath(tmp_path, follow_symlinks=True)
    r = repr(sb)
    assert "SandboxedPath" in r
    assert "follow_symlinks=True" in r


def test_sandboxed_path_non_absolute_root_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resolve = Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:  # type: ignore[override]
        if str(self) == "relative-sb":
            return Path("still-relative")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(ValidationError):
        SandboxedPath("relative-sb")
