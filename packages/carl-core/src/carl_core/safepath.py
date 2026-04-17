"""Path sandboxing: reject any path that escapes a sandbox root.

Single canonical primitive replacing ad-hoc ``Path(...).resolve()`` +
``startswith(workdir)`` checks scattered across the code base. The ad-hoc
pattern is bypassable via symlinks (``realpath`` resolves through symlinks,
which an attacker can swap after the check — TOCTOU).

``safe_resolve`` rejects:
  * null bytes,
  * paths whose normalized form escapes the sandbox root,
  * symlinks anywhere along the path chain (unless ``follow_symlinks=True``).
"""
from __future__ import annotations

import os
from pathlib import Path

from carl_core.errors import ValidationError


class PathEscape(ValidationError):
    """Raised when a user-supplied path escapes its sandbox root."""

    code = "carl.path_escape"


def safe_resolve(
    user_path: str | os.PathLike[str],
    sandbox_root: str | os.PathLike[str],
    *,
    follow_symlinks: bool = False,
    must_exist: bool = False,
) -> Path:
    """Resolve ``user_path`` and ensure it is within ``sandbox_root``.

    Rejects traversal (``../``), symlinks (by default), null bytes, and
    absolute paths outside the sandbox. Returns the resolved absolute
    :class:`Path`.

    When ``follow_symlinks=False`` (the default), any symlink on the input
    chain — including the tail component — is rejected even if its target is
    inside the sandbox. The symlink itself could be swapped to escape the
    sandbox after the check (TOCTOU), so rejection is the only safe policy.

    Parameters
    ----------
    user_path:
        Path supplied by the caller / user. May be relative (joined under
        ``sandbox_root``) or absolute (must be inside ``sandbox_root``).
    sandbox_root:
        Absolute-path root that ``user_path`` must resolve into. The root
        itself is resolved (strict=False) before the check so the caller may
        pass a relative sandbox path from the current working directory.
    follow_symlinks:
        If True, symlinks are traversed via ``Path.resolve(strict=...)``. The
        final containment check still applies. Use only when the operator
        owns every symlink in the chain.
    must_exist:
        If True, raises :class:`PathEscape` when the resolved path does not
        exist on disk.

    Raises
    ------
    PathEscape
        When the path escapes the sandbox, contains a null byte, or (in
        non-follow mode) traverses a symlink.
    ValidationError
        When ``sandbox_root`` does not resolve to an absolute path.
    """
    user_str = os.fspath(user_path)
    if "\x00" in user_str:
        raise PathEscape("path contains null byte", context={"path": user_str})

    root = Path(sandbox_root).resolve(strict=False)
    if not root.is_absolute():
        raise ValidationError(
            "sandbox_root must resolve to an absolute path",
            context={"root": str(sandbox_root)},
        )

    user_as_path = Path(user_str)
    joined = user_as_path if user_as_path.is_absolute() else root / user_str

    if follow_symlinks:
        # strict=must_exist raises FileNotFoundError when the path is missing.
        # Convert that to PathEscape for a uniform error surface.
        try:
            resolved = joined.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise PathEscape(
                "path does not exist",
                context={"path": str(joined)},
            ) from exc
    else:
        # Normalize without following symlinks. This collapses ".." segments
        # using lexical rules so the containment check against ``root`` is
        # meaningful. We must NOT call ``Path.resolve`` here because that
        # follows symlinks.
        normalized = os.path.normpath(str(joined))
        resolved = Path(normalized)
        if must_exist and not resolved.exists():
            raise PathEscape(
                "path does not exist",
                context={"path": str(resolved)},
            )
        # Walk every component of the normalized path. Any symlink encountered
        # — even if its target is inside the sandbox — is rejected, because
        # the symlink can be swapped after the check (TOCTOU).
        _reject_any_symlink(resolved)

    # Containment check. Compare as strings so we catch prefix-adjacency cases
    # (``/tmp/work2`` vs ``/tmp/work``) by requiring the path separator.
    root_str = str(root)
    resolved_str = str(resolved)
    sep = os.sep
    if resolved_str != root_str and not resolved_str.startswith(root_str + sep):
        raise PathEscape(
            "path escapes sandbox root",
            context={"path": resolved_str, "root": root_str},
        )
    return resolved


def _reject_any_symlink(path: Path) -> None:
    """Raise :class:`PathEscape` if any component of ``path`` is a symlink.

    Traverses the chain from the anchor outward. ``Path.is_symlink`` is safe
    for non-existent paths (returns False), so missing paths pass silently —
    callers that care about existence gate on ``must_exist``.
    """
    if not path.is_absolute():
        # Should not happen — ``safe_resolve`` always hands us an absolute
        # path after normalization. Keep a defensive guard.
        return
    anchor = path.anchor or os.sep
    cursor = Path(anchor)
    relative_parts = path.relative_to(anchor).parts
    for part in relative_parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PathEscape(
                "symlink encountered in path (follow_symlinks=False)",
                context={"path": str(path), "symlink_at": str(cursor)},
            )


def within(
    path: str | os.PathLike[str],
    root: str | os.PathLike[str],
) -> bool:
    """Return True iff ``path`` resolves inside ``root`` without symlinks.

    Mirrors :func:`safe_resolve` with ``follow_symlinks=False``,
    ``must_exist=False`` and swallows the resulting errors into a boolean.
    """
    try:
        safe_resolve(path, root, follow_symlinks=False, must_exist=False)
    except (PathEscape, ValidationError):
        return False
    return True


class SandboxedPath:
    """Bind a sandbox root for repeated safe resolves.

    Example
    -------
    >>> sandbox = SandboxedPath("/tmp/work")
    >>> sandbox.resolve("data/output.json")          # inside root -> ok
    >>> sandbox.resolve("../etc/passwd")             # raises PathEscape
    """

    __slots__ = ("_root", "_follow_symlinks")

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        follow_symlinks: bool = False,
    ) -> None:
        self._root: Path = Path(root).resolve(strict=False)
        if not self._root.is_absolute():
            raise ValidationError(
                "sandbox root must resolve to an absolute path",
                context={"root": str(root)},
            )
        self._follow_symlinks: bool = follow_symlinks

    @property
    def root(self) -> Path:
        return self._root

    @property
    def follow_symlinks(self) -> bool:
        return self._follow_symlinks

    def resolve(
        self,
        user_path: str | os.PathLike[str],
        *,
        must_exist: bool = False,
    ) -> Path:
        return safe_resolve(
            user_path,
            self._root,
            follow_symlinks=self._follow_symlinks,
            must_exist=must_exist,
        )

    def contains(self, path: str | os.PathLike[str]) -> bool:
        """Boolean membership check against this sandbox."""
        return within(path, self._root)

    def __repr__(self) -> str:
        return (
            f"SandboxedPath(root={self._root!s}, "
            f"follow_symlinks={self._follow_symlinks})"
        )


__all__ = [
    "PathEscape",
    "SandboxedPath",
    "safe_resolve",
    "within",
]
