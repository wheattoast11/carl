#!/usr/bin/env python3
"""Moat boundary check — enforce the public/private import seam.

Scans `packages/carl-core/src/` and `src/carl_studio/` for module-level
imports of the private runtime packages (`resonance`, `terminals_runtime`).
Such imports are forbidden at module load because they would pull private
code into the public (MIT) surface at import time.

The only allowed form is lazy imports inside functions, typically via::

    def some_fn(...):
        from carl_studio.admin import load_private
        if is_admin():
            mod = load_private("signals.heartbeat")
            ...

This pattern keeps the admin-gate seam intact: carl-studio imports
``carl_studio.admin`` (public), and the admin module resolves the private
``resonance.*`` modules lazily via ``load_private``.

Exit codes:
    0 — no violations (clean moat)
    1 — one or more top-level private imports found
    2 — internal error (e.g. failed to parse a file)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple


# Packages whose module-level import into carl-core / carl-studio is forbidden.
FORBIDDEN_ROOTS: frozenset[str] = frozenset({"resonance", "terminals_runtime"})

# Directories to scan (relative to repo root).
SCAN_ROOTS: tuple[Path, ...] = (
    Path("packages/carl-core/src"),
    Path("src/carl_studio"),
)

# Files explicitly exempt from the check. Keep this list minimal; every
# exemption should be documented with a reason.
EXEMPTIONS: frozenset[Path] = frozenset({
    # admin.py's whole purpose is to bridge public -> private via
    # load_private(). Its string references are fine; we check that no
    # top-level `import resonance` lands.
})


class Violation(NamedTuple):
    path: Path
    lineno: int
    module: str
    detail: str


def _module_root(name: str) -> str:
    """Return the top-level package of an import name (``a.b.c`` → ``a``)."""
    return name.split(".", 1)[0] if name else ""


def check_file(path: Path) -> list[Violation]:
    """Return every forbidden module-level import discovered in ``path``."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover — unlikely outside CI
        return [Violation(path, 0, "<read-error>", str(exc))]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Violation(path, exc.lineno or 0, "<parse-error>", str(exc))]

    violations: list[Violation] = []

    for node in tree.body:
        # `import X` (possibly dotted)
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _module_root(alias.name)
                if root in FORBIDDEN_ROOTS:
                    violations.append(
                        Violation(
                            path=path,
                            lineno=node.lineno,
                            module=alias.name,
                            detail=f"`import {alias.name}` at module level",
                        )
                    )

        # `from X import Y`
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            root = _module_root(node.module)
            if root in FORBIDDEN_ROOTS:
                names = ", ".join(a.name for a in node.names)
                violations.append(
                    Violation(
                        path=path,
                        lineno=node.lineno,
                        module=node.module,
                        detail=f"`from {node.module} import {names}` at module level",
                    )
                )

    return violations


def iter_py_files(roots: tuple[Path, ...]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return sorted(out)


def main() -> int:
    files = iter_py_files(SCAN_ROOTS)
    if not files:
        print("moat-check: no python files found under", SCAN_ROOTS, file=sys.stderr)
        return 2

    violations: list[Violation] = []
    for path in files:
        if path in EXEMPTIONS:
            continue
        violations.extend(check_file(path))

    if not violations:
        print(
            f"moat-check: ✓ clean ({len(files)} files scanned, "
            f"0 forbidden top-level imports of {sorted(FORBIDDEN_ROOTS)})"
        )
        return 0

    print("moat-check: ✗ forbidden top-level imports found:\n", file=sys.stderr)
    for v in violations:
        print(f"  {v.path}:{v.lineno}  {v.detail}", file=sys.stderr)
    print(
        "\nPrivate-runtime modules must be loaded lazily via "
        "`carl_studio.admin.load_private(...)` inside function bodies, "
        "NOT imported at module load. See docs/v17_admin_gate_pattern.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
