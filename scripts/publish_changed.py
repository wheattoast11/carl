"""Detect which sub-packages have changed since a reference tag.

Usage:

    python scripts/publish_changed.py --since carl-studio@0.3.0
    python scripts/publish_changed.py --since carl-core@0.1.0 --format tags

Emits either:

  - One package name per line (default)
  - Tag lines ``<package>@<current-version>`` when ``--format tags``

The tag form is suitable for piping into ``git tag``/``gh release`` to
create release tags only for what actually changed.

This is a *helper*, not a CI gate. The workflow itself is driven by
scoped tags (see ``.github/workflows/publish.yml``); this script tells
you which tags you probably want to create.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"


def _git_diff_names(since: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{since}..HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _changed_subpackages(since: str) -> list[str]:
    names: set[str] = set()
    for path in _git_diff_names(since):
        if path.startswith("packages/"):
            parts = path.split("/", 2)
            if len(parts) >= 2 and parts[1]:
                names.add(parts[1])
    return sorted(names)


def _root_changed(since: str) -> bool:
    """Did any file outside packages/ change? (carl-studio itself)."""
    for path in _git_diff_names(since):
        if not path.startswith("packages/"):
            return True
    return False


def _read_version(pyproject: Path) -> str:
    try:
        import tomllib  # py311+
    except ImportError:  # pragma: no cover - runtime guard for older pythons
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    data: dict[str, dict[str, str]] = tomllib.loads(pyproject.read_text())
    return str(data["project"]["version"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, help="Reference tag or commit to diff against")
    parser.add_argument("--format", choices=("names", "tags"), default="names")
    parser.add_argument("--include-root", action="store_true", help="Also emit carl-studio if root changed")
    args = parser.parse_args(argv)

    try:
        changed = _changed_subpackages(args.since)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"git diff failed: {exc.stderr.strip() or exc}\n")
        return 2

    outputs: list[str] = []
    for name in changed:
        pyproject = PACKAGES_DIR / name / "pyproject.toml"
        if not pyproject.exists():
            continue
        if args.format == "tags":
            outputs.append(f"{name}@{_read_version(pyproject)}")
        else:
            outputs.append(name)

    if args.include_root and _root_changed(args.since):
        pyproject = REPO_ROOT / "pyproject.toml"
        if args.format == "tags":
            outputs.append(f"carl-studio@{_read_version(pyproject)}")
        else:
            outputs.append("carl-studio")

    if not outputs:
        return 0
    print("\n".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
