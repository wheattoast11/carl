"""Regression tests for scripts/check_moat_boundary.py.

Ensures the moat enforcement:
- Passes on the current clean tree
- Correctly flags a synthetic top-level `import resonance`
- Ignores lazy imports inside functions
- Ignores string references (e.g. the literal `"resonance"` in admin.py)
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_moat_boundary.py"


def _run_check(cwd: Path) -> tuple[int, str, str]:
    """Run the moat check with ``cwd`` as the working directory."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return r.returncode, r.stdout, r.stderr


def test_main_tree_passes_moat_check() -> None:
    """The live repo must pass the check. Regression guard against drift."""
    rc, out, err = _run_check(REPO_ROOT)
    assert rc == 0, f"moat check failed on main tree:\n{out}\n{err}"
    assert "clean" in out.lower()


def test_top_level_import_fails(tmp_path: Path) -> None:
    """A synthetic tree with a forbidden top-level import must trip rc=1."""
    # Build a minimal repo skeleton: the script scans
    # packages/carl-core/src and src/carl_studio — we only need one of them.
    pkg_dir = tmp_path / "src" / "carl_studio"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "bad.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            # MOAT VIOLATION: private import at module level
            import resonance

            def use() -> None:
                _ = resonance
            """
        ).strip()
    )

    # Also create the scripts directory path expectation
    # (the script uses Path('packages/carl-core/src') + Path('src/carl_studio'))
    # Copy the script into the tmp repo so relative paths resolve correctly.
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "check_moat_boundary.py").write_text(SCRIPT.read_text())

    r = subprocess.run(
        [sys.executable, "scripts/check_moat_boundary.py"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 1
    assert "resonance" in r.stderr
    assert "import resonance" in r.stderr


def test_from_import_fails(tmp_path: Path) -> None:
    """`from resonance.X import Y` at module level must also trip."""
    pkg_dir = tmp_path / "packages" / "carl-core" / "src" / "carl_core"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "bad.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            from resonance.signals import constitutional  # MOAT VIOLATION
            """
        ).strip()
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "check_moat_boundary.py").write_text(SCRIPT.read_text())

    r = subprocess.run(
        [sys.executable, "scripts/check_moat_boundary.py"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 1
    assert "from resonance.signals import constitutional" in r.stderr


def test_lazy_import_inside_function_is_allowed(tmp_path: Path) -> None:
    """Imports inside function bodies are the canonical admin-gate pattern."""
    pkg_dir = tmp_path / "src" / "carl_studio"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "ok.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            def fetch() -> object:
                # Lazy import — the canonical pattern
                import resonance.signals.heartbeat as hb
                return hb
            """
        ).strip()
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "check_moat_boundary.py").write_text(SCRIPT.read_text())

    r = subprocess.run(
        [sys.executable, "scripts/check_moat_boundary.py"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 0, f"expected clean:\n{r.stdout}\n{r.stderr}"


def test_terminals_runtime_also_forbidden(tmp_path: Path) -> None:
    """The check covers `terminals_runtime` as well as `resonance`."""
    pkg_dir = tmp_path / "src" / "carl_studio"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "bad.py").write_text(
        "from __future__ import annotations\n"
        "from terminals_runtime.x import y\n"
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "check_moat_boundary.py").write_text(SCRIPT.read_text())

    r = subprocess.run(
        [sys.executable, "scripts/check_moat_boundary.py"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 1
    assert "terminals_runtime" in r.stderr


def test_string_reference_is_allowed(tmp_path: Path) -> None:
    """String references like `'resonance'` aren't imports — must pass."""
    pkg_dir = tmp_path / "src" / "carl_studio"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "ok.py").write_text(
        textwrap.dedent(
            '''
            """Module docstring mentioning resonance and terminals_runtime."""
            from __future__ import annotations

            _LABEL = "resonance"
            _OTHER = "terminals_runtime"

            def fetch(name: str = "resonance.signals.foo") -> str:
                return name
            '''
        ).strip()
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "check_moat_boundary.py").write_text(SCRIPT.read_text())

    r = subprocess.run(
        [sys.executable, "scripts/check_moat_boundary.py"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 0


@pytest.mark.parametrize("root_name", ["resonance", "terminals_runtime"])
def test_dotted_path_variants_fail(tmp_path: Path, root_name: str) -> None:
    """Top-level `import resonance.signals.constitutional` also fails."""
    pkg_dir = tmp_path / "src" / "carl_studio"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "bad.py").write_text(
        f"from __future__ import annotations\nimport {root_name}.signals.x\n"
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "check_moat_boundary.py").write_text(SCRIPT.read_text())

    r = subprocess.run(
        [sys.executable, "scripts/check_moat_boundary.py"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 1
    assert root_name in r.stderr
