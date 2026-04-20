"""Tests for the ``CARL_HOME`` canonical resolver.

Ticket: v0.7.1 WAVE-1 foundational migration. Verifies that the helper
:func:`carl_studio.settings.carl_home` is honored across the modules that
previously hardcoded ``Path.home() / ".carl"``.

Implementation note: to avoid leaking reload state into other test modules
(which would break ``isinstance`` checks against classes exported from
``wallet_store`` / ``db`` / ``llm``), module-level constants are verified by
loading a *fresh copy* of the target module under a shadow name via
``importlib.util.spec_from_file_location``. That leaves ``sys.modules`` entries
for the real modules untouched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper: shadow-load a module without polluting sys.modules.
# ---------------------------------------------------------------------------


def _shadow_load(module_name: str, shadow_alias: str):
    """Load ``module_name`` under ``shadow_alias`` with a fresh execution.

    We resolve the file path via the real import system, then build a new
    ``ModuleSpec`` targeting that path but registered under a different key.
    The freshly executed module will re-read ``os.environ["CARL_HOME"]`` via
    its own imports (including the ``from carl_studio.settings import carl_home``
    call, which invokes the live helper function).
    """
    real_spec = importlib.util.find_spec(module_name)
    if real_spec is None or real_spec.origin is None:
        raise RuntimeError(f"cannot find spec for {module_name}")
    spec = importlib.util.spec_from_file_location(shadow_alias, real_spec.origin)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# TestCarlHomeHelper
# ---------------------------------------------------------------------------


class TestCarlHomeHelper:
    """Unit tests for :func:`carl_studio.settings.carl_home`."""

    def test_env_set_returns_env_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARL_HOME", str(tmp_path))
        from carl_studio.settings import carl_home

        assert carl_home() == tmp_path.resolve()

    def test_env_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARL_HOME", raising=False)
        from carl_studio.settings import carl_home

        assert carl_home() == (Path.home() / ".carl").resolve()

    def test_expands_tilde(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARL_HOME", "~/custom-carl-home")
        from carl_studio.settings import carl_home

        expected = (Path.home() / "custom-carl-home").resolve()
        assert carl_home() == expected

    def test_resolves_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CARL_HOME", "./rel-carl")
        from carl_studio.settings import carl_home

        # resolve() against the chdir'd cwd should absolute-ify the relative path.
        result = carl_home()
        assert result.is_absolute()
        assert result == (tmp_path / "rel-carl").resolve()

    def test_empty_env_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARL_HOME", "")
        from carl_studio.settings import carl_home

        # Empty string is falsy → treat as unset.
        assert carl_home() == (Path.home() / ".carl").resolve()

    def test_carl_home_snapshot_constant_exposed(self) -> None:
        """The back-compat ``CARL_HOME`` module constant must still exist."""
        from carl_studio import settings

        assert isinstance(settings.CARL_HOME, Path)
        assert settings.GLOBAL_CONFIG == settings.CARL_HOME / "config.yaml"


# ---------------------------------------------------------------------------
# TestDbHonorsCarlHome
# ---------------------------------------------------------------------------


class TestDbHonorsCarlHome:
    """``carl_studio.db.CARL_DIR`` must be derived from :func:`carl_home`."""

    def test_db_carl_dir_uses_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARL_HOME", str(tmp_path))
        db = _shadow_load("carl_studio.db", "_carl_home_shadow_db_env")

        assert db.CARL_DIR == tmp_path.resolve()
        assert db.DB_PATH == tmp_path.resolve() / "carl.db"

    def test_db_carl_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARL_HOME", raising=False)
        db = _shadow_load("carl_studio.db", "_carl_home_shadow_db_default")

        assert db.CARL_DIR == (Path.home() / ".carl").resolve()


# ---------------------------------------------------------------------------
# TestWalletStoreHonorsCarlHome
# ---------------------------------------------------------------------------


class TestWalletStoreHonorsCarlHome:
    """``WalletStore()`` default home must follow :func:`carl_home`.

    ``WalletStore.__init__`` calls ``carl_home()`` live on every construction,
    so no reload is required — setting the env var and instantiating the real
    class proves the wiring.
    """

    def test_wallet_store_default_home_under_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARL_HOME", str(tmp_path))
        from carl_studio.wallet_store import WalletStore

        store = WalletStore()

        # ``path`` is the encrypted envelope path; its parent is the home dir.
        assert store.path.parent == tmp_path.resolve()

    def test_wallet_store_explicit_home_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARL_HOME", str(tmp_path / "env-home"))
        explicit = tmp_path / "explicit-home"
        from carl_studio.wallet_store import WalletStore

        store = WalletStore(home=explicit)

        assert store.path.parent == explicit


# ---------------------------------------------------------------------------
# TestLlmCacheHonorsCarlHome
# ---------------------------------------------------------------------------


class TestLlmCacheHonorsCarlHome:
    """``llm._CACHE_DB`` must live under :func:`carl_home`."""

    def test_llm_cache_db_under_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CARL_HOME", str(tmp_path))
        llm = _shadow_load("carl_studio.llm", "_carl_home_shadow_llm_env")

        assert llm._CACHE_DB == tmp_path.resolve() / "llm_cache.db"

    def test_llm_cache_db_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARL_HOME", raising=False)
        llm = _shadow_load("carl_studio.llm", "_carl_home_shadow_llm_default")

        assert llm._CACHE_DB == (Path.home() / ".carl").resolve() / "llm_cache.db"
