"""Regression: `carl camp login` JWT must be readable by publish commands.

Bug (2026-04-24): `carl camp login` writes the JWT to ``LocalDB.set_auth("jwt")``
(i.e. ``~/.carl/carl.db``), but both ``carl agent publish`` and
``carl resonant publish`` resolve the bearer token via env var + a
legacy ``~/.carl/camp_token`` file. After a successful login, neither
publish command can find the token — the user is prompted to
"run `carl camp login` first" despite just having done so.

Fix: extend both ``_resolve_bearer_token`` implementations to fall back
to ``LocalDB.get_auth("jwt")`` when no env var and no legacy file.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate HOME + CARL_HOME + db.DB_PATH so no real ~/.carl leaks in.

    ``db.DB_PATH`` is frozen at import time from ``carl_home()`` — setting
    CARL_HOME via env after import doesn't rebind it. We monkeypatch the
    module-level constant directly.
    """
    monkeypatch.delenv("CARL_CAMP_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    carl_dir = tmp_path / ".carl"
    carl_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CARL_HOME", str(carl_dir))

    from carl_studio import db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", carl_dir / "carl.db")
    return tmp_path


class TestA2ABearerFallbackToLocalDB:
    def test_resolves_jwt_from_localdb_when_no_env_no_file(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.a2a import _cli as a2a_cli
        from carl_studio.db import LocalDB

        db = LocalDB()
        db.set_auth("jwt", "test-jwt-from-login", ttl_hours=1)

        token = a2a_cli._resolve_bearer_token()  # pyright: ignore[reportPrivateUsage]
        assert token == "test-jwt-from-login", (
            "agent publish must find the JWT that `carl camp login` stores in LocalDB"
        )

    def test_env_var_still_wins_over_localdb(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.a2a import _cli as a2a_cli
        from carl_studio.db import LocalDB

        db = LocalDB()
        db.set_auth("jwt", "from-localdb", ttl_hours=1)
        monkeypatch.setenv("CARL_CAMP_TOKEN", "from-env")

        assert a2a_cli._resolve_bearer_token() == "from-env"  # pyright: ignore[reportPrivateUsage]

    def test_legacy_file_still_wins_over_localdb(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.a2a import _cli as a2a_cli
        from carl_studio.db import LocalDB

        db = LocalDB()
        db.set_auth("jwt", "from-localdb", ttl_hours=1)
        (clean_env / ".carl" / "camp_token").write_text("from-file\n")

        # Legacy file path is documented as #2 in resolution order; LocalDB is #3.
        assert a2a_cli._resolve_bearer_token() == "from-file"  # pyright: ignore[reportPrivateUsage]

    def test_returns_none_when_all_three_sources_empty(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.a2a import _cli as a2a_cli

        # No env, no file, no LocalDB entry.
        assert a2a_cli._resolve_bearer_token() is None  # pyright: ignore[reportPrivateUsage]


class TestResonantBearerFallbackToLocalDB:
    def test_resolves_jwt_from_localdb_when_no_env_no_file(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.cli import resonant as res_mod
        from carl_studio.db import LocalDB

        db = LocalDB()
        db.set_auth("jwt", "test-jwt-from-login", ttl_hours=1)

        token = res_mod._resolve_bearer_token()  # pyright: ignore[reportPrivateUsage]
        assert token == "test-jwt-from-login", (
            "resonant publish must find the JWT that `carl camp login` stores in LocalDB"
        )

    def test_env_var_still_wins_over_localdb(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.cli import resonant as res_mod
        from carl_studio.db import LocalDB

        db = LocalDB()
        db.set_auth("jwt", "from-localdb", ttl_hours=1)
        monkeypatch.setenv("CARL_CAMP_TOKEN", "from-env")

        assert res_mod._resolve_bearer_token() == "from-env"  # pyright: ignore[reportPrivateUsage]
