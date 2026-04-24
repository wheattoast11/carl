"""Tests for :mod:`carl_studio.cli_session` + ``carl session`` CLI.

Covers:
  * CliSession.start / load / save / list_sessions / current / set_current
  * Schema version enforcement + corruption handling
  * Invalid inputs (bad uuid, non-dict metadata, unreadable files)
  * CARL_CAMP_SESSIONS_SYNC stub (noop default, logs when opted in)
  * Typer CLI: start / resume / list (text + --json modes)
  * Boundary: CliSession does not collide with v0.17 handle-runtime Session
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Per-test project directory."""
    root = tmp_path / "myproject"
    root.mkdir()
    return root


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cd_project(monkeypatch: pytest.MonkeyPatch, project_root: Path) -> Path:
    """Chdir into project_root so cli.session_cmd resolves the right cwd."""
    monkeypatch.chdir(project_root)
    return project_root


def _session_app() -> Any:
    from carl_studio.cli.session_cmd import session_app

    return session_app


# ---------------------------------------------------------------------------
# Model behaviour
# ---------------------------------------------------------------------------


class TestCliSessionStart:
    def test_start_mints_uuid_and_persists(self, project_root: Path) -> None:
        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root, intent="hack the gibson")

        assert uuid.UUID(session.id).version == 4
        assert session.project_root == project_root.resolve()
        assert session.intent == "hack the gibson"
        assert session.status == "active"
        assert session.completed_at is None
        assert session.started_at.tzinfo is not None
        assert session.metadata == {}

        # File landed at the expected path with valid JSON.
        file_path = project_root / ".carl" / "sessions" / f"{session.id}.json"
        assert file_path.is_file()
        payload = json.loads(file_path.read_text())
        assert payload["id"] == session.id
        assert payload["intent"] == "hack the gibson"
        assert payload["schema_version"] == cli_session.SESSION_SCHEMA_VERSION

    def test_start_marks_as_current(self, project_root: Path) -> None:
        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root)
        pointer = project_root / ".carl" / "sessions" / "current.txt"

        assert pointer.read_text().strip() == session.id
        assert cli_session.current(project_root) is not None
        assert cli_session.current(project_root).id == session.id  # type: ignore[union-attr]

    def test_start_accepts_metadata(self, project_root: Path) -> None:
        from carl_studio import cli_session

        session = cli_session.CliSession.start(
            project_root, metadata={"branch": "main", "commit": "abc123"}
        )
        assert session.metadata == {"branch": "main", "commit": "abc123"}

    def test_start_rejects_non_dict_metadata(self, project_root: Path) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        with pytest.raises(ValidationError) as excinfo:
            cli_session.CliSession.start(
                project_root,
                metadata=["not", "a", "dict"],  # type: ignore[arg-type]
            )
        assert excinfo.value.code == "carl.cli_session.invalid_metadata"

    def test_start_rejects_non_json_serializable_metadata(
        self, project_root: Path
    ) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        class NotJson:
            pass

        with pytest.raises(ValidationError) as excinfo:
            cli_session.CliSession.start(
                project_root, metadata={"x": NotJson()}
            )
        assert excinfo.value.code == "carl.cli_session.invalid_metadata"

    def test_start_rejects_non_path_project_root(self, tmp_path: Path) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        with pytest.raises(ValidationError) as excinfo:
            cli_session.CliSession.start(str(tmp_path))  # type: ignore[arg-type]
        assert excinfo.value.code == "carl.cli_session.invalid_project_root"

    def test_start_frozen_model(self, project_root: Path) -> None:
        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root)
        with pytest.raises(Exception):  # Pydantic raises on frozen-mutation
            session.id = "different-id"  # type: ignore[misc]


class TestCliSessionLoad:
    def test_load_roundtrip(self, project_root: Path) -> None:
        from carl_studio import cli_session

        original = cli_session.CliSession.start(
            project_root, intent="roundtrip", metadata={"k": "v"}
        )
        loaded = cli_session.load(original.id, project_root)
        assert loaded.id == original.id
        assert loaded.intent == "roundtrip"
        assert loaded.metadata == {"k": "v"}
        assert loaded.started_at == original.started_at

    def test_load_missing_raises(self, project_root: Path) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        nonexistent = str(uuid.uuid4())
        with pytest.raises(ValidationError) as excinfo:
            cli_session.load(nonexistent, project_root)
        assert excinfo.value.code == "carl.cli_session.not_found"

    def test_load_rejects_invalid_uuid(self, project_root: Path) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        with pytest.raises(ValidationError) as excinfo:
            cli_session.load("not-a-uuid", project_root)
        assert excinfo.value.code == "carl.cli_session.invalid_id"

    def test_load_schema_mismatch(self, project_root: Path) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root)
        path = project_root / ".carl" / "sessions" / f"{session.id}.json"
        raw = json.loads(path.read_text())
        raw["schema_version"] = 99
        path.write_text(json.dumps(raw))

        with pytest.raises(ValidationError) as excinfo:
            cli_session.load(session.id, project_root)
        assert excinfo.value.code == "carl.cli_session.schema_mismatch"

    def test_load_corrupt_json(self, project_root: Path) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root)
        path = project_root / ".carl" / "sessions" / f"{session.id}.json"
        path.write_text("{not valid json")

        with pytest.raises(ValidationError) as excinfo:
            cli_session.load(session.id, project_root)
        assert excinfo.value.code == "carl.cli_session.corrupt"

    def test_load_non_object_payload(self, project_root: Path) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root)
        path = project_root / ".carl" / "sessions" / f"{session.id}.json"
        path.write_text(json.dumps(["not", "an", "object"]))

        with pytest.raises(ValidationError) as excinfo:
            cli_session.load(session.id, project_root)
        assert excinfo.value.code == "carl.cli_session.corrupt"


class TestListSessions:
    def test_list_empty(self, project_root: Path) -> None:
        from carl_studio import cli_session

        assert cli_session.list_sessions(project_root) == []

    def test_list_missing_dir(self, project_root: Path) -> None:
        from carl_studio import cli_session

        # No .carl subdir at all.
        assert cli_session.list_sessions(project_root) == []

    def test_list_sorted_newest_first(self, project_root: Path) -> None:
        from carl_studio import cli_session

        first = cli_session.CliSession.start(project_root, intent="first")
        # Force monotonic ordering on platforms with coarse clocks.
        second_raw = cli_session.CliSession(
            id=str(uuid.uuid4()),
            project_root=first.project_root,
            started_at=first.started_at.replace(year=first.started_at.year + 1),
            intent="second",
        )
        second_raw.save()

        listed = cli_session.list_sessions(project_root)
        assert [s.intent for s in listed] == ["second", "first"]

    def test_list_skips_corrupt_files(
        self, project_root: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from carl_studio import cli_session

        good = cli_session.CliSession.start(project_root, intent="good")
        # Plant a corrupt file with a valid-uuid stem.
        bad_id = str(uuid.uuid4())
        bad_path = project_root / ".carl" / "sessions" / f"{bad_id}.json"
        bad_path.write_text("{broken json")

        caplog.set_level(logging.WARNING, logger="carl_studio.cli_session")
        listed = cli_session.list_sessions(project_root)
        assert [s.id for s in listed] == [good.id]
        assert any("corrupt session file" in rec.message for rec in caplog.records)

    def test_list_ignores_non_uuid_stems(self, project_root: Path) -> None:
        from carl_studio import cli_session

        cli_session.CliSession.start(project_root, intent="real")
        # Plant a noise file.
        noise = project_root / ".carl" / "sessions" / "random-notes.json"
        noise.write_text(json.dumps({"note": "hi"}))

        listed = cli_session.list_sessions(project_root)
        assert len(listed) == 1
        assert listed[0].intent == "real"


class TestCurrent:
    def test_current_none_when_empty(self, project_root: Path) -> None:
        from carl_studio import cli_session

        assert cli_session.current(project_root) is None

    def test_current_follows_start(self, project_root: Path) -> None:
        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root)
        seen = cli_session.current(project_root)
        assert seen is not None and seen.id == session.id

    def test_current_returns_none_on_dangling_pointer(
        self, project_root: Path
    ) -> None:
        from carl_studio import cli_session

        pointer = project_root / ".carl" / "sessions" / "current.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(uuid.uuid4()))

        assert cli_session.current(project_root) is None

    def test_current_returns_none_on_malformed_pointer(
        self, project_root: Path
    ) -> None:
        from carl_studio import cli_session

        pointer = project_root / ".carl" / "sessions" / "current.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text("not-a-uuid")

        assert cli_session.current(project_root) is None

    def test_set_current_after_load(self, project_root: Path) -> None:
        from carl_studio import cli_session

        first = cli_session.CliSession.start(project_root, intent="first")
        second = cli_session.CliSession.start(project_root, intent="second")

        # second is current after its start().
        assert cli_session.current(project_root).id == second.id  # type: ignore[union-attr]
        cli_session.set_current(project_root, first.id)
        assert cli_session.current(project_root).id == first.id  # type: ignore[union-attr]

    def test_set_current_rejects_unknown_session(
        self, project_root: Path
    ) -> None:
        from carl_core.errors import ValidationError

        from carl_studio import cli_session

        with pytest.raises(ValidationError):
            cli_session.set_current(project_root, str(uuid.uuid4()))


class TestLifecycleTransitions:
    def test_complete_transitions_status_and_stamps_completed_at(
        self, project_root: Path
    ) -> None:
        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root)
        done = session.complete()

        assert done.status == "completed"
        assert done.completed_at is not None
        assert done.completed_at.tzinfo is not None
        # Re-loading from disk reflects the new state.
        loaded = cli_session.load(session.id, project_root)
        assert loaded.status == "completed"
        assert loaded.completed_at is not None

    def test_abandon_transitions_status(self, project_root: Path) -> None:
        from carl_studio import cli_session

        session = cli_session.CliSession.start(project_root)
        abandoned = session.abandon()

        assert abandoned.status == "abandoned"
        loaded = cli_session.load(session.id, project_root)
        assert loaded.status == "abandoned"


class TestSyncStub:
    def test_noop_by_default(
        self,
        project_root: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from carl_studio import cli_session

        monkeypatch.delenv("CARL_CAMP_SESSIONS_SYNC", raising=False)
        caplog.set_level(logging.WARNING, logger="carl_studio.cli_session")

        cli_session.CliSession.start(project_root)

        assert not any(
            "sync not wired" in rec.message for rec in caplog.records
        )

    def test_logs_todo_when_opted_in(
        self,
        project_root: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from carl_studio import cli_session

        monkeypatch.setenv("CARL_CAMP_SESSIONS_SYNC", "1")
        caplog.set_level(logging.WARNING, logger="carl_studio.cli_session")

        cli_session.CliSession.start(project_root)

        assert any(
            "sync not wired; platform endpoint pending" in rec.message
            for rec in caplog.records
        )

    def test_no_network_calls(
        self,
        project_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Defensive: the stub must never attempt HTTP even when opted in."""
        import socket

        from carl_studio import cli_session

        def _deny_socket(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("sync stub attempted a socket connection")

        monkeypatch.setenv("CARL_CAMP_SESSIONS_SYNC", "1")
        monkeypatch.setattr(socket.socket, "connect", _deny_socket)

        session = cli_session.CliSession.start(project_root)
        session.complete()


class TestNamingBoundary:
    def test_cli_session_does_not_collide_with_handle_runtime_session(
        self, project_root: Path
    ) -> None:
        """Regression: v0.17 Session + v0.18 CliSession must import cleanly."""
        from carl_studio import cli_session
        from carl_studio.session import Session as HandleSession

        cli = cli_session.CliSession.start(project_root)
        handle = HandleSession(user="tej")
        try:
            assert cli.id != getattr(handle, "id", None)  # no accidental merge
            assert handle.chain is not None
        finally:
            handle.teardown()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestSessionStartCmd:
    def test_start_json(self, runner: CliRunner, cd_project: Path) -> None:
        result = runner.invoke(_session_app(), ["start", "--json", "--intent", "debug"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert uuid.UUID(payload["id"]).version == 4
        assert payload["short"] == payload["id"][:8]
        assert payload["intent"] == "debug"
        assert payload["status"] == "active"

    def test_start_text(self, runner: CliRunner, cd_project: Path) -> None:
        result = runner.invoke(_session_app(), ["start", "--intent", "text"])
        assert result.exit_code == 0, result.output
        assert "Session started" in result.output
        assert "text" in result.output

    def test_start_with_metadata(
        self, runner: CliRunner, cd_project: Path
    ) -> None:
        result = runner.invoke(
            _session_app(),
            ["start", "--metadata", '{"branch": "main"}', "--json"],
        )
        assert result.exit_code == 0, result.output
        session_id = json.loads(result.stdout)["id"]

        from carl_studio import cli_session

        loaded = cli_session.load(session_id, cd_project)
        assert loaded.metadata == {"branch": "main"}

    def test_start_rejects_bad_metadata_json(
        self, runner: CliRunner, cd_project: Path
    ) -> None:
        result = runner.invoke(
            _session_app(), ["start", "--metadata", "{not valid"]
        )
        assert result.exit_code == 2, result.output
        assert "invalid --metadata JSON" in result.output or result.stderr

    def test_start_rejects_non_object_metadata(
        self, runner: CliRunner, cd_project: Path
    ) -> None:
        result = runner.invoke(
            _session_app(), ["start", "--metadata", '["not", "a", "dict"]']
        )
        assert result.exit_code == 2, result.output


class TestSessionResumeCmd:
    def test_resume_existing(self, runner: CliRunner, cd_project: Path) -> None:
        from carl_studio import cli_session

        first = cli_session.CliSession.start(cd_project, intent="first")
        second = cli_session.CliSession.start(cd_project, intent="second")
        # second is now current; resume first.
        assert cli_session.current(cd_project).id == second.id  # type: ignore[union-attr]

        result = runner.invoke(_session_app(), ["resume", first.id, "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["id"] == first.id
        assert payload["resumed"] is True

        # Pointer moved.
        assert cli_session.current(cd_project).id == first.id  # type: ignore[union-attr]

    def test_resume_missing(self, runner: CliRunner, cd_project: Path) -> None:
        result = runner.invoke(_session_app(), ["resume", str(uuid.uuid4())])
        assert result.exit_code == 1
        assert "session resume failed" in result.output

    def test_resume_invalid_id(self, runner: CliRunner, cd_project: Path) -> None:
        result = runner.invoke(_session_app(), ["resume", "not-a-uuid"])
        assert result.exit_code == 1

    def test_resume_completed_session_warns_but_succeeds(
        self, runner: CliRunner, cd_project: Path
    ) -> None:
        from carl_studio import cli_session

        session = cli_session.CliSession.start(cd_project)
        session.complete()

        result = runner.invoke(_session_app(), ["resume", session.id])
        assert result.exit_code == 0, result.output
        assert "completed" in result.output


class TestSessionListCmd:
    def test_list_empty(self, runner: CliRunner, cd_project: Path) -> None:
        result = runner.invoke(_session_app(), ["list"])
        assert result.exit_code == 0, result.output
        assert "No sessions yet" in result.output

    def test_list_empty_json(self, runner: CliRunner, cd_project: Path) -> None:
        result = runner.invoke(_session_app(), ["list", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == []

    def test_list_populated_json_marks_current(
        self, runner: CliRunner, cd_project: Path
    ) -> None:
        from carl_studio import cli_session

        first = cli_session.CliSession.start(cd_project, intent="first")
        second = cli_session.CliSession.start(cd_project, intent="second")
        # second is current.

        result = runner.invoke(_session_app(), ["list", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert {p["intent"] for p in payload} == {"first", "second"}
        current_flags = {p["id"]: p["is_current"] for p in payload}
        assert current_flags[second.id] is True
        assert current_flags[first.id] is False

    def test_list_with_all_flag_prints_limitation_hint(
        self, runner: CliRunner, cd_project: Path
    ) -> None:
        from carl_studio import cli_session

        cli_session.CliSession.start(cd_project)
        result = runner.invoke(_session_app(), ["list", "--all"])
        assert result.exit_code == 0, result.output
        assert "v0.18.1" in result.output

    def test_list_text_renders_table(
        self, runner: CliRunner, cd_project: Path
    ) -> None:
        from carl_studio import cli_session

        cli_session.CliSession.start(cd_project, intent="hello")
        result = runner.invoke(_session_app(), ["list"])
        assert result.exit_code == 0, result.output
        assert "hello" in result.output


class TestWiringRegistration:
    def test_session_app_registered_on_root_app(self) -> None:
        """`carl session` must appear on the top-level Typer app."""
        from typer.main import get_command

        from carl_studio.cli.apps import app
        from carl_studio.cli import wiring  # noqa: F401 — side-effect import

        command = get_command(app)
        assert "session" in command.commands
