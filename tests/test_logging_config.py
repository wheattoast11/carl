"""Tests for carl_studio.logging_config.configure_logging.

Covers:
  * Idempotency — re-calling configure_logging() without force=True leaves
    the existing handler alone.
  * JSON formatter — emits single-line JSON with level, logger, message,
    and any extra attributes passed via logger.info(..., extra={...}).
  * Level honoring — explicit level wins; CARL_LOG_LEVEL env fallback;
    unknown values collapse to INFO rather than raising.
  * CARL_LOG_JSON env toggles JSON output when json_output=None.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import pytest

from carl_studio.logging_config import configure_logging


@pytest.fixture(autouse=True)
def _reset_root_logger() -> Any:
    """Snapshot + restore the root logger between tests.

    configure_logging() mutates the root logger, so every test either
    needs a fresh state or a clean teardown. Yielding preserves the
    original handler list to prevent cross-test interference.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def test_configure_installs_single_handler() -> None:
    configure_logging(force=True)
    root = logging.getLogger()
    assert len(root.handlers) == 1


def test_idempotent_without_force() -> None:
    """A second call without force=True must not add a duplicate handler."""
    configure_logging(force=True)
    assert len(logging.getLogger().handlers) == 1

    configure_logging()  # no force — should no-op
    configure_logging()
    assert len(logging.getLogger().handlers) == 1


def test_force_replaces_handlers() -> None:
    configure_logging(force=True, level="info")
    first_handler = logging.getLogger().handlers[0]

    configure_logging(force=True, level="debug")
    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    # force=True should have created a new handler instance.
    assert handlers[0] is not first_handler


def test_level_honoring_explicit() -> None:
    configure_logging(force=True, level="debug")
    assert logging.getLogger().level == logging.DEBUG

    configure_logging(force=True, level="warning")
    assert logging.getLogger().level == logging.WARNING


def test_level_honoring_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARL_LOG_LEVEL", "error")
    configure_logging(force=True)
    assert logging.getLogger().level == logging.ERROR


def test_level_unknown_value_falls_back_to_info() -> None:
    # Unknown level names should not raise — fall back to INFO.
    configure_logging(force=True, level="not-a-real-level")
    assert logging.getLogger().level == logging.INFO


def test_level_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARL_LOG_LEVEL", "DEBUG")
    configure_logging(force=True)
    assert logging.getLogger().level == logging.DEBUG


def test_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(force=True, level="debug", json_output=True)
    logger = logging.getLogger("carl_test.json_format")
    logger.info("hello-world")

    captured = capsys.readouterr()
    # Root handler writes to stderr.
    line = captured.err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "carl_test.json_format"
    assert payload["message"] == "hello-world"
    assert "ts" in payload


def test_json_format_extras(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(force=True, level="debug", json_output=True)
    logger = logging.getLogger("carl_test.extras")
    logger.info("with-extras", extra={"run_id": "abc-123", "step": 7})

    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip().splitlines()[-1])
    assert payload["run_id"] == "abc-123"
    assert payload["step"] == 7


def test_json_env_toggle(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CARL_LOG_JSON=1 should flip on JSON output when json_output=None."""
    monkeypatch.setenv("CARL_LOG_JSON", "1")
    configure_logging(force=True, level="info")

    logging.getLogger("carl_test.env").info("env-json")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    # The line should parse as JSON.
    payload = json.loads(line)
    assert payload["message"] == "env-json"


def test_json_env_disabled_emits_text(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without CARL_LOG_JSON, default formatter should emit plain text."""
    monkeypatch.delenv("CARL_LOG_JSON", raising=False)
    configure_logging(force=True, level="info")

    logging.getLogger("carl_test.text").info("plain-text")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    assert "plain-text" in line
    # Plain format is not valid JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_json_formatter_handles_non_serializable_extra(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-JSON-serializable extras should fall back to repr, not crash."""
    configure_logging(force=True, level="debug", json_output=True)
    logger = logging.getLogger("carl_test.repr")

    class _Opaque:
        def __repr__(self) -> str:
            return "<Opaque obj>"

    logger.info("opaque", extra={"obj": _Opaque()})
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip().splitlines()[-1])
    assert payload["obj"] == "<Opaque obj>"


def test_handler_writes_to_stderr() -> None:
    configure_logging(force=True)
    root = logging.getLogger()
    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr
