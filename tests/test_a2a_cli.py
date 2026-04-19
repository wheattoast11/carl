"""Tests for ``carl agent send`` input validation (UAT-050).

Covers:

* ``--inputs`` must be a JSON **object** — bare strings, numbers,
  lists, ``null`` and booleans must be rejected at the CLI boundary
  before a malformed task lands on the bus.
* The ``skill`` argument must name a registered skill — dispatching
  ``made_up_skill`` used to queue an unexecutable task; it now fails
  fast with a helpful error listing available skills.
* Happy path still works (regression guard).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from carl_studio.a2a._cli import agent_app

# Newer Click (>=8.2) dropped the ``mix_stderr`` parameter and merges
# stderr into ``result.output`` by default. We assert against the
# combined stream, which keeps the tests portable across Click
# versions.
runner = CliRunner()


# ---------------------------------------------------------------------------
# --inputs must be a JSON object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input",
    [
        '"just-a-string"',
        "42",
        "3.14",
        "[1, 2, 3]",
        "null",
        "true",
        "false",
    ],
)
def test_send_rejects_non_dict_input(tmp_path: Path, bad_input: str) -> None:
    """Non-object JSON values must be rejected before the bus is touched.

    We patch ``LocalBus`` so that if the CLI accidentally posts a task
    with a malformed ``inputs`` the test would surface via a mock
    assertion rather than a silent pass.
    """
    with patch("carl_studio.a2a.bus.LocalBus") as mock_bus_cls:
        result = runner.invoke(
            agent_app,
            ["send", "observer", "--inputs", bad_input],
        )

    assert result.exit_code == 1, (
        f"expected non-zero exit for input={bad_input!r}, got output={result.output!r}"
    )
    assert "must be a JSON object" in result.output
    # The bus must never have been instantiated — we failed fast at the
    # validation layer, before the skill-registry check even.
    assert not mock_bus_cls.called


def test_send_rejects_invalid_json(tmp_path: Path) -> None:
    """Syntactically bad JSON still produces the pre-existing error path."""
    result = runner.invoke(
        agent_app,
        ["send", "observer", "--inputs", "{not-json"],
    )
    assert result.exit_code == 1
    assert "not valid JSON" in result.output


# ---------------------------------------------------------------------------
# skill must be in the registry
# ---------------------------------------------------------------------------


def test_send_rejects_unregistered_skill(tmp_path: Path) -> None:
    """Unknown skill names are rejected with an allowlist in the error."""
    with patch("carl_studio.a2a.bus.LocalBus") as mock_bus_cls:
        result = runner.invoke(
            agent_app,
            ["send", "made_up_skill_does_not_exist", "--inputs", "{}"],
        )

    assert result.exit_code == 1
    assert "unknown skill" in result.output.lower()
    assert "made_up_skill_does_not_exist" in result.output
    # Available skills are surfaced to the caller — at least one of the
    # builtins should appear so the user knows what to try.
    assert any(
        name in result.output
        for name in ("observer", "grader", "trainer", "synthesizer", "deployer")
    ), f"expected builtin skills in error, got: {result.output!r}"
    # Bus was never constructed — the allowlist blocked dispatch.
    assert not mock_bus_cls.called


# ---------------------------------------------------------------------------
# Happy path regression guard
# ---------------------------------------------------------------------------


def test_send_registered_skill_with_object_inputs_succeeds(tmp_path: Path) -> None:
    """Known skill + valid JSON object posts to the bus and prints the id."""
    from carl_studio.a2a.task import A2ATask

    posted: dict[str, object] = {}

    class _FakeBus:
        def post(self, task: A2ATask) -> str:
            posted["task"] = task
            return task.id

        def close(self) -> None:
            posted["closed"] = True

    with patch("carl_studio.a2a.bus.LocalBus", return_value=_FakeBus()):
        result = runner.invoke(
            agent_app,
            [
                "send",
                "observer",
                "--inputs",
                '{"foo": "bar"}',
                "--sender",
                "tester",
            ],
        )

    assert result.exit_code == 0, (
        f"expected success, got output={result.output!r}"
    )
    assert "task" in posted, "post() was never called"
    task = posted["task"]
    assert isinstance(task, A2ATask)
    assert task.skill == "observer"
    assert task.inputs == {"foo": "bar"}
    assert task.sender == "tester"
    # The printed id must match the task id.
    assert task.id in result.output
    assert posted.get("closed") is True
