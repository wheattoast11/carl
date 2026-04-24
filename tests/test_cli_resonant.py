"""Tests for ``carl resonant [publish|list|whoami|eval]`` CLI."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from carl_core.eml import EMLNode, EMLOp, EMLTree
from carl_core.resonant import make_resonant


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    import importlib

    import carl_studio.resonant_store as mod

    importlib.reload(mod)
    return tmp_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _resonant():  # noqa: ANN202
    tree = EMLTree(
        root=EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
            right=EMLNode(op=EMLOp.CONST, const=1.0),
        ),
        input_dim=1,
    )
    projection = np.array([[1.0]])
    readout = np.array([[1.0]])
    return make_resonant(tree, projection, readout, metadata={"kind": "test"})


def _app():  # noqa: ANN202
    from carl_studio.cli.resonant import resonant_app

    return resonant_app


class TestWhoami:
    def test_creates_secret_and_prints_fingerprint(self, runner: CliRunner) -> None:
        result = runner.invoke(_app(), ["whoami", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert len(payload["sig_public_component"]) == 32
        assert len(payload["short"]) == 8

    def test_stable_across_invocations(self, runner: CliRunner) -> None:
        first = json.loads(runner.invoke(_app(), ["whoami", "--json"]).stdout)
        second = json.loads(runner.invoke(_app(), ["whoami", "--json"]).stdout)
        assert first == second


class TestList:
    def test_empty_says_no_resonants(self, runner: CliRunner) -> None:
        result = runner.invoke(_app(), ["list"])
        assert result.exit_code == 0
        assert "No local Resonants" in result.output

    def test_lists_saved(self, runner: CliRunner) -> None:
        from carl_studio.resonant_store import save_resonant

        save_resonant("alpha", _resonant())
        save_resonant("beta", _resonant())
        result = runner.invoke(_app(), ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        names = sorted(e["name"] for e in data)
        assert names == ["alpha", "beta"]


class TestEval:
    def test_eval_runs_end_to_end(self, runner: CliRunner) -> None:
        from carl_studio.resonant_store import save_resonant

        save_resonant("e1", _resonant())
        result = runner.invoke(_app(), ["eval", "e1", "--inputs", "[0.5]", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert len(data["action"]) == 1
        assert len(data["latent"]) == 1
        assert len(data["tree_hash"]) == 64

    def test_eval_missing_resonant_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(_app(), ["eval", "nope", "--inputs", "[1.0]"])
        assert result.exit_code != 0

    def test_eval_bad_json_exits_nonzero(self, runner: CliRunner) -> None:
        from carl_studio.resonant_store import save_resonant

        save_resonant("e2", _resonant())
        result = runner.invoke(_app(), ["eval", "e2", "--inputs", "not-json"])
        assert result.exit_code != 0


class TestPublish:
    @pytest.fixture(autouse=True)
    def chdir_and_mkdir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # Project root must NOT equal $HOME (project_context home-guard
        # excludes it). HOME is pinned to tmp_path by isolated_home, so
        # the project lives one level deeper.
        proj = tmp_path / "proj"
        proj.mkdir()
        monkeypatch.chdir(proj)
        (proj / ".carl").mkdir(parents=True, exist_ok=True)
        (proj / "carl.yaml").write_text("name: test\n")

    def test_dry_run_builds_request_without_sending(self, runner: CliRunner) -> None:
        from carl_studio.resonant_store import save_resonant

        save_resonant("p1", _resonant())
        result = runner.invoke(
            _app(),
            ["publish", "p1", "--base-url", "https://example.invalid", "--dry-run", "--json"],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["url"] == "https://example.invalid/api/resonants"
        assert data["body_bytes"] > 0
        assert data["headers"]["X-Carl-User-Secret"] == "<redacted>"
        assert data["headers"]["Content-Type"] == "application/octet-stream"
        assert "X-Carl-Projection" in data["headers"]
        assert "X-Carl-Readout" in data["headers"]

    def test_refuses_http_without_dry_run(self, runner: CliRunner) -> None:
        from carl_studio.resonant_store import save_resonant

        save_resonant("p2", _resonant())
        result = runner.invoke(_app(), ["publish", "p2", "--base-url", "http://example.invalid"])
        assert result.exit_code == 2

    def test_live_post_success(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from carl_studio.resonant_store import save_resonant

        save_resonant("p3", _resonant())
        monkeypatch.setenv("CARL_CAMP_TOKEN", "fake-token")

        captured: dict[str, object] = {}

        def fake_post(url: str, body: bytes, headers: dict[str, str], timeout: float = 20.0):
            captured["url"] = url
            captured["body_len"] = len(body)
            captured["headers"] = dict(headers)
            return 200, json.dumps({"id": "res_abc", "content_hash": "c0ffee"}).encode(), {}

        monkeypatch.setattr("carl_studio.cli.resonant._http_post_bytes", fake_post)

        result = runner.invoke(
            _app(),
            ["publish", "p3", "--base-url", "https://carl.camp", "--json"],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["status"] == "published"
        assert data["resonant_id"] == "res_abc"
        assert captured["url"] == "https://carl.camp/api/resonants"
        assert captured["body_len"] > 0
        h = captured["headers"]
        assert "X-Carl-User-Secret" in h
        assert h["Authorization"] == "Bearer fake-token"
        assert base64.b64decode(h["X-Carl-User-Secret"])

    def test_422_attestation_failure(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from carl_studio.resonant_store import save_resonant

        save_resonant("p4", _resonant())
        monkeypatch.setenv("CARL_CAMP_TOKEN", "fake")

        def fake_post(url, body, headers, timeout=20.0):  # noqa: ANN001
            return 422, b'{"error":"attestation_failed"}', {}

        monkeypatch.setattr("carl_studio.cli.resonant._http_post_bytes", fake_post)
        result = runner.invoke(_app(), ["publish", "p4", "--base-url", "https://carl.camp"])
        assert result.exit_code == 1

    def test_no_token_exits_with_hint(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from carl_studio.resonant_store import save_resonant

        save_resonant("p5", _resonant())
        monkeypatch.delenv("CARL_CAMP_TOKEN", raising=False)
        tok = Path.home() / ".carl" / "camp_token"
        if tok.exists():
            tok.unlink()
        result = runner.invoke(_app(), ["publish", "p5", "--base-url", "https://carl.camp"])
        assert result.exit_code == 1
