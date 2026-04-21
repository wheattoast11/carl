"""Tests for the ``carl contract constitution`` CLI surface."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from carl_studio.cli import app


# ---------------------------------------------------------------------------
# help + command registration
# ---------------------------------------------------------------------------


def test_contract_constitution_help_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["contract", "--help"])
    assert result.exit_code == 0
    assert "constitution" in result.output


def test_constitution_help_lists_subactions() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["contract", "constitution", "--help"])
    assert result.exit_code == 0
    # subactions documented in the docstring
    for sub in ("genesis", "verify", "evaluate", "status"):
        assert sub in result.output


# ---------------------------------------------------------------------------
# genesis + verify roundtrip
# ---------------------------------------------------------------------------


def test_constitution_genesis_creates_ledger(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "ledger"
    result = runner.invoke(
        app,
        [
            "contract",
            "constitution",
            "genesis",
            "--ledger-root",
            str(root),
            "--threshold",
            "0.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Genesis block written" in result.output
    assert (root / "policy.json").exists()
    assert (root / "chain.jsonl").exists()
    assert (root / "signing_key.bin").exists()


def test_constitution_verify_fresh_ledger(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "ledger"
    genesis = runner.invoke(
        app,
        ["contract", "constitution", "genesis", "--ledger-root", str(root)],
    )
    assert genesis.exit_code == 0, genesis.output
    verify = runner.invoke(
        app,
        ["contract", "constitution", "verify", "--ledger-root", str(root)],
    )
    assert verify.exit_code == 0, verify.output
    assert "chain valid" in verify.output


def test_constitution_evaluate_allow_and_deny(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "ledger"
    runner.invoke(
        app,
        ["contract", "constitution", "genesis", "--ledger-root", str(root), "--threshold", "0.0"],
    )
    # A high coherence_phi should produce ALLOW under the default exp(phi) policy.
    allow = runner.invoke(
        app,
        [
            "contract",
            "constitution",
            "evaluate",
            "--ledger-root",
            str(root),
            "--action-json",
            '{"type": "TOOL", "coherence_phi": 2.0}',
        ],
    )
    assert allow.exit_code == 0, allow.output
    assert "ALLOW" in allow.output

    # With a huge threshold forced via a fresh ledger, DENY should exit nonzero.
    root2 = tmp_path / "ledger2"
    runner.invoke(
        app,
        [
            "contract",
            "constitution",
            "genesis",
            "--ledger-root",
            str(root2),
            "--threshold",
            "1e12",
        ],
    )
    deny = runner.invoke(
        app,
        [
            "contract",
            "constitution",
            "evaluate",
            "--ledger-root",
            str(root2),
            "--action-json",
            '{"type": "GATE", "coherence_phi": 0.0}',
        ],
    )
    assert deny.exit_code == 1
    assert "DENY" in deny.output


def test_constitution_status_before_and_after_genesis(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "ledger"
    pre = runner.invoke(
        app, ["contract", "constitution", "status", "--ledger-root", str(root)]
    )
    assert pre.exit_code == 0
    assert "no genesis" in pre.output.lower() or "genesis" in pre.output.lower()

    runner.invoke(
        app, ["contract", "constitution", "genesis", "--ledger-root", str(root)]
    )
    post = runner.invoke(
        app, ["contract", "constitution", "status", "--ledger-root", str(root)]
    )
    assert post.exit_code == 0
    assert "height" in post.output.lower()


def test_constitution_unknown_action_rejected(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "ledger"
    result = runner.invoke(
        app,
        ["contract", "constitution", "nonsense", "--ledger-root", str(root)],
    )
    assert result.exit_code == 2
    assert "unknown" in result.output.lower()


def test_constitution_evaluate_requires_action_json(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "ledger"
    runner.invoke(
        app, ["contract", "constitution", "genesis", "--ledger-root", str(root)]
    )
    result = runner.invoke(
        app, ["contract", "constitution", "evaluate", "--ledger-root", str(root)]
    )
    assert result.exit_code == 2
    assert "--action-json" in result.output


def test_constitution_genesis_with_custom_policy(tmp_path: Path) -> None:
    """Genesis with a policy loaded from disk."""
    from carl_core.constitutional import ConstitutionalPolicy
    from carl_studio.fsm_ledger import build_default_policy

    pol_path = tmp_path / "policy.json"
    pol = build_default_policy(threshold=0.1)
    pol.save(pol_path)

    runner = CliRunner()
    root = tmp_path / "ledger"
    result = runner.invoke(
        app,
        [
            "contract",
            "constitution",
            "genesis",
            "--ledger-root",
            str(root),
            "--policy-path",
            str(pol_path),
        ],
    )
    assert result.exit_code == 0, result.output
    loaded = ConstitutionalPolicy.load(root / "policy.json")
    assert loaded.policy_id == pol.policy_id
