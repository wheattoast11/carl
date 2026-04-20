"""Tests for ``carl_studio.run_diff`` — aggregate + per-step diffs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import carl_studio.db as db_mod
from carl_studio.cli import app
from carl_studio.run_diff import (
    RunDiffReport,
    RunSummary,
    compute_diff,
    summarize_run,
)


# ---------------------------------------------------------------------------
# summarize_run
# ---------------------------------------------------------------------------


def test_summarize_run_populates_from_result_dict():
    run = {
        "id": "run-a",
        "status": "completed",
        "result": {
            "phi_mean": 0.42,
            "q_hat": 0.81,
            "sample_size": 128,
            "contraction_holds": True,
            "crystallization_count": 3,
        },
    }
    metrics = [{"step": 0, "data": {"phi": 0.1}}, {"step": 1, "data": {"phi": 0.2}}]
    summary = summarize_run(run, metrics)
    assert summary.run_id == "run-a"
    assert summary.phi_mean == pytest.approx(0.42)
    assert summary.q_hat == pytest.approx(0.81)
    assert summary.sample_size == 128
    assert summary.contraction_holds is True
    assert summary.crystallization_count == 3
    assert summary.step_count == 2
    assert summary.status == "completed"


def test_summarize_run_handles_missing_result_fields():
    run = {"id": "run-b", "status": "training", "result": None}
    summary = summarize_run(run, [])
    assert summary.run_id == "run-b"
    assert summary.phi_mean is None
    assert summary.q_hat is None
    assert summary.sample_size is None
    assert summary.contraction_holds is None
    assert summary.crystallization_count is None
    assert summary.step_count == 0
    assert summary.status == "training"


def test_summarize_run_handles_json_string_result():
    payload = {
        "phi_mean": 0.55,
        "q_hat": 0.7,
        "crystallizations": [{"step": 3}, {"step": 7}],
    }
    run = {"id": "run-c", "result": json.dumps(payload)}
    summary = summarize_run(run, [])
    assert summary.phi_mean == pytest.approx(0.55)
    assert summary.q_hat == pytest.approx(0.7)
    # falls back to len(crystallizations) when count is missing
    assert summary.crystallization_count == 2


# ---------------------------------------------------------------------------
# compute_diff — aggregate
# ---------------------------------------------------------------------------


def test_compute_diff_aggregates_scalars():
    run_a = {
        "id": "a",
        "result": {"phi_mean": 0.4, "q_hat": 0.7, "sample_size": 100, "crystallization_count": 2},
    }
    run_b = {
        "id": "b",
        "result": {"phi_mean": 0.6, "q_hat": 0.9, "sample_size": 150, "crystallization_count": 5},
    }
    report = compute_diff(run_a, [], run_b, [])
    assert isinstance(report, RunDiffReport)
    assert report.phi_mean_delta == pytest.approx(0.2)
    assert report.q_hat_delta == pytest.approx(0.2)
    assert report.sample_size_delta == 50
    assert report.crystallization_delta == 3
    assert report.contraction_holds_change is None
    assert report.step_rows == []


def test_compute_diff_handles_one_side_missing_q_hat():
    run_a = {"id": "a", "result": {"phi_mean": 0.4, "q_hat": 0.7}}
    run_b = {"id": "b", "result": {"phi_mean": 0.5}}  # q_hat missing
    report = compute_diff(run_a, [], run_b, [])
    assert report.phi_mean_delta == pytest.approx(0.1)
    assert report.q_hat_delta is None
    assert report.a.q_hat == pytest.approx(0.7)
    assert report.b.q_hat is None


def test_compute_diff_contraction_holds_change_string():
    run_a = {"id": "a", "result": {"contraction_holds": True}}
    run_b = {"id": "b", "result": {"contraction_holds": False}}
    report = compute_diff(run_a, [], run_b, [])
    assert report.contraction_holds_change == "True->False"

    # Same on both sides → None
    run_b2 = {"id": "b", "result": {"contraction_holds": True}}
    report2 = compute_diff(run_a, [], run_b2, [])
    assert report2.contraction_holds_change is None


# ---------------------------------------------------------------------------
# compute_diff — per-step
# ---------------------------------------------------------------------------


def test_compute_diff_step_rows_when_steps_true():
    metrics_a = [
        {"step": 0, "data": {"phi": 0.1}},
        {"step": 1, "data": {"phi": 0.2}},
        {"step": 2, "data": {"phi": 0.3}},
    ]
    metrics_b = [
        {"step": 0, "data": {"phi": 0.15}},
        {"step": 1, "data": {"phi": 0.25}},
    ]
    report = compute_diff({"id": "a"}, metrics_a, {"id": "b"}, metrics_b, steps=True)
    assert len(report.step_rows) == 2  # min(3, 2)
    assert report.step_rows[0] == {
        "step": 0,
        "phi_a": pytest.approx(0.1),
        "phi_b": pytest.approx(0.15),
        "delta": pytest.approx(0.05),
    }
    assert report.step_rows[1]["step"] == 1


def test_compute_diff_step_rows_empty_when_steps_false():
    metrics_a = [{"step": 0, "data": {"phi": 0.1}}]
    metrics_b = [{"step": 0, "data": {"phi": 0.2}}]
    report = compute_diff({"id": "a"}, metrics_a, {"id": "b"}, metrics_b, steps=False)
    assert report.step_rows == []


def test_compute_diff_divergence_step_returned():
    metrics_a = [
        {"step": 0, "data": {"phi": 0.10}},
        {"step": 1, "data": {"phi": 0.20}},
        {"step": 2, "data": {"phi": 0.30}},
    ]
    metrics_b = [
        {"step": 0, "data": {"phi": 0.12}},  # |Δ|=0.02, below 0.1
        {"step": 1, "data": {"phi": 0.25}},  # |Δ|=0.05, below 0.1
        {"step": 2, "data": {"phi": 0.50}},  # |Δ|=0.20, above 0.1
    ]
    report = compute_diff({"id": "a"}, metrics_a, {"id": "b"}, metrics_b)
    assert report.divergence_step == 2


def test_compute_diff_divergence_none_when_all_within_threshold():
    metrics_a = [{"step": i, "data": {"phi": 0.1 * i}} for i in range(5)]
    metrics_b = [{"step": i, "data": {"phi": 0.1 * i + 0.01}} for i in range(5)]
    report = compute_diff({"id": "a"}, metrics_a, {"id": "b"}, metrics_b)
    assert report.divergence_step is None


def test_compute_diff_summaries_attached():
    run_a = {"id": "a", "status": "completed", "result": {"phi_mean": 0.3}}
    run_b = {"id": "b", "status": "failed", "result": {"phi_mean": 0.4}}
    report = compute_diff(run_a, [], run_b, [])
    assert isinstance(report.a, RunSummary)
    assert report.a.status == "completed"
    assert report.b.status == "failed"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_run_diff_missing_run_returns_nonzero(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(db_mod, "CARL_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "carl.db")

    runner = CliRunner()
    result = runner.invoke(app, ["run", "diff", "nope-a", "nope-b"])
    assert result.exit_code == 2
    assert "run not found" in (result.stderr or result.output).lower()


def test_cli_run_diff_happy_path(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(db_mod, "CARL_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "carl.db")

    from carl_studio.db import LocalDB

    db = LocalDB()
    db.insert_run(
        {
            "id": "run-aa",
            "model_id": "org/m",
            "mode": "train",
            "status": "completed",
            "config": {},
            "result": {"phi_mean": 0.4, "q_hat": 0.7, "crystallization_count": 1},
        }
    )
    db.insert_run(
        {
            "id": "run-bb",
            "model_id": "org/m",
            "mode": "train",
            "status": "completed",
            "config": {},
            "result": {"phi_mean": 0.5, "q_hat": 0.8, "crystallization_count": 2},
        }
    )
    db.insert_metric("run-aa", 0, {"phi": 0.1})
    db.insert_metric("run-bb", 0, {"phi": 0.15})

    runner = CliRunner()
    result = runner.invoke(app, ["run", "diff", "run-aa", "run-bb"])
    assert result.exit_code == 0, result.output
    assert "run-aa" in result.output
    assert "run-bb" in result.output
