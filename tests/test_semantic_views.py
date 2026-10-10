"""Accepted view head: scoring moves from the 256 prefix to the judged int8 view, prefix behind the flag."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
from carl_studio.semantic.views import (
    SHIPPED,
    ViewHead,
    ViewHeadAcceptance,
    activate_view_head,
    load_view_head,
)

from carl_studio.db import LocalDB
from carl_studio.semantic import types, views
from carl_studio.semantic.types import Carrier


@pytest.fixture(autouse=True)
def reset_heads(monkeypatch):  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv(views.FLAG, raising=False)
    types.clear_view_heads()
    yield
    types.clear_view_heads()


def carrier(seed: int = 3) -> Carrier:
    rng = random.Random(seed)
    return Carrier(values=tuple(rng.gauss(0.0, 1.0) for _ in range(768)))


def prefix_scoring(vector: Carrier, dimension: int) -> tuple[float, ...]:
    raw = vector.values[:dimension]
    norm = math.hypot(*raw)
    return tuple(v / norm for v in raw)


def test_shipped_head_is_accepted_and_replaces_the_256_prefix():
    head = load_view_head(SHIPPED)
    assert head.width == 256 and head.element_bytes == 1 and head.acceptance.view_bytes == 256
    assert head.acceptance.judge["passed"] and head.acceptance.judge["rows"]["candidate"] >= 0.9
    vector = carrier()
    view = vector.scoring(256)
    assert len(view) == 256
    assert abs(math.hypot(*view) - 1.0) < 1e-6
    assert view != prefix_scoring(vector, 256)
    raw = head.weights @ np.asarray(vector.values, dtype=np.float32)
    raw = raw / np.linalg.norm(raw)
    grid = np.round(raw / np.abs(raw).max() * 127) / 127
    expected = grid / np.linalg.norm(grid)
    assert np.allclose(np.asarray(view), expected, atol=1e-6)
    scaled = np.asarray(view) / np.abs(view).max() * 127
    assert np.allclose(scaled, np.round(scaled), atol=1e-3)
    for dimension in (128, 512, 768):
        assert vector.scoring(dimension) == prefix_scoring(vector, dimension)
    assert vector.score(vector, 256) == pytest.approx(1.0)
    other = carrier(4)
    assert vector.score(other, 256) == pytest.approx(float(np.dot(view, other.scoring(256))))


def test_flag_keeps_the_prefix_path(monkeypatch):
    monkeypatch.setenv(views.FLAG, "prefix")
    vector = carrier()
    assert vector.scoring(256) == prefix_scoring(vector, 256)
    assert views.arm_shipped() is None
    with pytest.raises(ValueError):
        vector.scoring(384)


def test_acceptance_falsifiers(tmp_path: Path):
    acceptance = json.loads(SHIPPED.read_text())
    weights = (SHIPPED.parent / acceptance["weights_file"]).read_bytes()

    def build(**updates):
        body = json.loads(json.dumps(acceptance))
        for key, value in updates.items():
            node = body
            parts = key.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
        return ViewHead(ViewHeadAcceptance.model_validate(body), weights)

    assert build().width == 256
    with pytest.raises(ValueError):
        build(**{"judge.passed": False})
    with pytest.raises(ValueError):
        build(**{"judge.arms.within_budget": False})
    with pytest.raises(ValueError):
        build(**{"judge.rows.candidate": 0.85})
    with pytest.raises(ValueError):
        build(budget_bytes=255)
    with pytest.raises(ValueError):
        build(view_bytes=512)
    with pytest.raises(ValueError):
        build(weights_sha256="0" * 64)
    with pytest.raises(ValueError):
        ViewHead(ViewHeadAcceptance.model_validate(acceptance), weights[:-4])
    bad = tmp_path / "head.json"
    bad.write_text(json.dumps({**acceptance, "weights_sha256": hashlib.sha256(b"x").hexdigest()}))
    (tmp_path / acceptance["weights_file"]).write_bytes(weights)
    with pytest.raises(ValueError):
        load_view_head(bad)


def test_activation_declares_its_effect_and_keeps_the_predecessor(tmp_path: Path):
    db = LocalDB(tmp_path / "carl.db")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    effect = f"Activate accepted view head 256 for {workspace.resolve()}"
    with pytest.raises(ValueError):
        activate_view_head(SHIPPED, workspace, effects=("Train local encoder frozen_heads",), db=db)
    assert db.get_config("view-head:" + views.content_hash(str(workspace.resolve()))) is None
    generation = activate_view_head(SHIPPED, workspace, effects=(effect,), db=db)
    key = "view-head:" + views.content_hash(str(workspace.resolve()))
    record = json.loads(db.get_config(key))
    assert record["current"]["weights_sha256"] == json.loads(SHIPPED.read_text())["weights_sha256"]
    assert record["current"]["judge_candidate"] == pytest.approx(0.9203125)
    assert record["predecessor"] is None
    assert activate_view_head(SHIPPED, workspace, effects=(effect,), db=db) == generation
    assert json.loads(db.get_config(key)) == record
    db.set_config(key, json.dumps({"current": {"width": 128, "weights_sha256": "0" * 64}, "predecessor": None}))
    activate_view_head(SHIPPED, workspace, effects=(effect,), db=db)
    assert json.loads(db.get_config(key))["predecessor"]["current"]["width"] == 128
    assert types.active_view_head(256, 768) is not None
    assert types.active_view_head(256, 384) is None


def test_bind_arms_the_head_and_recall_starts_at_its_width(monkeypatch, tmp_path: Path):
    from carl_studio.semantic.types import EncoderBinding, ExecutionBinding
    from carl_studio.session import Session

    monkeypatch.setenv("CARL_HOME", str(tmp_path))
    session = Session()
    binding = EncoderBinding(artifact_sha256="a" * 64, processor_sha256="b" * 64)
    execution = ExecutionBinding(
        interpreter="python", interpreter_sha256="c" * 64, dependencies={}, processor_sha256="b" * 64
    )
    seeds = {"query": 1, "near": 1, "far": 2}

    def encode(request):
        text = request.parts[0].text
        base = carrier(seeds[text])
        if text == "near":
            return Carrier(values=tuple(v + 0.05 for v in base.values))
        return base

    session.semantic.bind(binding, execution, encode)
    assert types.active_view_head(256, 768) is not None
    rows = session.semantic.recall("query", [("near", "near", 0.0), ("far", "far", 0.0)], limit=2)
    assert rows[0]["source_ref"] == "near"
    assert all(row["mode"] == "semantic" for row in rows)
    session.semantic.begin_turn()
    session.semantic.refinements = 2
    rows = session.semantic.recall("query", [("near", "near", 0.0), ("far", "far", 0.0)], limit=2)
    assert rows[0]["resolution"] == 256
    monkeypatch.setenv(views.FLAG, "prefix")
    types.clear_view_heads()
    session.semantic.begin_turn()
    session.semantic.refinements = 2
    rows = session.semantic.recall("query", [("near", "near", 0.0), ("far", "far", 0.0)], limit=2)
    assert rows[0]["resolution"] == 128
