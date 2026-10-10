"""Accepted view heads: scoring moves from the 256 and 512 prefixes to the judged int8 views, prefix behind the flag."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from carl_studio.db import LocalDB
from carl_studio.semantic import types, views
from carl_studio.semantic.types import Carrier
from carl_studio.semantic.views import (
    SHIPPED,
    SHIPPED_256,
    SHIPPED_512,
    ViewHead,
    ViewHeadAcceptance,
    activate_view_head,
    load_view_head,
)


@pytest.fixture(autouse=True)
def reset_heads(monkeypatch):  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv(views.FLAG, raising=False)
    monkeypatch.delenv(views.WIDTH_FLAG, raising=False)
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


def expected_view(head: ViewHead, vector: Carrier) -> np.ndarray:
    raw = head.weights @ np.asarray(vector.values, dtype=np.float32)
    raw = raw / np.linalg.norm(raw)
    grid = np.round(raw / np.abs(raw).max() * 127) / 127
    return grid / np.linalg.norm(grid)


@pytest.mark.parametrize(("path", "width", "floor_score"), [(SHIPPED_256, 256, 0.92), (SHIPPED_512, 512, 0.99)])
def test_shipped_heads_are_accepted_and_replace_their_prefix(path: Path, width: int, floor_score: float):
    head = load_view_head(path)
    assert head.width == width and head.element_bytes == 1 and head.acceptance.view_bytes == width
    assert head.acceptance.view_bytes <= head.acceptance.budget_bytes == 512
    assert head.acceptance.judge["passed"] and head.acceptance.judge["rows"]["candidate"] >= floor_score
    vector = carrier()
    view = vector.scoring(width)
    assert len(view) == width
    assert abs(math.hypot(*view) - 1.0) < 1e-6
    assert view != prefix_scoring(vector, width)
    assert np.allclose(np.asarray(view), expected_view(head, vector), atol=1e-6)
    scaled = np.asarray(view) / np.abs(view).max() * 127
    assert np.allclose(scaled, np.round(scaled), atol=1e-3)
    for dimension in (128, 768):
        assert vector.scoring(dimension) == prefix_scoring(vector, dimension)
    assert vector.score(vector, width) == pytest.approx(1.0)
    other = carrier(4)
    assert vector.score(other, width) == pytest.approx(float(np.dot(view, other.scoring(width))))


def test_both_heads_arm_together_and_default_is_512():
    assert SHIPPED == SHIPPED_512
    armed = views.arm_shipped()
    assert sorted(head.width for head in armed) == [256, 512]
    assert views.default_view_width() == 512
    vector = carrier()
    assert vector.scoring(256) != prefix_scoring(vector, 256)
    assert vector.scoring(512) != prefix_scoring(vector, 512)
    assert types.active_view_head(256, 768).acceptance.training["loss"] == "kd"
    assert types.active_view_head(512, 768).acceptance.training["loss"] == "pca"


def test_flag_keeps_the_prefix_path(monkeypatch):
    monkeypatch.setenv(views.FLAG, "prefix")
    vector = carrier()
    assert vector.scoring(256) == prefix_scoring(vector, 256)
    assert vector.scoring(512) == prefix_scoring(vector, 512)
    assert views.arm_shipped() == []
    with pytest.raises(ValueError):
        vector.scoring(384)


def test_acceptance_falsifiers(tmp_path: Path):
    acceptance = json.loads(SHIPPED_256.read_text())
    weights = (SHIPPED_256.parent / acceptance["weights_file"]).read_bytes()

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


def test_a_failing_shipped_head_falls_back_to_its_prefix(monkeypatch, tmp_path: Path):
    good = json.loads(SHIPPED_512.read_text())
    (tmp_path / good["weights_file"]).write_bytes((SHIPPED_512.parent / good["weights_file"]).read_bytes())
    (tmp_path / "embeddinggemma2-view512.json").write_text(json.dumps({**good, "judge": {**good["judge"], "passed": False}}))
    monkeypatch.setattr(views, "HEADS", tmp_path)
    assert views.arm_shipped() == []
    vector = carrier()
    assert vector.scoring(512) == prefix_scoring(vector, 512)


def test_activation_declares_its_effect_and_keeps_the_predecessor(tmp_path: Path):
    db = LocalDB(tmp_path / "carl.db")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    effect = f"Activate accepted view head 512 for {workspace.resolve()}"
    with pytest.raises(ValueError):
        activate_view_head(SHIPPED, workspace, effects=("Train local encoder frozen_heads",), db=db)
    key = "view-head:" + views.content_hash(str(workspace.resolve()))
    assert db.get_config(key) is None
    generation = activate_view_head(SHIPPED, workspace, effects=(effect,), db=db)
    record = json.loads(db.get_config(key))
    assert record["current"]["weights_sha256"] == json.loads(SHIPPED.read_text())["weights_sha256"]
    assert record["current"]["judge_candidate"] == pytest.approx(0.993359375)
    assert record["predecessor"] is None
    assert activate_view_head(SHIPPED, workspace, effects=(effect,), db=db) == generation
    assert json.loads(db.get_config(key)) == record
    effect_256 = f"Activate accepted view head 256 for {workspace.resolve()}"
    activate_view_head(SHIPPED_256, workspace, effects=(effect_256,), db=db)
    assert json.loads(db.get_config(key))["predecessor"]["current"]["width"] == 512
    assert types.active_view_head(256, 768) is not None
    assert types.active_view_head(512, 768) is not None
    assert types.active_view_head(512, 384) is None


def test_bind_arms_the_heads_and_recall_starts_at_the_default_width(monkeypatch, tmp_path: Path):
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

    def resolution() -> int:
        session.semantic.begin_turn()
        session.semantic.refinements = 2
        rows = session.semantic.recall("query", [("near", "near", 0.0), ("far", "far", 0.0)], limit=2)
        assert rows[0]["source_ref"] == "near"
        assert all(row["mode"] == "semantic" for row in rows)
        return rows[0]["resolution"]

    session.semantic.bind(binding, execution, encode)
    assert types.active_view_head(512, 768) is not None and types.active_view_head(256, 768) is not None
    assert resolution() == 512
    monkeypatch.setenv(views.WIDTH_FLAG, "256")
    assert resolution() == 256
    monkeypatch.setenv(views.WIDTH_FLAG, "384")
    assert resolution() == 256
    monkeypatch.delenv(views.WIDTH_FLAG)
    monkeypatch.setenv(views.FLAG, "prefix")
    types.clear_view_heads()
    assert resolution() == 128
