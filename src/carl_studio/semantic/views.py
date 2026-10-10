"""Accepted linear view heads: judged 768->width int8 views replace the scoring prefix at their widths (rulings 60, 61, 63, 66)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from carl_core.hashing import content_hash
from pydantic import Field

from .types import Contract, register_view_head

HEADS = Path(__file__).resolve().parent / "heads"
SHIPPED_256 = HEADS / "embeddinggemma2-view256.json"
SHIPPED_512 = HEADS / "embeddinggemma2-view512.json"
SHIPPED = SHIPPED_512
FLAG = "CARL_VIEW_HEAD"
WIDTH_FLAG = "CARL_VIEW_WIDTH"
DEFAULT_WIDTH = 512


class ViewHeadAcceptance(Contract):
    """Judge receipt and bindings that admit one view head (field plan rulings 60, 61, 63)."""

    record_schema: str = Field(default="carl.view-head-acceptance/v1", alias="schema")
    space: dict[str, str]
    width: int = Field(ge=1)
    native_dimension: int = Field(ge=1)
    element_bytes: int
    budget_bytes: int = Field(ge=1)
    view_bytes: int = Field(ge=1)
    weights_file: str
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights_layout: str
    training: dict[str, Any]
    judge: dict[str, Any]
    anchor_set: dict[str, str]
    rank_arm: dict[str, str]
    rulings: dict[str, str]
    source: dict[str, str]


class ViewHead:
    """Row-major weights W (width x native); view = normalize(int8(normalize(W @ carrier)))."""

    def __init__(self, acceptance: ViewHeadAcceptance, weights: bytes) -> None:
        import numpy as np

        if acceptance.element_bytes not in (1, 2, 4):
            raise ValueError("View head element bytes must be 1, 2 or 4")
        if acceptance.view_bytes != acceptance.width * acceptance.element_bytes:
            raise ValueError("View bytes disagree with width and element bytes")
        if acceptance.view_bytes > acceptance.budget_bytes:
            raise ValueError("View head exceeds its byte budget")
        if hashlib.sha256(weights).hexdigest() != acceptance.weights_sha256:
            raise ValueError("View head weights do not match their acceptance")
        expected = acceptance.width * acceptance.native_dimension * 4
        if len(weights) != expected:
            raise ValueError("View head weights have the wrong shape")
        judge = acceptance.judge
        arms = judge.get("arms", {})
        rows = judge.get("rows", {})
        if not judge.get("passed") or not arms or not all(arms.values()):
            raise ValueError("View head judge did not pass")
        if rows.get("candidate", 0.0) < judge.get("floor", 1.0):
            raise ValueError("View head candidate is below the judge floor")
        if rows.get("view_width") != acceptance.width or rows.get("view_bytes") != acceptance.view_bytes:
            raise ValueError("Judged view does not match the accepted head")
        self.acceptance = acceptance
        self.width = acceptance.width
        self.native_dimension = acceptance.native_dimension
        self.element_bytes = acceptance.element_bytes
        self.weights = np.frombuffer(weights, dtype="<f4").reshape(self.width, self.native_dimension)

    @property
    def generation(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "element_bytes": self.element_bytes,
            "weights_sha256": self.acceptance.weights_sha256,
            "space": self.acceptance.space,
            "judge_candidate": self.acceptance.judge["rows"]["candidate"],
            "judge_margin": self.acceptance.judge["rows"]["margin_over_random_projection"],
            "rank_arm": self.acceptance.rank_arm,
            "source": self.acceptance.source,
        }

    def view(self, values: tuple[float, ...]) -> tuple[float, ...]:
        import numpy as np

        if len(values) != self.native_dimension:
            raise ValueError("View head expects the native carrier width")
        projected = self.weights @ np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(projected))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Collapsed view")
        projected = projected / norm
        if self.element_bytes == 1:
            scale = float(np.abs(projected).max()) or 1.0
            projected = np.round(projected / scale * 127) / 127
            projected = projected / float(np.linalg.norm(projected))
        elif self.element_bytes == 2:
            projected = projected.astype(np.float16).astype(np.float32)
            projected = projected / float(np.linalg.norm(projected))
        return tuple(float(v) for v in projected)


def enabled() -> bool:
    return os.environ.get(FLAG, "accepted").lower() not in {"prefix", "off", "0"}


def load_view_head(acceptance_path: Path) -> ViewHead:
    acceptance = ViewHeadAcceptance.model_validate_json(acceptance_path.read_text())
    return ViewHead(acceptance, (acceptance_path.parent / acceptance.weights_file).read_bytes())


def default_view_width() -> int:
    """Width recall starts at: 512 int8 (full budget) unless CARL_VIEW_WIDTH names another armed width."""
    raw = os.environ.get(WIDTH_FLAG, "")
    return int(raw) if raw.isdigit() else DEFAULT_WIDTH


def arm_shipped() -> list[ViewHead]:
    """Register every shipped accepted head; a head whose acceptance fails is skipped and the prefix stays at its width."""
    if not enabled():
        return []
    armed: list[ViewHead] = []
    for path in sorted(HEADS.glob("*.json")):
        try:
            head = load_view_head(path)
        except (ValueError, OSError):
            continue
        register_view_head(head)
        armed.append(head)
    return armed


def activate_view_head(
    acceptance_path: Path, workspace: str | os.PathLike[str], *, effects: tuple[str, ...], db: Any = None
) -> str:
    """Record the declared workspace effect with its predecessor, then arm the head."""
    head = load_view_head(acceptance_path)
    resolved = str(Path(workspace).resolve())
    effect = f"Activate accepted view head {head.width} for {resolved}"
    if effect not in effects:
        raise ValueError("View head activation was not declared")
    if db is None:
        from carl_studio.db import LocalDB

        db = LocalDB()
    key = "view-head:" + content_hash(resolved)
    prior = db.get_config(key)
    generation = head.generation
    if not (prior and json.loads(prior).get("current") == generation):
        db.set_config(key, json.dumps({"current": generation, "predecessor": json.loads(prior) if prior else None}))
    register_view_head(head)
    return content_hash(generation)
