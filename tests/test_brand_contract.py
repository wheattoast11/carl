"""The brand palette and authored artifact provenance stay consistent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carl_studio.theme import CARL_PALETTE, CARL_VOICE

ROOT = Path(__file__).resolve().parents[1]


def test_palette_maps_to_authored_diagram_roles() -> None:
    tokens = json.loads((ROOT / "assets/brand/tokens.json").read_text())
    assert CARL_PALETTE.primary == tokens["night_ink"]
    assert CARL_PALETTE.accent == tokens["night_attention"]
    assert CARL_PALETTE.success == tokens["night_confirmed"]
    assert CARL_PALETTE.secondary == tokens["night_secondary"]
    assert "acceptance" not in CARL_VOICE.eval_pass.lower()


def test_generated_artifacts_match_their_bound_source() -> None:
    root = ROOT / "assets/brand"
    descriptor = json.loads((root / "provenance.json").read_text())
    assert descriptor["performance_claim"] is False
    assert descriptor["fonts_redistributed"] is False
    assert (
        hashlib.sha256((ROOT / descriptor["source"]).read_bytes()).hexdigest()
        == descriptor["source_sha256"]
    )
    assert len(descriptor["files"]) == 12
    for name, expected in descriptor["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
