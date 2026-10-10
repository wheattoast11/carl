"""Package extraction and exact model release admission."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from carl_core.encoder import Carrier
from carl_encoders.release import EncoderRelease, ReleaseArtifact
from carl_encoders.types import Carrier as PackageCarrier

from carl_studio.semantic.types import Carrier as StudioCarrier
from carl_studio.training.encoder import implementation_sources


def test_shared_contract_identity_and_prepared_sources() -> None:
    assert Carrier is PackageCarrier is StudioCarrier
    paths = implementation_sources()
    assert any(p.parent.name == "carl_encoders" and p.name == "worker.py" for p in paths)
    assert any(p.parent.name == "carl_encoders" and p.name == "fit_worker.py" for p in paths)
    assert any(p.parent.name == "carl_core" and p.name == "encoder.py" for p in paths)
    assert any(p.parent.name == "carl_core" and p.name == "encoder_settings.py" for p in paths)


def bundle(tmp_path: Path) -> EncoderRelease:
    artifacts = []
    for kind in ("weights", "heads", "processor", "notice"):
        path = tmp_path / kind
        path.write_bytes(kind.encode())
        artifacts.append(
            ReleaseArtifact(
                path=kind,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                kind=kind,
                provenance_ref="fixture-provenance",
            )
        )
    return EncoderRelease(
        version="0.1.0-rc.1",
        encoder_space_id="space",
        execution_id="execution",
        artifacts=tuple(artifacts),
    )


def test_exact_candidate_bytes_and_changed_source(tmp_path: Path) -> None:
    candidate = bundle(tmp_path)
    candidate.validate_artifacts(tmp_path)
    (tmp_path / "weights").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        candidate.validate_artifacts(tmp_path)


def test_accepted_and_gguf_require_separate_evidence(tmp_path: Path) -> None:
    candidate = bundle(tmp_path)
    with pytest.raises(ValueError, match="independent acceptance"):
        EncoderRelease.model_validate({**candidate.model_dump(), "stage": "accepted"})
    gguf = ReleaseArtifact(
        path="candidate.gguf", sha256="a" * 64, kind="gguf", provenance_ref="conversion"
    )
    with pytest.raises(ValueError, match="runtime qualification"):
        EncoderRelease.model_validate(
            {**candidate.model_dump(), "artifacts": (*candidate.artifacts, gguf)}
        )


def test_bundle_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the release"):
        ReleaseArtifact(
            path="../weights", sha256="a" * 64, kind="weights", provenance_ref="fixture"
        )
    candidate = bundle(tmp_path)
    outside = tmp_path.parent / "outside-model-fixture"
    outside.write_bytes(b"weights")
    (tmp_path / "weights").unlink()
    (tmp_path / "weights").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        candidate.validate_artifacts(tmp_path)
