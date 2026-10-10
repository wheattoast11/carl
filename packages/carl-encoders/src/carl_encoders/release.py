"""Immutable model artifact descriptors, separate from software releases."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .types import Contract


class ReleaseArtifact(Contract):
    """An exact bundle member with a separate provenance/notice reference."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal["weights", "heads", "processor", "notice", "evaluation", "gguf"]
    provenance_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def relative_member(self) -> ReleaseArtifact:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError("Artifact paths must remain inside the release bundle")
        return self


class EncoderRelease(Contract):
    """A model candidate descriptor; this does not authorize publication."""

    name: str = Field(default="CARL Encoder", min_length=1)
    version: str = Field(min_length=1)
    stage: Literal["candidate", "accepted"] = "candidate"
    encoder_space_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    predecessor_ref: str | None = None
    artifacts: tuple[ReleaseArtifact, ...] = Field(min_length=1)
    acceptance_ref: str | None = None
    gguf_qualification_ref: str | None = None

    @model_validator(mode="after")
    def complete_bundle(self) -> EncoderRelease:
        if len({a.path for a in self.artifacts}) != len(self.artifacts):
            raise ValueError("Duplicate model bundle member")
        if not {"weights", "heads", "processor", "notice"} <= {a.kind for a in self.artifacts}:
            raise ValueError("Weights, heads, processor and notices are required")
        evaluations = {a.path for a in self.artifacts if a.kind == "evaluation"}
        if self.stage == "accepted" and self.acceptance_ref not in evaluations:
            raise ValueError("Accepted releases require independent acceptance evidence")
        if (
            any(a.kind == "gguf" for a in self.artifacts)
            and self.gguf_qualification_ref not in evaluations
        ):
            raise ValueError("GGUF requires conversion and runtime qualification evidence")
        return self

    def validate_artifacts(self, root: Path) -> None:
        """Reject unavailable, escaping or changed members before consumption."""
        root = root.resolve()
        for artifact in self.artifacts:
            path = (root / artifact.path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Model artifact is unavailable or escapes its bundle")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise ValueError("Model artifact digest changed")
